import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import sys
import copy
import numpy as np

# Add project root to sys.path to allow imports from backend module
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from backend.models import ParkingonsImageModel
from backend.utils import get_device, get_num_workers


def mixup_data(x, y, alpha=0.2):
    """MixUp augmentation: blends pairs of images and labels for regularization."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """MixUp loss: weighted combination of losses for both mixed labels."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def train_image_model():
    print("=" * 60)
    print("  PARKINSON'S IMAGE MODEL TRAINING (Enhanced)")
    print("=" * 60)

    device = get_device()
    print(f"Using device: {device}")

    # --- Heavy data augmentation for training (critical for ~72 images) ---
    train_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(15),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.15)),
    ])

    # Validation transforms: no augmentation, just resize + normalize
    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Dataset paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, 'datasets', 'drawings')
    spiral_dir = os.path.join(data_dir, 'spiral')

    train_dir = os.path.join(spiral_dir, 'training')
    test_dir = os.path.join(spiral_dir, 'testing')

    if not os.path.exists(train_dir):
        print(f"Dataset not found at {train_dir}.")
        return

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    val_dataset = datasets.ImageFolder(test_dir, transform=val_transforms)

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Classes: {train_dataset.classes}")

    # Compute class weights for imbalanced data
    class_counts = [0] * len(train_dataset.classes)
    for _, label in train_dataset.samples:
        class_counts[label] += 1
    total = sum(class_counts)
    class_weights = torch.tensor(
        [total / (len(class_counts) * c) for c in class_counts],
        dtype=torch.float32
    ).to(device)
    print(f"Class weights: {dict(zip(train_dataset.classes, class_weights.tolist()))}")

    num_workers = get_num_workers()
    print(f"Using {num_workers} workers for data loading")

    train_loader = DataLoader(
        train_dataset, batch_size=16, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=16, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    # --- Model setup ---
    model = ParkingonsImageModel().to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # ================================================================
    #  PHASE 1: Train only the classifier head (backbone frozen)
    # ================================================================
    print(f"\n{'=' * 60}")
    print(f"  PHASE 1: Training classifier head (backbone frozen)")
    print(f"{'=' * 60}")

    model.freeze_backbone()
    head_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(head_params, lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)

    phase1_epochs = 15
    best_acc = 0.0
    best_model_state = None

    for epoch in range(phase1_epochs):
        model.train()
        running_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Apply MixUp augmentation
            mixed_inputs, y_a, y_b, lam = mixup_data(inputs, labels, alpha=0.2)

            optimizer.zero_grad()
            outputs = model(mixed_inputs)
            loss = mixup_criterion(criterion, outputs, y_a, y_b, lam)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        scheduler.step()

        # Validation
        val_acc, val_loss = evaluate(model, val_loader, criterion, device)
        avg_train_loss = running_loss / len(train_loader)
        lr = optimizer.param_groups[0]['lr']
        print(f"  Epoch {epoch+1:2d}/{phase1_epochs} | "
              f"Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Val Acc: {val_acc:.1f}% | LR: {lr:.6f}")

        if val_acc > best_acc:
            best_acc = val_acc
            best_model_state = copy.deepcopy(model.state_dict())

    print(f"  Phase 1 Best Val Accuracy: {best_acc:.1f}%")

    # ================================================================
    #  PHASE 2: Fine-tune layer4 + head with lower learning rate
    # ================================================================
    print(f"\n{'=' * 60}")
    print(f"  PHASE 2: Fine-tuning layer4 + classifier head")
    print(f"{'=' * 60}")

    # Load best Phase 1 model
    model.load_state_dict(best_model_state)
    model.unfreeze_layer4()

    # Differential learning rates: backbone layers get lower LR
    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'fc' in name:
                head_params.append(param)
            else:
                backbone_params.append(param)

    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': 1e-5},
        {'params': head_params, 'lr': 5e-4}
    ], weight_decay=1e-4)

    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    phase2_epochs = 40
    patience = 12
    epochs_no_improve = 0

    for epoch in range(phase2_epochs):
        model.train()
        running_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            mixed_inputs, y_a, y_b, lam = mixup_data(inputs, labels, alpha=0.2)

            optimizer.zero_grad()
            outputs = model(mixed_inputs)
            loss = mixup_criterion(criterion, outputs, y_a, y_b, lam)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item()

        scheduler.step()

        # Validation
        val_acc, val_loss = evaluate(model, val_loader, criterion, device)
        avg_train_loss = running_loss / len(train_loader)
        lr_bb = optimizer.param_groups[0]['lr']
        lr_hd = optimizer.param_groups[1]['lr']
        print(f"  Epoch {epoch+1:2d}/{phase2_epochs} | "
              f"Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Val Acc: {val_acc:.1f}% | LR: bb={lr_bb:.2e}, hd={lr_hd:.2e}")

        if val_acc > best_acc:
            best_acc = val_acc
            best_model_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"  Early stopping at epoch {epoch + 1}")
            break

    print(f"\n  Phase 2 Best Val Accuracy: {best_acc:.1f}%")

    # --- Final evaluation ---
    model.load_state_dict(best_model_state)
    final_acc, final_loss = evaluate(model, val_loader, criterion, device, verbose=True)
    print(f"\n{'=' * 60}")
    print(f"  FINAL BEST VALIDATION ACCURACY: {best_acc:.1f}%")
    print(f"{'=' * 60}")

    # Save model
    os.makedirs('tensors', exist_ok=True)
    torch.save(best_model_state, 'tensors/spiral_model.pth')
    print(f"Model saved to tensors/spiral_model.pth")


def evaluate(model, loader, criterion, device, verbose=False):
    """Evaluate model on a data loader, returns (accuracy, avg_loss)."""
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = 100.0 * correct / total
    avg_loss = running_loss / len(loader)

    if verbose:
        from sklearn.metrics import classification_report
        print("\nDetailed Classification Report:")
        print(classification_report(all_labels, all_preds,
                                    target_names=['healthy', 'parkinson']))

    return accuracy, avg_loss


if __name__ == "__main__":
    train_image_model()
