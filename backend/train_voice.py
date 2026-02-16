import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import os
import sys
import copy

# Add project root to sys.path to allow imports from backend module
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from backend.utils import get_device


class TabularVoiceDataset(Dataset):
    def __init__(self, X, y, augment=False):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.augment = augment

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx].clone()
        if self.augment:
            # Gaussian noise injection for data augmentation
            noise = torch.randn_like(x) * 0.05
            x = x + noise
            # Random feature dropout (zero out ~10% of features)
            mask = torch.rand(x.shape) > 0.1
            x = x * mask.float()
        return x, self.y[idx]


class VoiceClassifier(nn.Module):
    """
    Deep 4-hidden-layer classifier with BatchNorm and Dropout.
    Architecture: 22 -> 256 -> 128 -> 64 -> 32 -> 2
    ~10x more parameters than the original (64 -> 32 -> 2).
    """
    def __init__(self, input_dim):
        super(VoiceClassifier, self).__init__()
        self.network = nn.Sequential(
            # Layer 1: input -> 256
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.4),

            # Layer 2: 256 -> 128
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.4),

            # Layer 3: 128 -> 64
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),

            # Layer 4: 64 -> 32
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),

            # Output: 32 -> 2
            nn.Linear(32, 2)
        )

    def forward(self, x):
        return self.network(x)


def train_voice_model():
    print("=" * 60)
    print("  PARKINSON'S VOICE MODEL TRAINING (Enhanced)")
    print("=" * 60)

    # Load data
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Priority 1: Custom Dataset
    custom_data_path = os.path.join(current_dir, 'datasets', 'voice', 'custom_data.csv')
    
    # Check if we should auto-build the dataset
    raw_healthy = os.path.join(current_dir, 'datasets', 'voice', 'raw', 'healthy')
    raw_parkinson = os.path.join(current_dir, 'datasets', 'voice', 'raw', 'parkinson')
    
    # If custom CSV doesn't exist, but raw files DO exist, build it automatically!
    if not os.path.exists(custom_data_path):
        has_raw_files = False
        if os.path.exists(raw_healthy) and len(os.listdir(raw_healthy)) > 0:
            has_raw_files = True
        if os.path.exists(raw_parkinson) and len(os.listdir(raw_parkinson)) > 0:
            has_raw_files = True
            
        if has_raw_files:
            print("Found raw .wav files but no dataset CSV. Building dataset automatically...")
            # Import dynamically to avoid circular imports if any
            try:
                from build_voice_dataset import build_dataset
                build_dataset()
            except ImportError:
                # In case running from different cwd
                sys.path.append(current_dir)
                from build_voice_dataset import build_dataset
                build_dataset()

    # Priority 2: Original UCI Dataset
    uci_data_path = os.path.join(current_dir, 'datasets', 'voice', 'parkinsons.data')

    if os.path.exists(custom_data_path):
        print(f"Loading CUSTOM dataset from: {custom_data_path}")
        df = pd.read_csv(custom_data_path)
    elif os.path.exists(uci_data_path):
        print(f"Loading UCI dataset from: {uci_data_path}")
        df = pd.read_csv(uci_data_path)
    else:
        print(f"No data found! Please run build_voice_dataset.py or check paths.")
        return

    feature_cols = [c for c in df.columns if c not in ['name', 'status']]
    X = df[feature_cols].values
    y = df['status'].values

    print(f"Features: {len(feature_cols)}")
    print(f"Total samples: {len(X)}")
    print(f"Class distribution: Healthy={np.sum(y == 0)}, Parkinson={np.sum(y == 1)}")

    # Compute class weights for imbalanced data (147 PD vs 48 Healthy)
    class_counts = np.bincount(y)
    total = len(y)
    class_weights = torch.tensor(
        [total / (2.0 * c) for c in class_counts], dtype=torch.float32
    )
    print(f"Class weights: Healthy={class_weights[0]:.3f}, Parkinson={class_weights[1]:.3f}")

    device = get_device()
    print(f"Device: {device}")
    print()

    # --- Stratified 5-Fold Cross Validation ---
    n_folds = 5
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    best_overall_acc = 0.0
    best_model_state = None
    best_scaler = None
    fold_accuracies = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n{'-' * 50}")
        print(f"  FOLD {fold + 1}/{n_folds}")
        print(f"{'-' * 50}")

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Scale per fold
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        # Create datasets with augmentation on training set
        train_dataset = TabularVoiceDataset(X_train_scaled, y_train, augment=True)
        val_dataset = TabularVoiceDataset(X_val_scaled, y_val, augment=False)

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

        # Model, loss, optimizer
        model = VoiceClassifier(input_dim=len(feature_cols)).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=10
        )

        # Training with early stopping
        epochs = 300
        patience = 30
        best_fold_acc = 0.0
        best_fold_state = None
        epochs_no_improve = 0

        for epoch in range(epochs):
            # --- Train ---
            model.train()
            train_loss = 0.0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += loss.item()

            # --- Validate ---
            model.eval()
            correct = 0
            total = 0
            val_loss = 0.0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            val_acc = 100.0 * correct / total
            scheduler.step(val_acc)

            if (epoch + 1) % 50 == 0 or epoch == 0:
                avg_train_loss = train_loss / len(train_loader)
                avg_val_loss = val_loss / len(val_loader)
                lr = optimizer.param_groups[0]['lr']
                print(f"  Epoch {epoch+1:3d} | Train Loss: {avg_train_loss:.4f} | "
                      f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.1f}% | LR: {lr:.6f}")

            # Track best model for this fold
            if val_acc > best_fold_acc:
                best_fold_acc = val_acc
                best_fold_state = copy.deepcopy(model.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                print(f"  Early stopping at epoch {epoch + 1}")
                break

        fold_accuracies.append(best_fold_acc)
        print(f"  Fold {fold + 1} Best Accuracy: {best_fold_acc:.1f}%")

        # Track best model across all folds
        if best_fold_acc > best_overall_acc:
            best_overall_acc = best_fold_acc
            best_model_state = best_fold_state
            best_scaler = copy.deepcopy(scaler)

    # --- Summary ---
    print(f"\n{'=' * 60}")
    print(f"  CROSS-VALIDATION RESULTS")
    print(f"{'=' * 60}")
    for i, acc in enumerate(fold_accuracies):
        print(f"  Fold {i + 1}: {acc:.1f}%")
    print(f"  Mean: {np.mean(fold_accuracies):.1f}% ± {np.std(fold_accuracies):.1f}%")
    print(f"  Best:  {best_overall_acc:.1f}%")

    # --- Final evaluation with best model ---
    model.load_state_dict(best_model_state)
    model.eval()

    # Re-evaluate on a fresh split with the best scaler for detailed report
    from sklearn.model_selection import train_test_split
    X_final_train, X_final_test, y_final_train, y_final_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_final_test_scaled = best_scaler.transform(X_final_test)
    test_dataset = TabularVoiceDataset(X_final_test_scaled, y_final_test, augment=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print(f"\nFinal Test Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=['Healthy', 'Parkinson']))

    # --- Save best model and scaler ---
    os.makedirs('tensors', exist_ok=True)
    torch.save(best_model_state, 'tensors/voice_model_tabular.pth')
    print("Best model saved to tensors/voice_model_tabular.pth")

    import joblib
    joblib.dump(best_scaler, 'tensors/voice_scaler.save')
    print("Best scaler saved to tensors/voice_scaler.save")


if __name__ == "__main__":
    train_voice_model()
