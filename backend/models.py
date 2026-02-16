import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


class ParkingonsImageModel(nn.Module):
    """
    ResNet50-based Parkinson's image classifier with custom head.
    Upgraded from ResNet18 for significantly more capacity (25M vs 11M params).
    Uses a multi-layer classifier head with dropout for better generalization.
    """
    def __init__(self):
        super(ParkingonsImageModel, self).__init__()
        # Load pretrained ResNet50 with best available weights
        self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

        # Replace the final FC with a custom classifier head
        num_ftrs = self.model.fc.in_features  # 2048 for ResNet50
        self.model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, 2)
        )

    def forward(self, x):
        return self.model(x)

    def freeze_backbone(self):
        """Freeze all ResNet layers except the classifier head."""
        for name, param in self.model.named_parameters():
            if 'fc' not in name:
                param.requires_grad = False

    def unfreeze_layer4(self):
        """Unfreeze layer4 + classifier for fine-tuning phase."""
        for name, param in self.model.named_parameters():
            if 'layer4' in name or 'fc' in name:
                param.requires_grad = True

    def unfreeze_all(self):
        """Unfreeze all layers."""
        for param in self.model.parameters():
            param.requires_grad = True


def get_image_model(device):
    model = ParkingonsImageModel()
    return model.to(device)
