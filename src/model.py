import torch
import torch.nn as nn
from torchvision import models


class ArtworkCNN(nn.Module):
    def __init__(self, num_classes, variant="b1", dropout=0.3, freeze_backbone=True):
        super().__init__()
        self.num_classes = num_classes
        self.variant = variant

        if variant == "b0":
            weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
            self.backbone = models.efficientnet_b0(weights=weights)
        elif variant == "b1":
            weights = models.EfficientNet_B1_Weights.IMAGENET1K_V1
            self.backbone = models.efficientnet_b1(weights=weights)
        else:
            raise ValueError("variant must be 'b0' or 'b1'")

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)

    def unfreeze_top_layers(self):
        for p in self.backbone.features[-2:].parameters():
            p.requires_grad = True

    def unfreeze_all(self):
        for p in self.backbone.parameters():
            p.requires_grad = True


def save_model(model, path, epoch=None, val_acc=None, artist_names=None):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "num_classes": model.num_classes,
        "variant": model.variant,
    }
    if epoch is not None:
        checkpoint["epoch"] = epoch
    if val_acc is not None:
        checkpoint["val_acc"] = val_acc
    if artist_names is not None:
        checkpoint["artist_names"] = artist_names
    torch.save(checkpoint, path)


def load_model(path, device="cpu"):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = ArtworkCNN(
        num_classes=checkpoint["num_classes"],
        variant=checkpoint.get("variant", "b1"),
        freeze_backbone=False
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint.get("artist_names", [])