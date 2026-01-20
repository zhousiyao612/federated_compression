"""Model registry for CNNs and ViTs."""

from __future__ import annotations

import timm
import torch.nn as nn
from torchvision import models


def build_model(name: str, num_classes: int) -> nn.Module:
    lower = name.lower()
    if lower == "resnet18":
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    if lower == "densenet50":
        return timm.create_model("densenet50", num_classes=num_classes, pretrained=False)
    if lower in {"vit_small_patch16_224", "deit_small_patch16_224"}:
        return timm.create_model(lower, num_classes=num_classes, pretrained=False)
    raise ValueError(f"Unsupported model: {name}")
