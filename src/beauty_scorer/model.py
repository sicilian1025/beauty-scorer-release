"""Beauty regression network."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18

from .errors import AssetNotFoundError

FEATURE_DIM = 512
HIDDEN_DIM = 256


class BeautyNet(nn.Module):
    """ResNet18 backbone with global average pooling and a regression head."""

    def __init__(self, backbone: nn.Module, head: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        pooled = F.adaptive_avg_pool2d(features, 1).flatten(1)
        return self.head(pooled).squeeze(1)


def build_model() -> BeautyNet:
    """Create the model architecture used by the released checkpoint."""
    resnet = resnet18(weights=None)
    backbone = nn.Sequential(*list(resnet.children())[:-2])
    head = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(FEATURE_DIM, HIDDEN_DIM),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(HIDDEN_DIM, 1),
    )
    return BeautyNet(backbone, head)


def load_checkpoint(
    path: str | Path, device: str | torch.device = "cpu"
) -> Tuple[BeautyNet, float, float]:
    """Load a checkpoint and its linear calibration parameters."""
    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        raise AssetNotFoundError(f"model checkpoint not found: {checkpoint_path}")

    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

    state = checkpoint.get("state_dict", checkpoint)
    model = build_model()
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    calibration = checkpoint.get("calibration", {})
    return (
        model,
        float(calibration.get("a", 1.0)),
        float(calibration.get("b", 0.0)),
    )


def count_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))
