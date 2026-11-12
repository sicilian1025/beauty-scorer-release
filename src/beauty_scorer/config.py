"""Configuration objects for the scoring pipeline."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

PACKAGE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = Path(os.environ.get("BEAUTY_SCORER_ASSETS", PACKAGE_DIR / "assets"))


def _as_path(value: Any) -> Path:
    return value if isinstance(value, Path) else Path(str(value))


@dataclass
class DetectorConfig:
    """Face detector settings."""

    model_path: Path = field(
        default_factory=lambda: ASSETS_DIR / "face_detection_yunet_2023mar.onnx"
    )
    score_threshold: float = 0.55
    nms_threshold: float = 0.3
    max_side: int = 1024
    top_k: int = 5000
    dedup_iou: float = 0.2
    dedup_ios: float = 0.6
    dedup_center: float = 0.2


@dataclass
class ModelConfig:
    """Beauty regression model settings."""

    checkpoint_path: Path = field(default_factory=lambda: ASSETS_DIR / "model.pt")
    input_size: int = 224
    device: str = "auto"


@dataclass
class ScoringConfig:
    """Score calibration and percentile settings."""

    reference_path: Path = field(default_factory=lambda: ASSETS_DIR / "reference.json")
    population: str = "asian"
    margin: float = 0.25


@dataclass
class AppConfig:
    """Top-level configuration."""

    detector: DetectorConfig = field(default_factory=DetectorConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)

    @classmethod
    def load(cls, path: Optional[str | Path] = None) -> "AppConfig":
        """Load a JSON config file, keeping defaults for omitted fields."""
        config = cls()
        if path is None:
            return config

        config_path = Path(path).resolve()
        data = json.loads(config_path.read_text(encoding="utf-8"))
        base = config_path.parent

        for name, obj in (
            ("detector", config.detector),
            ("model", config.model),
            ("scoring", config.scoring),
        ):
            section = data.get(name, {})
            if not isinstance(section, dict):
                raise ValueError(f"config section {name!r} must be an object")
            for key, value in section.items():
                if not hasattr(obj, key):
                    raise ValueError(f"unknown config key: {name}.{key}")
                if key.endswith("_path"):
                    value = _as_path(value)
                    if not value.is_absolute():
                        value = (base / value).resolve()
                setattr(obj, key, value)

        return config

    def resolve_device(self) -> str:
        """Resolve ``model.device`` to a concrete torch device string."""
        import torch

        requested = (self.model.device or "auto").lower()
        if requested == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        if requested.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        return requested

    def to_dict(self) -> Dict[str, Any]:
        return {
            "detector": {
                "model_path": str(self.detector.model_path),
                "score_threshold": self.detector.score_threshold,
                "nms_threshold": self.detector.nms_threshold,
                "max_side": self.detector.max_side,
                "top_k": self.detector.top_k,
                "dedup_iou": self.detector.dedup_iou,
                "dedup_ios": self.detector.dedup_ios,
                "dedup_center": self.detector.dedup_center,
            },
            "model": {
                "checkpoint_path": str(self.model.checkpoint_path),
                "input_size": self.model.input_size,
                "device": self.model.device,
            },
            "scoring": {
                "reference_path": str(self.scoring.reference_path),
                "population": self.scoring.population,
                "margin": self.scoring.margin,
            },
        }
