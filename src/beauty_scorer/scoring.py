"""End-to-end face scoring pipeline."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from .config import AppConfig
from .detector import FaceBox, FaceDetector
from .errors import AssetNotFoundError
from .image_utils import crop_square, imread
from .model import load_checkpoint
from .percentile import percentile_of

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


@dataclass(frozen=True)
class FaceScore:
    """A scored face."""

    box: FaceBox
    score: float
    percentile: float
    raw_score: float
    inference_ms: float

    def to_dict(self) -> dict:
        return {
            "box": {
                "x": self.box.x,
                "y": self.box.y,
                "width": self.box.width,
                "height": self.box.height,
                "confidence": self.box.confidence,
            },
            "score": round(self.score, 4),
            "percentile": round(self.percentile, 2),
            "raw_score": round(self.raw_score, 4),
            "inference_ms": round(self.inference_ms, 3),
        }


class BeautyScorer:
    """Detect faces and predict calibrated beauty scores."""

    def __init__(self, config: AppConfig | None = None) -> None:
        self.config = config or AppConfig()
        self.device = torch.device(self.config.resolve_device())

        self.detector = FaceDetector(self.config.detector)
        self.model, self._calibration_a, self._calibration_b = load_checkpoint(
            self.config.model.checkpoint_path, self.device
        )

        reference_path = Path(self.config.scoring.reference_path)
        if not reference_path.exists():
            raise AssetNotFoundError(f"reference distribution not found: {reference_path}")
        reference = json.loads(reference_path.read_text(encoding="utf-8"))
        population = self.config.scoring.population
        if population not in reference:
            raise ValueError(f"unknown population: {population}")
        self._reference = np.sort(np.asarray(reference[population], dtype=float))

        self._transform = transforms.Compose(
            [
                transforms.Resize(
                    (self.config.model.input_size, self.config.model.input_size)
                ),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )

        warmup = torch.zeros(
            1, 3, self.config.model.input_size, self.config.model.input_size,
            device=self.device,
        )
        with torch.no_grad():
            self.model(warmup)
        if self.device.type == "cuda":
            torch.cuda.synchronize()

    def score_image(self, image: np.ndarray, all_faces: bool = True) -> List[FaceScore]:
        """Score every detected face in a BGR image."""
        boxes = self.detector.detect(image, all_faces=all_faces)
        return [self._score_box(image, box) for box in boxes]

    def score_file(self, path: str | Path, all_faces: bool = True) -> List[FaceScore]:
        """Read an image from disk and score its faces."""
        return self.score_image(imread(path), all_faces=all_faces)

    def score_crop(self, crop: np.ndarray) -> Tuple[float, float, float, float]:
        """Score an already-cropped BGR face image.

        Returns ``(score, percentile, raw_score, inference_ms)``.
        """
        if crop is None or crop.size == 0:
            raise ValueError("empty crop")

        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        tensor = self._transform(Image.fromarray(rgb)).unsqueeze(0).to(self.device)

        if self.device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            raw = float(self.model(tensor).item())
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        score = float(np.clip(self._calibration_a * raw + self._calibration_b, 1.0, 5.0))
        percentile = float(percentile_of(score, self._reference))
        return score, percentile, raw, elapsed_ms

    def _score_box(self, image: np.ndarray, box: FaceBox) -> FaceScore:
        crop = crop_square(
            image, box.as_tuple(), margin=self.config.scoring.margin
        )
        score, percentile, raw, elapsed_ms = self.score_crop(crop)
        return FaceScore(
            box=box,
            score=score,
            percentile=percentile,
            raw_score=raw,
            inference_ms=elapsed_ms,
        )

    def annotate(
        self, image: np.ndarray, results: Sequence[FaceScore]
    ) -> np.ndarray:
        """Return a copy of the image with boxes and scores drawn."""
        canvas = image.copy()
        for result in results:
            x, y, w, h = result.box.as_tuple()
            cv2.rectangle(canvas, (x, y), (x + w, y + h), (0, 255, 0), 2)
            label = f"{result.score:.2f}  {result.percentile:.1f}%"
            cv2.putText(
                canvas,
                label,
                (x, max(20, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
        return canvas
