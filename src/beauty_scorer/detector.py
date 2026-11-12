"""Face detection with duplicate suppression."""

from __future__ import annotations

import math
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import cv2
import numpy as np

from .config import DetectorConfig
from .errors import AssetNotFoundError, DetectionError

Detection = Tuple[float, float, float, float, float]


@dataclass(frozen=True)
class FaceBox:
    """A detected face box in original image coordinates."""

    x: float
    y: float
    width: float
    height: float
    confidence: float

    def as_tuple(self) -> Tuple[int, int, int, int]:
        return (
            int(round(self.x)),
            int(round(self.y)),
            int(round(self.width)),
            int(round(self.height)),
        )

    @property
    def center(self) -> Tuple[float, float]:
        return self.x + self.width / 2.0, self.y + self.height / 2.0

    @property
    def area(self) -> float:
        return self.width * self.height


def _ascii_safe_path(path: Path) -> Path:
    """OpenCV cannot load ONNX files from non-ASCII paths on Windows."""
    try:
        str(path).encode("ascii")
        return path
    except UnicodeEncodeError:
        cache = Path(tempfile.gettempdir()) / "beauty_scorer"
        cache.mkdir(parents=True, exist_ok=True)
        target = cache / path.name
        if not target.exists() or target.stat().st_size != path.stat().st_size:
            shutil.copy2(path, target)
        return target


def _iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax, ay, aw, ah = a[:4]
    bx, by, bw, bh = b[:4]
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def _ios(a: Sequence[float], b: Sequence[float]) -> float:
    ax, ay, aw, ah = a[:4]
    bx, by, bw, bh = b[:4]
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    smaller = min(aw * ah, bw * bh)
    return inter / smaller if smaller > 0 else 0.0


def _is_duplicate(
    a: Sequence[float],
    b: Sequence[float],
    iou_threshold: float,
    ios_threshold: float,
    center_threshold: float,
) -> bool:
    if _iou(a, b) > iou_threshold:
        return True
    if _ios(a, b) > ios_threshold:
        return True

    acx, acy = a[0] + a[2] / 2.0, a[1] + a[3] / 2.0
    bcx, bcy = b[0] + b[2] / 2.0, b[1] + b[3] / 2.0
    distance = math.hypot(acx - bcx, acy - bcy)
    max_side = max(a[2], a[3], b[2], b[3])
    return max_side > 0 and distance < center_threshold * max_side


def _dedup(
    detections: Iterable[Detection],
    iou_threshold: float = 0.2,
    ios_threshold: float = 0.6,
    center_threshold: float = 0.2,
) -> List[Detection]:
    """Merge duplicate detections with transitive clustering.

    Each cluster is replaced by the union of its boxes, keeping the highest
    confidence. This removes repeated detections of one face while keeping
    nearby but distinct faces separate.
    """
    dets = [d for d in detections if d[2] > 0 and d[3] > 0]
    n = len(dets)
    if n <= 1:
        return dets

    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    for i in range(n):
        for j in range(i + 1, n):
            if _is_duplicate(
                dets[i], dets[j], iou_threshold, ios_threshold, center_threshold
            ):
                union(i, j)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)

    merged: List[Detection] = []
    for indices in groups.values():
        x1 = min(dets[i][0] for i in indices)
        y1 = min(dets[i][1] for i in indices)
        x2 = max(dets[i][0] + dets[i][2] for i in indices)
        y2 = max(dets[i][1] + dets[i][3] for i in indices)
        confidence = max(dets[i][4] for i in indices)
        merged.append((x1, y1, x2 - x1, y2 - y1, confidence))

    return sorted(merged, key=lambda d: d[4], reverse=True)


class FaceDetector:
    """YuNet-based face detector."""

    def __init__(self, config: DetectorConfig | None = None) -> None:
        self.config = config or DetectorConfig()
        model_path = Path(self.config.model_path)
        if not model_path.exists():
            raise AssetNotFoundError(f"face detector not found: {model_path}")

        self._detector = cv2.FaceDetectorYN.create(
            str(_ascii_safe_path(model_path)),
            "",
            (320, 320),
            self.config.score_threshold,
            self.config.nms_threshold,
            self.config.top_k,
        )

    def detect(self, image: np.ndarray, all_faces: bool = True) -> List[FaceBox]:
        """Detect faces in a BGR image."""
        if image is None or image.size == 0:
            raise DetectionError("empty image")

        height, width = image.shape[:2]
        scale = 1.0
        working = image
        if self.config.max_side and max(height, width) > self.config.max_side:
            scale = self.config.max_side / float(max(height, width))
            working = cv2.resize(
                image,
                (int(round(width * scale)), int(round(height * scale))),
                interpolation=cv2.INTER_AREA,
            )

        self._detector.setInputSize((working.shape[1], working.shape[0]))
        _, faces = self._detector.detect(working)

        detections: List[Detection] = []
        if faces is not None:
            for face in faces:
                x, y, w, h, score = (
                    float(face[0]),
                    float(face[1]),
                    float(face[2]),
                    float(face[3]),
                    float(face[14]),
                )
                if w <= 0 or h <= 0:
                    continue
                detections.append(
                    (x / scale, y / scale, w / scale, h / scale, score)
                )

        detections = _dedup(
            detections,
            self.config.dedup_iou,
            self.config.dedup_ios,
            self.config.dedup_center,
        )
        if not all_faces:
            detections = detections[:1]

        return [FaceBox(*d) for d in detections]
