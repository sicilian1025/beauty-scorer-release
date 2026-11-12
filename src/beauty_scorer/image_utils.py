"""Image I/O and cropping helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import cv2
import numpy as np

from .errors import ImageReadError

Box = Tuple[int, int, int, int]


def imread(path: str | Path) -> np.ndarray:
    """Read an image from a path that may contain non-ASCII characters."""
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        raise ImageReadError(f"empty file: {path}")
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ImageReadError(f"cannot decode image: {path}")
    return image


def imwrite(path: str | Path, image: np.ndarray) -> None:
    """Write an image to a path that may contain non-ASCII characters."""
    suffix = Path(path).suffix or ".jpg"
    ok, buffer = cv2.imencode(suffix, image)
    if not ok:
        raise ImageReadError(f"cannot encode image: {path}")
    buffer.tofile(str(path))


def crop_square(image: np.ndarray, box: Box, margin: float = 0.25) -> np.ndarray:
    """Crop a square region around a face box, padding if necessary."""
    x, y, w, h = box
    if w <= 0 or h <= 0:
        raise ValueError(f"invalid box: {box}")

    cx, cy = x + w / 2.0, y + h / 2.0
    side = max(w, h) * (1.0 + 2.0 * margin)
    x0 = int(round(cx - side / 2.0))
    y0 = int(round(cy - side / 2.0))
    x1 = int(round(cx + side / 2.0))
    y1 = int(round(cy + side / 2.0))

    pad_l = max(0, -x0)
    pad_t = max(0, -y0)
    pad_r = max(0, x1 - image.shape[1])
    pad_b = max(0, y1 - image.shape[0])
    if pad_l or pad_t or pad_r or pad_b:
        image = cv2.copyMakeBorder(
            image, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_REFLECT_101
        )
        x0 += pad_l
        x1 += pad_l
        y0 += pad_t
        y1 += pad_t
    return image[y0:y1, x0:x1]


def center_square(image: np.ndarray, margin: float = 0.0) -> np.ndarray:
    """Return the largest centered square crop."""
    h, w = image.shape[:2]
    side = int(round(min(w, h) * (1.0 + 2.0 * margin)))
    cx, cy = w // 2, h // 2
    x0 = max(0, cx - side // 2)
    y0 = max(0, cy - side // 2)
    return image[y0 : y0 + side, x0 : x0 + side]


def fit_display(image: np.ndarray, max_side: int = 900) -> np.ndarray:
    """Downscale an image for display while preserving aspect ratio."""
    h, w = image.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale >= 1.0:
        return image
    return cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
