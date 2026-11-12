"""Real-time video scoring."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque, List, Optional, Tuple

import cv2
import numpy as np

from .image_utils import crop_square, imwrite
from .scoring import BeautyScorer


def _iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


@dataclass
class _Track:
    box: Tuple[int, int, int, int]
    history: Deque[Tuple[float, float]] = field(default_factory=deque)

    def update(self, box: Tuple[int, int, int, int], score: float, pct: float, keep: int):
        self.box = box
        self.history.append((score, pct))
        while len(self.history) > keep:
            self.history.popleft()
        n = len(self.history)
        return (
            sum(s for s, _ in self.history) / n,
            sum(p for _, p in self.history) / n,
        )


def run_video(
    scorer: BeautyScorer,
    source: int | str = 0,
    max_faces: int = 3,
    smooth: int = 10,
    mirror: bool = False,
    display: bool = True,
    save_dir: Optional[str | Path] = None,
    max_frames: int = 0,
) -> int:
    """Run real-time scoring on a camera or video file.

    Returns the number of processed frames.
    """
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video source: {source}")

    if save_dir is not None:
        Path(save_dir).mkdir(parents=True, exist_ok=True)

    tracks: List[_Track] = []
    fps_history: Deque[float] = deque(maxlen=30)
    last_time = time.perf_counter()
    frames = 0

    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames += 1

        now = time.perf_counter()
        if frames > 1:
            fps_history.append(1.0 / max(1e-6, now - last_time))
        last_time = now

        if mirror:
            frame = cv2.flip(frame, 1)

        boxes = scorer.detector.detect(frame, all_faces=True)[:max_faces]

        next_tracks: List[_Track] = []
        for box in boxes:
            crop = crop_square(
                frame, box.as_tuple(), margin=scorer.config.scoring.margin
            )
            score, pct, _, _ = scorer.score_crop(crop)

            best, best_iou = None, 0.0
            for track in tracks:
                value = _iou(box.as_tuple(), track.box)
                if value > best_iou:
                    best, best_iou = track, value

            if best is not None and best_iou > 0.3:
                smooth_score, smooth_pct = best.update(
                    box.as_tuple(), score, pct, max(1, smooth)
                )
                next_tracks.append(best)
            else:
                track = _Track(box.as_tuple())
                smooth_score, smooth_pct = track.update(
                    box.as_tuple(), score, pct, max(1, smooth)
                )
                next_tracks.append(track)

            x, y, w, h = box.as_tuple()
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"{smooth_score:.2f}  {smooth_pct:.1f}%",
                (x, max(20, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
        tracks = next_tracks

        if fps_history:
            cv2.putText(
                frame,
                f"FPS {np.mean(fps_history):.1f}",
                (10, 26),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

        if display:
            cv2.imshow("beauty-scorer", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("s") and save_dir is not None:
                stamp = time.strftime("%Y%m%d_%H%M%S")
                imwrite(Path(save_dir) / f"frame_{stamp}.jpg", frame)

        if max_frames and frames >= max_frames:
            break

    capture.release()
    if display:
        cv2.destroyAllWindows()
    return frames
