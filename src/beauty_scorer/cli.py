"""Command-line interface."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Sequence

from . import __version__
from .config import AppConfig
from .errors import BeautyScorerError
from .image_utils import imread, imwrite
from .scoring import BeautyScorer

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _collect_images(inputs: Sequence[str]) -> List[Path]:
    paths: List[Path] = []
    for item in inputs:
        path = Path(item)
        if path.is_dir():
            paths.extend(
                sorted(
                    p
                    for p in path.rglob("*")
                    if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
                )
            )
        elif path.is_file():
            paths.append(path)
        else:
            raise FileNotFoundError(f"not found: {path}")
    return paths


def _print_table(rows: List[dict]) -> None:
    print(f"{'image':<42}{'face':>6}{'score':>8}{'pct':>9}{'ms':>8}")
    print("-" * 73)
    for row in rows:
        name = row["image"]
        if len(name) > 40:
            name = name[:37] + "..."
        print(
            f"{name:<42}{row['face']:>6}{row['score']:>8.2f}"
            f"{row['percentile']:>8.1f}%{row['inference_ms']:>8.2f}"
        )


def cmd_score(args: argparse.Namespace) -> int:
    config = AppConfig.load(args.config)
    if args.population:
        config.scoring.population = args.population
    if args.device:
        config.model.device = args.device

    scorer = BeautyScorer(config)
    images = _collect_images(args.input)
    if not images:
        print("no images found", file=sys.stderr)
        return 1

    output_dir = Path(args.output_dir)
    if not args.no_save:
        output_dir.mkdir(parents=True, exist_ok=True)

    rows: List[dict] = []
    payload: List[dict] = []

    for path in images:
        try:
            image = imread(path)
            results = scorer.score_image(image, all_faces=not args.single_face)
        except BeautyScorerError as exc:
            print(f"[skip] {path}: {exc}", file=sys.stderr)
            continue

        if not results:
            print(f"[skip] {path}: no face detected", file=sys.stderr)
            continue

        for index, result in enumerate(results):
            rows.append(
                {
                    "image": str(path),
                    "face": index + 1,
                    "score": result.score,
                    "percentile": result.percentile,
                    "inference_ms": result.inference_ms,
                }
            )
            entry = result.to_dict()
            entry["image"] = str(path)
            entry["face_index"] = index
            payload.append(entry)

        if not args.no_save:
            annotated = scorer.annotate(image, results)
            target = output_dir / f"{path.stem}_scored.jpg"
            imwrite(target, annotated)

    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        _print_table(rows)
        if not args.no_save:
            print(f"\nannotated images: {output_dir}")

    return 0


def cmd_video(args: argparse.Namespace) -> int:
    from .video import run_video

    config = AppConfig.load(args.config)
    if args.device:
        config.model.device = args.device
    scorer = BeautyScorer(config)
    source = int(args.source) if str(args.source).isdigit() else args.source
    frames = run_video(
        scorer,
        source=source,
        max_faces=args.max_faces,
        smooth=args.smooth,
        mirror=args.mirror,
        display=not args.no_display,
        save_dir=args.save_dir,
        max_frames=args.max_frames,
    )
    print(f"processed {frames} frames")
    return 0


def cmd_info(args: argparse.Namespace) -> int:
    import torch

    from .model import count_parameters, load_checkpoint

    config = AppConfig.load(args.config)
    device = config.resolve_device()
    model, a, b = load_checkpoint(config.model.checkpoint_path, device)
    print(f"beauty-scorer {__version__}")
    print(f"device: {device}")
    print(f"torch: {torch.__version__}")
    print(f"parameters: {count_parameters(model) / 1e6:.2f}M")
    print(f"calibration: a={a:.4f} b={b:.4f}")
    print(f"detector: {config.detector.model_path}")
    print(f"checkpoint: {config.model.checkpoint_path}")
    print(f"reference: {config.scoring.reference_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="beauty-score",
        description="Facial beauty scoring with percentile output.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    score = subparsers.add_parser("score", help="score images or a folder")
    score.add_argument("--input", nargs="+", required=True, help="image files or folders")
    score.add_argument("--output-dir", default="outputs", help="folder for annotated images")
    score.add_argument("--config", default=None, help="path to a JSON config")
    score.add_argument("--population", choices=["asian", "all", "caucasian"], default=None)
    score.add_argument("--device", default=None, help="auto, cpu, cuda, cuda:0, ...")
    score.add_argument("--single-face", action="store_true", help="keep only the top face")
    score.add_argument("--no-save", action="store_true", help="do not write annotated images")
    score.add_argument("--json", action="store_true", help="print JSON instead of a table")
    score.set_defaults(func=cmd_score)

    video = subparsers.add_parser("video", help="real-time camera or video scoring")
    video.add_argument("--source", default="0", help="camera index or video path")
    video.add_argument("--config", default=None)
    video.add_argument("--device", default=None)
    video.add_argument("--max-faces", type=int, default=3)
    video.add_argument("--smooth", type=int, default=10)
    video.add_argument("--mirror", action="store_true")
    video.add_argument("--no-display", action="store_true")
    video.add_argument("--save-dir", default=None, help="where to save snapshots")
    video.add_argument("--max-frames", type=int, default=0)
    video.set_defaults(func=cmd_video)

    info = subparsers.add_parser("info", help="show model and device information")
    info.add_argument("--config", default=None)
    info.set_defaults(func=cmd_info)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (BeautyScorerError, FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
