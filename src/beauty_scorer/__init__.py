"""Production-ready facial beauty scoring pipeline."""

from __future__ import annotations

__version__ = "1.0.0"

__all__ = ["BeautyScorer", "FaceScore", "AppConfig", "__version__"]


def __getattr__(name: str):
    if name in ("BeautyScorer", "FaceScore"):
        from .scoring import BeautyScorer, FaceScore

        return {"BeautyScorer": BeautyScorer, "FaceScore": FaceScore}[name]
    if name == "AppConfig":
        from .config import AppConfig

        return AppConfig
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
