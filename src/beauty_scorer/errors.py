"""Package-specific exceptions."""

from __future__ import annotations


class BeautyScorerError(Exception):
    """Base class for all errors raised by this package."""


class AssetNotFoundError(BeautyScorerError):
    """Raised when a required asset (model, detector, reference) is missing."""


class DetectionError(BeautyScorerError):
    """Raised when face detection cannot be performed."""


class ImageReadError(BeautyScorerError):
    """Raised when an image cannot be decoded."""
