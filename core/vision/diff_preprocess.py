"""Deterministic differential preprocessing and clamped ROI extraction for 640x640 PCB tiles."""

from __future__ import annotations

import cv2
import numpy as np

from core.agent.schemas import BOARD_DIM_PX, BoundingBoxDetection

_MORPH_KERNEL_3X3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))


def ensure_grayscale_640(img: np.ndarray, name: str = "image") -> np.ndarray:
    """Validate that `img` has 640x640 spatial dimensions and convert BGR to 2D uint8 grayscale."""
    if not isinstance(img, np.ndarray) or img.size == 0:
        raise ValueError(f"{name} must be a non-empty numpy ndarray.")

    if img.ndim == 2:
        height, width = img.shape
        gray = img
    elif img.ndim == 3:
        height, width, channels = img.shape
        if channels == 1:
            gray = img[:, :, 0]
        elif channels == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        elif channels == 4:
            gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        else:
            raise ValueError(
                f"{name} has unsupported channel count {channels}; expected 1, 3, or 4."
            )
    else:
        raise ValueError(f"{name} must be a 2D grayscale or 3D BGR array, got ndim={img.ndim}.")

    if (height, width) != (BOARD_DIM_PX, BOARD_DIM_PX):
        raise ValueError(
            f"{name} spatial dimensions must be ({BOARD_DIM_PX}, {BOARD_DIM_PX}), "
            f"got ({height}, {width})."
        )

    if gray.dtype != np.uint8:
        gray = np.clip(gray, 0, 255).astype(np.uint8)

    return gray


def compute_differential_map(
    template_img: np.ndarray,
    test_img: np.ndarray,
) -> np.ndarray:
    """Absolute difference of two 640×640 images, cleaned up and scaled to 0–255.

    Inputs are checked for size and converted to 8-bit grayscale. A 3×3 opening
    drops isolated pixels; a 3×3 closing joins neighbors. If opening removes the
    only signal, the raw difference is kept. Identical boards stay all zeros
    and are not rescaled.
    """
    template_gray = ensure_grayscale_640(template_img, name="template_img")
    test_gray = ensure_grayscale_640(test_img, name="test_img")

    raw_diff = cv2.absdiff(template_gray, test_gray)
    if int(raw_diff.max()) == 0:
        return raw_diff

    # Suppress isolated 1-pixel edge jitter, then bridge adjacent defect pixels
    opened = cv2.morphologyEx(raw_diff, cv2.MORPH_OPEN, _MORPH_KERNEL_3X3)
    cleaned = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, _MORPH_KERNEL_3X3)

    # Fall back to raw_diff if morphological opening erodes a very thin 1-2px feature
    active_map = cleaned if int(cleaned.max()) > 0 else raw_diff

    normalized = np.empty_like(active_map, dtype=np.uint8)
    cv2.normalize(active_map, normalized, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    return normalized


def crop_defect_roi(
    test_img: np.ndarray,
    bbox: BoundingBoxDetection,
    padding: int = 32,
) -> np.ndarray:
    """Crop local PCB circuit context around `bbox` with `padding` clamped safely to [0, 640]."""
    if not isinstance(test_img, np.ndarray) or test_img.size == 0:
        raise ValueError("test_img must be a non-empty numpy ndarray.")
    if test_img.ndim not in (2, 3):
        raise ValueError(f"test_img must be 2D or 3D, got ndim={test_img.ndim}.")

    height, width = test_img.shape[:2]
    if (height, width) != (BOARD_DIM_PX, BOARD_DIM_PX):
        raise ValueError(
            f"test_img spatial dimensions must be ({BOARD_DIM_PX}, {BOARD_DIM_PX}), "
            f"got ({height}, {width})."
        )
    if padding < 0:
        raise ValueError(f"padding must be >= 0, got {padding}.")

    x1_clamped = max(0, bbox.x1 - padding)
    y1_clamped = max(0, bbox.y1 - padding)
    x2_clamped = min(BOARD_DIM_PX, bbox.x2 + padding)
    y2_clamped = min(BOARD_DIM_PX, bbox.y2 + padding)

    return test_img[y1_clamped:y2_clamped, x1_clamped:x2_clamped].copy()
