from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple

import cv2
import numpy as np


@dataclass(frozen = True)
class WarpConfig:
    board_size: int = 1000
    dst_points: Tuple[Tuple[float, float], ...] = (
        (0.0, 0.0),     # top-left
        (999.0, 0.0),   # top-right
        (999.0, 999.0), # bottom-right
        (0.0, 999.0),   # bottom-left
    )


def _as_float32_points(pts: Iterable[Tuple[float, float]]) -> np.ndarray:
    arr = np.array(list(pts), dtype=np.float32)
    if arr.shape != (4, 2):
        raise ValueError(f"Expected 4 points with shape (4,2), got {arr.shape}")
    return arr


def order_points(pts: Iterable[Tuple[float, float]]) -> np.ndarray:
    arr = _as_float32_points(pts)

    s = arr.sum(axis=1)
    diff = np.diff(arr, axis=1).reshape(-1)

    tl = arr[np.argmin(s)]
    tr = arr[np.argmin(diff)]
    br = arr[np.argmax(s)]
    bl = arr[np.argmax(diff)]

    return np.array([tl, tr, br, bl], dtype=np.float32)


def compute_homography(
    src_points: Iterable[Tuple[float, float]],
    cfg: WarpConfig = WarpConfig(),
) -> np.ndarray:
    
    src = order_points(src_points)
    dst = _as_float32_points(cfg.dst_points)

    H = cv2.getPerspectiveTransform(src, dst) # 3x3 matrix
    return H


def warp_to_board(
    image_bgr: np.ndarray,
    H: np.ndarray,
    cfg: WarpConfig = WarpConfig(),
) -> np.ndarray:
    
    size = (cfg.board_size, cfg.board_size) # (width, height)
    warped = cv2.warpPerspective(image_bgr, H, size, flags=cv2.INTER_LINEAR)
    return warped


def warp_from_corners(
    image_bgr: np.ndarray,
    corners_src: Iterable[Tuple[float, float]],
    cfg: WarpConfig = WarpConfig(),
) -> Tuple[np.ndarray, np.ndarray]:
    
    H = compute_homography(corners_src, cfg)
    warped = warp_to_board(image_bgr, H, cfg)
    return warped, H


def load_image(path: str | Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img
