import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))



from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from src.homography import compute_homography, warp_to_board, WarpConfig
from src.vision.board_bbox import best_board_box


def clamp_box(x1, y1, x2, y2, w, h, margin=30):
    x1 = max(0, x1 - margin)
    y1 = max(0, y1 - margin)
    x2 = min(w - 1, x2 + margin)
    y2 = min(h - 1, y2 + margin)
    return x1, y1, x2, y2


def export_warp_bbox(
    model: YOLO,
    img_path: Path,
    out_path: Path,
    cfg: WarpConfig = WarpConfig(),
) -> bool:
    img = cv2.imread(str(img_path))
    if img is None:
        return False

    results = model(img)
    box = best_board_box(results[0])
    if box is None:
        return False

    h, w = img.shape[:2]
    x1, y1, x2, y2 = clamp_box(*box, w, h, margin=30)

    corners = np.array(
            [(x1, y1), (x2, y1), (x2, y2), (x1, y2)],
            dtype=np.float32
        )

    H = compute_homography(corners, cfg)
    warped = warp_to_board(img, H, cfg)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), warped)
    return True


if __name__ == "__main__":
    model = YOLO("runs/detect/board/weights/best.pt")

    in_dir = Path("tools/toWarp")
    out_dir = Path("tools/results")

    cfg = WarpConfig(board_size=1000)

    exts = {".jpg", ".jpeg", ".png", ".webp"}
    for p in sorted(in_dir.glob("*")):
        if p.suffix.lower() not in exts:
            continue
        out = out_dir / f"{p.stem}.jpg"
        ok = export_warp_bbox(model, p, out, cfg)
        print(p.name, "->", "OK" if ok else "FAIL")