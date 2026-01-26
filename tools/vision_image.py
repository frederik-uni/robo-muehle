import argparse
import cv2
from imagedetection.detector import Detector

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    args = ap.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        raise SystemExit(f"Could not read image: {args.image}")

    det = Detector(
        board_indices_csv="assets/indices/board_indices.csv",
        board_model_path="assets/models/board_best.pt",
        stones_model_path="assets/models/stones_best.pt",
        stacks_model_path="assets/models/stacks_best.pt",
        conf_min=0.25,
        conf_accept=0.45,
        dist_max_factor=0.7,
        black_cls_id=0,
        white_cls_id=1,
    )

    state, info = det.get_gamestate(img)
    print("state_arr:", info["state_arr"])

    cv2.imshow("debug_vis (orig | warp)", info["debug_vis"])
    cv2.waitKey(0)

if __name__ == "__main__":
    main()
