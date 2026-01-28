import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


import cv2
from time import sleep

from muehle_game import Muehle
from imagedetection.detector import Detector
from runtime.camera_provider import get_frame_bgr, close_camera


def main():
    detector = Detector(
        board_indices_csv="assets/indices/board_indices.csv",
        board_model_path="assets/models/board_best.pt",
        stones_model_path="assets/models/stones_best.pt",
        stacks_model_path="assets/models/stacks_best.pt",
        conf_min=0.25,
        conf_accept=0.45,
        dist_max_factor=0.7,
        black_cls_id=0,
        white_cls_id=1,
        frame_provider=get_frame_bgr,
    )

    game = Muehle()
    print("Starte Human-Input-Test. Mach einen Zug am Brett. (q = quit)")

    try:
        while True:
            try:
                out = detector.get_next_gamestate(
                    game,
                    stable_n=3,
                    max_frames=120,
                    sleep_s=0.05,
                )
            except TimeoutError:
                continue
            except RuntimeError as e:
                print("IGNORED:", e)
                continue
            except ValueError as e:
                print("ILLEGAL (game rule):", e)
                continue

            print("delta:", out["delta"])
            print("board:", game.board.astype(int).tolist())


            #vis = out["vision"]["debug_vis"]
            #cv2.imshow("debug_vis", vis)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break

            sleep(0.2)

    finally:
        close_camera()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
