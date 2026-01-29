from ai.play import load_model
from imagedetection.detector import Detector
from piecewalker.ned2 import Ned2
from runtime import game_loop
from imagedetection.detector import Detector

from .args import parse_args

from runtime.camera_provider import get_frame_bgr

from .args import parse_args

if __name__ == "__main__":
    args = parse_args()
    ned2 = Ned2()
    ai = load_model(args.model_play)
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
    game_loop.run_game_loop(ned2, detector, ai, args.human_start, args.robot_only)
