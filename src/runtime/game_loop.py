from ai.train import SelfPlayAgent
from imagedetection.detector import Detector
from muehle_game import Muehle
from piecewalker.ned2 import Ned2


def run_game_loop(
    robot: Ned2,
    detector: Detector,
    ai: SelfPlayAgent,
    human_start: bool,
    robot_only=False,
):
    game = Muehle()
    skip_first = not human_start
    while True:
        if skip_first:
            skip_first = False
        else:
            if robot_only:
                from_idx, to, remove = ai.get_complete_move(game)
                robot.move(from_idx=from_idx, to_idx=to)
                game.move(from_idx, to)
                if remove:
                    robot.move(from_idx=remove, to_idx=None)
                    game.remove_piece(remove)
            else:
                # todo: update
                detector.get_next_gamestate(game)
            if game.is_terminal():
                break
        from_idx, to, remove = ai.get_complete_move(game)
        robot.move(from_idx=from_idx, to_idx=to)
        game.move(from_idx, to)
        if remove is not None:
            robot.move(from_idx=remove, to_idx=None)
            game.remove_piece(remove)
        if game.is_terminal():
            break
