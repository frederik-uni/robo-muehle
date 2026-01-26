import math
import numpy as np
from imagedetection.board_geometry import BoardIndexMap


board = BoardIndexMap("assets/indices/board_indices.csv")


def nearest_index(x_obj, y_obj, board):
    best_idx = None
    best_dist = float("inf")

    for idx, (x_rel, y_rel) in board.indices.items():
        x_i, y_i = board.rel_to_abs(x_rel, y_rel)
        d = math.hypot(x_obj - x_i, y_obj - y_i)
        if d < best_dist:
            best_dist = d
            best_idx = idx

    return best_idx, best_dist


def estimate_grid_spacing_px(board) -> float:
    pts = []
    for _, (x_rel, y_rel) in board.indices.items():
        x, y = board.rel_to_abs(x_rel, y_rel)
        pts.append((x, y))

    P = np.array(pts, dtype=np.float32)
    D = np.sqrt(((P[:, None, :] - P[None, :, :]) ** 2).sum(axis=2))
    np.fill_diagonal(D, np.inf)
    nn = D.min(axis=1)
    return float(np.median(nn))


def build_state_dict(board, best_per_idx, black_cls_id=-1, white_cls_id=1):
    
    state = {idx: -1 for idx in board.indices.keys()}

    for idx, detection in best_per_idx.items():
        if detection["cls_id"] == black_cls_id:
            state[idx] = 0
        elif detection["cls_id"] == white_cls_id:
            state[idx] = 1

    return state
