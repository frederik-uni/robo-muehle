from ultralytics import YOLO
from typing import Callable, Optional, Any
import time
import numpy as np

from imagedetection.board_geometry import BoardIndexMap
from imagedetection.homography import warp_from_corners
from imagedetection.index_mapping import nearest_index, estimate_grid_spacing_px, build_state_dict
from imagedetection.vision.stone_center import stone_centers
from imagedetection.vision.board_bbox import best_board_box, corners_from_bbox
from imagedetection.vision.debug_vis import _resize_to_height, _hstack_safe


class Detector:
    def __init__(
        self,
        board_indices_csv: str,
        board_model_path: str | None=None,
        stones_model_path: str | None=None,
        stacks_model_path: str | None=None,
        conf_min: float = 0.25,
        conf_accept: float = 0.45,
        dist_max_factor: float = 0.7,
        black_cls_id: int = 0,
        white_cls_id: int = 1,
        frame_provider: Optional[Callable[[], Any]] = None
    ):
        self.board = BoardIndexMap(board_indices_csv)
        self.board_model = YOLO(board_model_path) if board_model_path else None
        self.stones_model = YOLO(stones_model_path) if stones_model_path else None
        self.stacks_model = YOLO(stacks_model_path) if stacks_model_path else None
        self.conf_min = conf_min
        self.conf_accept = conf_accept
        self.dist_max_factor = dist_max_factor
        self.black_cls_id = black_cls_id
        self.white_cls_id = white_cls_id
        self.frame_provider = frame_provider

        grid = estimate_grid_spacing_px(self.board)
        self.dist_max = self.dist_max_factor * grid


    def get_gamestate(self, img_bgr):
        
        board_res0 = self.board_model(img_bgr)[0]
        box = best_board_box(board_res0)
        if box is None:
            raise RuntimeError("No board detected")

        corners = corners_from_bbox(*box)
        warped, H = warp_from_corners(img_bgr, corners)
        warped = np.ascontiguousarray(warped)
        
        stones_res0 = self.stones_model(warped)[0]
        
        best_per_idx = {}

        for cls_id, conf, cx, cy in stone_centers(stones_res0, conf_thres=self.conf_min):
            idx, dist = nearest_index(cx, cy, self.board)
            if idx is None:
                continue

            if dist > self.dist_max:
                continue
                
            candidate = {
                "cls_id": cls_id, 
                "conf": conf, 
                "cx": cx, 
                "cy": cy, 
                "dist": dist
            }

            if idx not in best_per_idx:
                best_per_idx[idx] = candidate
            else:
                prev = best_per_idx.get(idx)
                if (prev is None
                    or candidate["conf"] > prev["conf"]
                    or (candidate["conf"] == prev["conf"] and candidate["dist"] < prev["dist"])):
                    best_per_idx[idx] = candidate

        accepted_per_idx = {
            i: d for i, 
            d in best_per_idx.items() if d["conf"] >= self.conf_accept
        }


        vis_orig = board_res0.plot(img=img_bgr.copy())
        
        if self.stacks_model is not None:
            stacks_res0 = self.stacks_model(img_bgr)[0]
            vis_orig = stacks_res0.plot(img=vis_orig)

        vis_warp = stones_res0.plot(img=warped.copy())


        state = build_state_dict(
            self.board, 
            accepted_per_idx, 
            black_cls_id=self.black_cls_id, 
            white_cls_id=self.white_cls_id
        )

        remap = {-1: 0, 0: -1, 1: 1}
        state = {i: remap[v] for i, v in state.items()}


        h = 560
        left = _resize_to_height(vis_orig, h)
        right = _resize_to_height(vis_warp, h)
        
        debug_vis = _hstack_safe(left, right)


        order = self.board.get_abs_indices()
        state_arr = [state[i] for i in order]


        return state, { 
            "debug_vis": debug_vis,
            "state_arr": state_arr
        }    


    def _game_state_arr(self, game) -> list[int]:

        return game.board.astype(int).tolist()


    def _infer_human_delta(self, prev: list[int], curr: list[int]) -> dict:

        changed = []
        for i, (a, b) in enumerate(zip(prev, curr)):
            if a != b:
                changed.append((i, a, b))

        added = [(i, b) for (i, a, b) in changed if a == 0 and b != 0]
        removed = [(i, a) for (i, a, b) in changed if a != 0 and b == 0]

        if not changed:
            return {"from_idx": None, "to_idx": None, "remove_idx": None, "player": 0}

        if len(added) == 1 and len(removed) == 0:
            to_idx, player = added[0][0], added[0][1]
            return {"from_idx": None, "to_idx": to_idx, "remove_idx": None, "player": player}

        if len(added) == 1 and len(removed) == 1:
            to_idx, player = added[0][0], added[0][1]
            rem_idx, rem_val = removed[0][0], removed[0][1]
            if rem_val == player:
                return {"from_idx": rem_idx, "to_idx": to_idx, "remove_idx": None, "player": player}
            else:
                return {"from_idx": None, "to_idx": to_idx, "remove_idx": rem_idx, "player": player}

        if len(added) == 1 and len(removed) == 2:
            to_idx, player = added[0][0], added[0][1]
            from_idx = None
            remove_idx = None
            for (i, old_val) in removed:
                if old_val == player:
                    from_idx = i
                else:
                    remove_idx = i
            if from_idx is not None:
                return {"from_idx": from_idx, "to_idx": to_idx, "remove_idx": remove_idx, "player": player}

        raise RuntimeError(f"Unclear Delta transition prev->curr, changes={changed}")


    def _apply_delta_to_game(self, game, delta: dict) -> None:
        
        from_idx = delta["from_idx"]
        to_idx = delta["to_idx"]
        remove_idx = delta["remove_idx"]

        if to_idx is not None:
             game.move(from_idx, to_idx)

        if remove_idx is not None:
            game.remove_piece(remove_idx)


    def get_next_gamestate(
        self,
        game,
        img_bgr=None,
        stable_n: int = 3,
        max_frames: int = 120,
        sleep_s: float = 0.05,
    ):

        prev = self._game_state_arr(game)

        stable_state = None
        stable_info = None
        stable_count = 0

        for _ in range(max_frames):
            if img_bgr is None:
                if self.frame_provider is None:
                    raise RuntimeError("No img_bgr was passed and no frame_provider was set in the detector.")
                frame = self.frame_provider()
            else:
                frame = img_bgr

            _, info = self.get_gamestate(frame)
            curr = info["state_arr"]

            if curr == prev:
                stable_state = None
                stable_info = None
                stable_count = 0
            else:
                if stable_state is None or curr != stable_state:
                    stable_state = curr
                    stable_info = info
                    stable_count = 1
                else:
                    stable_count += 1

                if stable_count >= stable_n:
                    delta = self._infer_human_delta(prev, stable_state)
                    self._apply_delta_to_game(game, delta)

                    return {
                        "delta": delta,
                        "vision": {
                            "debug_vis": stable_info["debug_vis"],
                            "state_arr": stable_info["state_arr"],
                        },
                    }

            if img_bgr is None:
                time.sleep(sleep_s)

        raise TimeoutError("No stable human train detected (max_frames reached).")

