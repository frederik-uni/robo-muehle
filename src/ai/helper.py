from typing import Literal

import numpy as np
import torch

from muehle_game import Muehle


def encode_data(
    env: Muehle, player: Literal[1, -1]
) -> tuple[torch.Tensor, torch.Tensor]:
    board = env.board

    my = (board == player).astype(np.float32)
    opp = (board == -player).astype(np.float32)
    empty = (board == 0).astype(np.float32)
    board_tensor = torch.tensor(
        np.stack([my, opp, empty], axis=0), dtype=torch.float32
    )  # (3, 24)

    to_place_me_norm = env.to_place[player] / 9.0
    to_place_opp_norm = env.to_place[-player] / 9.0

    phase_onehot = np.zeros(3, dtype=np.float32)
    phase_onehot[env.phase(player).value] = 1.0
    phase_opp_onehot = np.zeros(3, dtype=np.float32)
    phase_opp_onehot[env.phase(-player).value] = 1.0
    global_features = np.concatenate(
        [
            phase_onehot,  # 3
            phase_opp_onehot,  # 3
            [to_place_me_norm],  # 1
            [to_place_opp_norm],  # 1
            [env.phase(player).value == 2],  # 1
            [env.phase(-player).value == 2],  # 1
        ]
    ).astype(np.float32)  # (10,)

    return board_tensor, torch.tensor(global_features, dtype=torch.float32)
