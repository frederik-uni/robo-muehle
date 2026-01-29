from typing import Literal

import numpy as np
import torch

from muehle_game import Muehle


def encode_data(
    env: Muehle, player: Literal[1, -1], removal_pending: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encodes the game state into tensors for the policy network.

    Args:
        env: The Muehle game environment.
        player: The current player's perspective (1 or -1).
        removal_pending: Whether a piece removal is pending for the current player.

    Returns:
        A tuple containing:
        - board_tensor: A (3, 24) tensor representing the board state (my pieces, opponent pieces, empty spaces).
        - global_features: A (11,) tensor with global game features.
    """
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
            [float(removal_pending)],  # 1
        ]
    ).astype(np.float32)  # (11,)

    return board_tensor, torch.tensor(global_features, dtype=torch.float32)
