from enum import Enum
from typing import Literal, cast

import numpy as np

from muhle_renderer import (
    calc_coords_muhle,
    load_board,
    pieces,
    state24_to_points,
)
from renderer import render


class Phase(Enum):
    PLACING = 0
    MOVING = 1
    JUMPING = 2


class Muehle:
    """Implements the complete game logic for Muehle (Nine Men's Morris).

    This class manages the game board, player turns, piece placement, movement,
    mill detection, and win/loss conditions.
    """
    board: np.ndarray
    to_place: dict[int, int]
    "valid moves if empty"
    vm: dict[int, list[int]]
    mills: list[list[int]]
    player: Literal[1, -1]

    """Nur klassenbasiert für dich Eugen"""

    def __init__(self):
        """Initializes the game board and state."""
        self.vm = self._make_vm()
        self.mills = self._make_mills()
        self.reset(reinit_board=True)

    def _is_jumping(self, player: Literal[1, -1]) -> bool:
        """Checks if a player is in the jumping phase (has only 3 pieces left)."""
        return (self.board == player).sum() == 3 and self._is_moving(player)

    def _is_moving(self, player: Literal[1, -1]) -> bool:
        """Checks if a player has placed all their pieces and is in the moving phase."""
        return self.to_place[player] == 0

    def _has_lost(self, player: Literal[1, -1]) -> bool:
        """Checks if a player has lost the game.

        A player loses if they have fewer than 3 pieces or have no legal moves.
        """
        if self._is_moving(player) and (self.board == player).sum() < 3:
            return True
        if self.phase(player) == Phase.MOVING:
            for source in range(24):
                if self.board[source] == player:
                    for target in self.vm.get(source, []):
                        if self.board[target] == 0:
                            return False  # found move
            return True  # no moves found
        return False

    def _truce(self):
        """Checks for a draw condition (both players are in the jumping phase)."""
        return self._is_jumping(1) and self._is_jumping(-1)

    def done(self):
        """Determines if the game is over and returns the winner.

        Returns:
            1 if player 1 has won, -1 if player -1 has won, or 0 if the game is ongoing.
        """
        if self._has_lost(1):
            return -1
        if self._has_lost(-1):
            return 1
        return 0

    def is_terminal(self):
        """Checks if the game has ended."""
        return self.done() != 0 or self._truce()

    def phase(self, player: Literal[1, -1]) -> Phase:
        """Gets the current game phase for a specific player."""
        if not self._is_moving(player):
            return Phase.PLACING
        elif not self._is_jumping(player):
            return Phase.MOVING
        else:
            return Phase.JUMPING

    def reset(self, reinit_board=False):
        """Resets the game to its initial state."""
        if reinit_board:
            self.board = np.zeros(24, dtype=np.int8)
        else:
            self.board[:] = 0
        self.to_place = {1: 9, -1: 9}
        self.player = 1

    def legal_actions_mask(self):
        """Returns a boolean mask of legal target positions for the current player."""
        mask = np.zeros(24, dtype=bool)
        p = self.player
        ph = self.phase(p)

        if ph == Phase.PLACING:
            mask[self.board == 0] = True
        elif ph == Phase.MOVING:
            for i in range(24):
                if self.board[i] == p:
                    for target in self.vm[i]:
                        if self.board[target] == 0:
                            mask[target] = True
        elif ph == Phase.JUMPING:
            for i in range(24):
                if self.board[i] == 0:
                    mask[i] = True
        return mask

    def move(self, source: int | None, target: int):
        """Executes a game move.

        Args:
            source: The starting position index (0-23) of the piece to move.
                    Should be None during the placing phase.
            target: The target position index (0-23).

        Returns:
            'remove' if the move creates a mill, 'done' if the move ends the game,
            'ok' otherwise.

        Raises:
            ValueError: If the move is illegal.
        """
        p = self.player
        ph = self.phase(p)

        if ph == Phase.PLACING:
            if self.board[target] != 0:
                raise ValueError("Invalid move")
            self.board[target] = p
            self.to_place[p] -= 1
            if self.is_mill(target, p):
                return "remove"
        else:
            if source is None:
                raise ValueError("Invalid source none")
            if self.board[source] != p:
                raise ValueError("Invalid source not player id")
            if ph == Phase.MOVING and target not in self.vm[source]:
                raise ValueError("Invalid target")
            if self.board[target] != 0:
                raise ValueError("Invalid target")

            self.board[source] = 0
            self.board[target] = p
            if self.is_mill(target, p):
                return "remove"

        if self.done() != 0:
            return "done"
        self.player = cast(Literal[-1, 1], self.player * -1)
        return "ok"

    def remove_piece(self, pos: int):
        """Removes an opponent's piece after a mill has been formed.

        Args:
            pos: The position of the piece to remove.

        Returns:
            The winner of the game (1 or -1) if the removal ends the game, otherwise 0.

        Raises:
            ValueError: If the removal is illegal (e.g., trying to remove a piece from a mill
                        when other pieces are available).
        """
        if not self.can_remove(pos, self.player):
            raise ValueError("Cannot remove piece from mill")
        self.player = cast(Literal[-1, 1], self.player * -1)
        self.board[pos] = 0
        return self.done()

    def is_mill(self, pos: int, player: Literal[1, -1]) -> bool:
        """Checks if a piece at a given position completes a mill for the specified player."""
        for mill in self.mills:
            if pos in mill and all(self.board[p] == player for p in mill):
                return True
        return False

    def count_almost_mills(self, player: Literal[1, -1]) -> int:
        """Counts how many almost mills a player has."""
        count = 0
        for mill in self.mills:
            pieces = [self.board[p] for p in mill]
            if pieces.count(player) == 2 and pieces.count(0) == 1:
                count += 1
        return count

    def _make_mills(self):
        """Defines all 16 possible mill combinations."""
        return [
            [0, 1, 2],
            [3, 4, 5],
            [6, 7, 8],
            [9, 10, 11],
            [12, 13, 14],
            [15, 16, 17],
            [18, 19, 20],
            [21, 22, 23],
            [0, 9, 21],
            [3, 10, 18],
            [6, 11, 15],
            [1, 4, 7],
            [16, 19, 22],
            [8, 12, 17],
            [5, 13, 20],
            [2, 14, 23],
        ]

    def can_remove(self, pos: int, remover: Literal[1, -1]) -> bool:
        """Check if a piece at pos can be removed."""

        remover = remover * -1
        if self.board[pos] != remover:
            return False

        remover_pieces = [i for i in range(24) if self.board[i] == remover]

        non_mill_pieces = [p for p in remover_pieces if not self.is_mill(p, remover)]

        if non_mill_pieces:
            return pos in non_mill_pieces
        return True

    def render(self):
        """Renders the current board state into a PIL Image."""
        state = np.where(self.board == -1, 2, self.board).astype(np.int8)

        points = state24_to_points(state)

        result = render(
            img=load_board(),
            pieces=pieces,
            points=points,
            calc_coords=calc_coords_muhle,
        )
        return result

    def _make_vm(self):
        """Defines the graph of valid moves (adjacency list) between board positions."""
        # It took longer to create the graphic than writing the code
        # 0 ---------------------- 1 --------------------- 2
        # |                        |                       |
        # |                        |                       |
        # |         3 ------------ 4 ----------- 5         |
        # |         |              |             |         |
        # |         |              |             |         |
        # |         |      6 ----- 7 ----- 8     |         |
        # |         |      |               |     |         |
        # |         |      |               |     |         |
        # 9 ------  10 --- 11             12 --- 13 ----- 14
        # |         |      |               |     |         |
        # |         |      |               |     |         |
        # |         |      15 ---- 16 ----17     |         |
        # |         |              |             |         |
        # |         |              |             |         |
        # |         18 ----------- 19 ---------- 20        |
        # |                        |                       |
        # |                        |                       |
        # 21 --------------------- 22 -------------------- 23

        return {
            0: [1, 9],
            1: [0, 2, 4],
            2: [1, 14],
            3: [4, 10],
            4: [1, 3, 5, 7],
            5: [4, 13],
            6: [7, 11],
            7: [4, 6, 8],
            8: [7, 12],
            9: [0, 10, 21],
            10: [3, 9, 11, 18],
            11: [6, 10, 15],
            12: [8, 13, 17],
            13: [5, 12, 14, 20],
            14: [2, 13, 23],
            15: [11, 16],
            16: [15, 17, 19],
            17: [12, 16],
            18: [10, 19],
            19: [16, 18, 20, 22],
            20: [13, 19],
            21: [9, 22],
            22: [19, 21, 23],
            23: [20, 22],
        }


if __name__ == "__main__":
    game = Muehle()
    game.move(None, 0)
    game.move(None, 23)

    result = game.render()
    result.save("muehle.png")
