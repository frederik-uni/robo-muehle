from pathlib import Path

import numpy as np
import torch

from muehle_game import Muehle, Phase
from muhle_renderer import MUEHLE_INDEX_MAP, calc_coords_muhle

from .policy import ThePolicy
from .train import SelfPlayAgent


def load_model(path: Path):
    """Loads a trained model for gameplay.

    Args:
        path: The path to the saved model state dictionary.

    Returns:
        A SelfPlayAgent initialized with the loaded model.
    """
    model = ThePolicy()
    state_dict = torch.load(path, map_location="cpu")
    model.load_state_dict(state_dict)
    return SelfPlayAgent(model)


def play_game():
    """Starts an interactive game of Muehle using matplotlib for UI.

    Allows a human player to click on the board to make moves.
    The game state is displayed and updated in the matplotlib window.
    """
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt

    game = Muehle()
    game_state = {"move_count": 0, "state": None, "selected": None}

    def show_board_clickable():
        """Sets up and displays the clickable game board using matplotlib."""
        fig, ax = plt.subplots()
        board_img = ax.imshow(game.render())

        coords_map = []
        for i in range(MUEHLE_INDEX_MAP.shape[0]):
            for j in range(MUEHLE_INDEX_MAP.shape[1]):
                index = MUEHLE_INDEX_MAP[i, j]
                if index != 0:
                    x, y, size, _, _, _ = calc_coords_muhle(i, j)
                    coords_map.append((x, y, size, index))

        selected_marker = None

        state_text = ax.text(
            0, -1, "", fontsize=12, color="blue", ha="left", va="center"
        )

        def update_state_label():
            """Updates the text label displaying the current game phase or action."""
            state = game_state.get("state")
            phase = game.phase(game.player)
            if state == "remove":
                label = "Removing"
            elif phase == Phase.PLACING:
                label = game.to_place[game.player]
            elif phase == Phase.MOVING:
                label = "Moving"
            else:
                label = "Jumping"
            state_text.set_text(label)
            fig.canvas.draw_idle()

        def draw_selected_marker():
            """Draws a red circle to indicate the currently selected piece."""
            nonlocal selected_marker
            if selected_marker is not None:
                selected_marker.remove()
                selected_marker = None
            sel_idx = game_state["selected"]
            if sel_idx is not None:
                for x, y, size, index in coords_map:
                    if index - 1 == sel_idx:
                        selected_marker = patches.Circle(
                            (x, y),
                            radius=size / 2 + 2,
                            edgecolor="red",
                            facecolor="none",
                            linewidth=2,
                        )
                        ax.add_patch(selected_marker)
                        break
            fig.canvas.draw_idle()

        def onclick(event):
            """Handles mouse click events on the board for making moves."""
            click_x, click_y = event.xdata, event.ydata
            if click_x is None or click_y is None:
                return
            idx = -1
            for x, y, size, index in coords_map:
                if abs(click_x - x) <= size / 2 and abs(click_y - y) <= size / 2:
                    idx = index - 1
                    break
            if idx != -1:
                if game_state["state"] == "remove":
                    game.remove_piece(idx)
                    if game.is_terminal():
                        exit(0)
                    game_state["state"] = None
                    board_img.set_data(game.render())
                    update_state_label()
                    fig.canvas.draw_idle()
                    return
                if game.phase(game.player) == Phase.PLACING:
                    game_state["state"] = (
                        "remove" if game.move(None, idx) == "remove" else None
                    )
                    if game.is_terminal():
                        exit(0)
                    board_img.set_data(game.render())
                    fig.canvas.draw_idle()
                    update_state_label()
                    game_state["move_count"] += 1
                else:
                    sel = game_state["selected"]
                    on_board = game.board[idx]
                    if on_board == game.player:
                        game_state["selected"] = idx
                        draw_selected_marker()
                    else:
                        if sel is not None:
                            game_state["state"] = (
                                "remove" if game.move(sel, idx) == "remove" else None
                            )
                            if game.is_terminal():
                                exit(0)
                            update_state_label()
                            board_img.set_data(game.render())
                            fig.canvas.draw_idle()
                            game_state["selected"] = None
                            game_state["move_count"] += 1

        fig.canvas.mpl_connect("button_press_event", onclick)
        plt.axis("off")
        plt.show()

    show_board_clickable()


if __name__ == "__main__":
    # play_game()
    game = Muehle()
    model = load_model(Path("models/policy_final-4.pth"))
    count = 0
    Path("game").mkdir(parents=True, exist_ok=True)

    rendered = game.render()
    rendered.save(f"game/{count}.png")
    count += 1

    legal_moves = np.where(game.legal_actions_mask())[0]
    random_move = 4
    game.move(None, random_move)

    while True:
        rendered = game.render()
        rendered.save(f"game/{count}.png")
        count += 1
        if game.is_terminal():
            print("Game over!")
            break
        s, t, r = model.get_complete_move(game)
        if t is None:
            raise ValueError("Invalid move")
        game.move(s, t)
        if r is not None:
            game.remove_piece(r)
