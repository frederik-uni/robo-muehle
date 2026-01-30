import copy
import random
from collections import deque
from typing import List, Optional, cast

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from typing_extensions import Literal

from muehle_game import Muehle, Phase

from .helper import encode_data
from .policy import ThePolicy


class ActionMapper:
    """
    Maps between game actions and policy network output indices.

    The policy network outputs 600 logits (24*25). We interpret this as:
    - Sources: 0-23 for actual piece positions, 24 for "no source" (place/remove)
    - Targets: 0-23 for board positions
    Action index = source * 24 + target
    """

    NUM_SOURCES = 25  # 0-23: piece positions, 24: no source (place/remove)
    NUM_TARGETS = 24
    TOTAL_ACTIONS = NUM_SOURCES * NUM_TARGETS

    @staticmethod
    def to_index(source: int, target: int) -> int:
        """Convert source (0-24) and target (0-23) to action index."""
        if not (
            0 <= source < ActionMapper.NUM_SOURCES
            and 0 <= target < ActionMapper.NUM_TARGETS
        ):
            raise ValueError("Invalid source or target for action index.")
        return source * ActionMapper.NUM_TARGETS + target

    @staticmethod
    def from_index(idx: int) -> tuple[int, int]:
        """Convert action index to (source, target) tuple."""
        if not (0 <= idx < ActionMapper.TOTAL_ACTIONS):
            raise ValueError(f"Invalid action index {idx}")
        source = idx // ActionMapper.NUM_TARGETS
        target = idx % ActionMapper.NUM_TARGETS
        return source, target

    @staticmethod
    def get_legal_mask(
        env: Muehle, player: Literal[-1, 1], removal_pending: bool = False
    ) -> np.ndarray:
        """Create a legal action mask of size TOTAL_ACTIONS.

        Args:
            env: The Muehle game environment.
            player: The player for whom to get the legal moves.
            removal_pending: Whether a piece removal is pending.

        Returns:
            A boolean numpy array of size TOTAL_ACTIONS, where True indicates a legal move.
        """
        mask = np.zeros(ActionMapper.TOTAL_ACTIONS, dtype=bool)
        board = env.board

        if removal_pending:
            for target in range(24):
                if env.can_remove(target, player):
                    idx = ActionMapper.to_index(24, target)
                    mask[idx] = True
            return mask

        phase = env.phase(player)

        if phase == Phase.PLACING:
            for target in range(24):
                if board[target] == 0:
                    idx = ActionMapper.to_index(24, target)
                    mask[idx] = True
        elif phase == Phase.MOVING:
            for source in range(24):
                if board[source] == player:
                    for target in env.vm.get(source, []):
                        if board[target] == 0:
                            idx = ActionMapper.to_index(source, target)
                            mask[idx] = True
        elif phase == Phase.JUMPING:
            for source in range(24):
                if board[source] == player:
                    for target in range(24):
                        if board[target] == 0:
                            idx = ActionMapper.to_index(source, target)
                            mask[idx] = True
        return mask


class SelfPlayAgent:
    """Agent that plays using the policy network with exploration."""

    def __init__(self, model: ThePolicy, device: torch.device | None = None):
        """Initializes the SelfPlayAgent.

        Args:
            model: The policy network to use for action selection.
            device: The torch device (CPU or CUDA) to run the model on.
        """
        self.model = model
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)

    def select_action(
        self,
        board_tensor: torch.Tensor,
        global_features: torch.Tensor,
        legal_mask: np.ndarray,
        temperature: float = 1.0,
        epsilon: float = 0.1,
    ) -> tuple[int, torch.Tensor, torch.Tensor]:
        """Select an action using the policy network, with exploration.

        Args:
            board_tensor: The encoded board state.
            global_features: The encoded global game features.
            legal_mask: A boolean mask of legal actions.
            temperature: Controls the randomness of action selection. Higher values lead to more random actions.
            epsilon: The probability of choosing a random action (epsilon-greedy exploration).

        Returns:
            A tuple containing:
            - The selected action index.
            - The raw policy logits from the network.
            - The predicted state value from the network.
        """
        board_tensor = board_tensor.unsqueeze(0).to(self.device)
        global_features = global_features.unsqueeze(0).to(self.device)

        with torch.no_grad():
            self.model.eval()
            policy_logits, value = self.model(board_tensor, global_features)

        legal_mask_tensor = torch.tensor(
            legal_mask, dtype=torch.bool, device=self.device
        )

        masked_logits = policy_logits.clone()
        masked_logits[0, ~legal_mask_tensor] = -1e9

        if random.random() < epsilon:
            legal_indices = torch.where(legal_mask_tensor)[0]
            if len(legal_indices) == 0:
                raise RuntimeError("No legal moves available for action selection.")
            action_idx = legal_indices[torch.randint(len(legal_indices), (1,))].item()
        else:
            scaled_logits = masked_logits / temperature
            probs = F.softmax(scaled_logits, dim=-1)
            dist = Categorical(probs=probs)
            action_idx = dist.sample().item()

        return action_idx, policy_logits.squeeze(0), value.squeeze(0)

    def next_move(
        self, game: Muehle, removal_pending: bool = False
    ) -> tuple[Optional[int], Optional[int], Optional[int]]:
        """
        Determines the next best move for the agent given the current game state.

        Args:
            game: The current Muehle game instance.
            removal_pending: True if the current action is to remove an opponent's piece.

        Returns:
            A tuple (from_idx, to_idx, remove_idx) describing a single action.
        """
        current_player = game.player
        board_tensor, global_features = encode_data(
            game, current_player, removal_pending
        )
        legal_mask = ActionMapper.get_legal_mask(game, current_player, removal_pending)

        if not legal_mask.any():
            return None, None, None

        board_tensor_in = board_tensor.unsqueeze(0).to(self.device)
        global_features_in = global_features.unsqueeze(0).to(self.device)

        with torch.no_grad():
            self.model.eval()
            policy_logits, _ = self.model(board_tensor_in, global_features_in)

        legal_mask_tensor = torch.tensor(
            legal_mask, dtype=torch.bool, device=self.device
        )
        masked_logits = policy_logits.clone()
        masked_logits[0, ~legal_mask_tensor] = -1e9

        action_idx = torch.argmax(masked_logits).item()
        source, target = ActionMapper.from_index(action_idx)

        from_idx: Optional[int] = None
        to_idx: Optional[int] = None
        remove_idx: Optional[int] = None

        if removal_pending:
            remove_idx = target
        elif source == ActionMapper.NUM_SOURCES - 1:
            to_idx = target
        else:
            from_idx = source
            to_idx = target

        return from_idx, to_idx, remove_idx

    def get_complete_move(
        self, game: Muehle
    ) -> tuple[Optional[int], Optional[int], Optional[int]]:
        """
        Computes a full turn, including a move and a subsequent removal if a mill is formed.

        Args:
            game: The current Muehle game instance.

        Returns:
            A tuple (from_idx, to_idx, remove_idx) for the complete turn.
        """
        remove_idx: Optional[int] = None

        from_idx, to_idx, _ = self.next_move(game, removal_pending=False)

        if to_idx is None:
            return None, None, None

        temp_game = copy.deepcopy(game)
        try:
            result = temp_game.move(from_idx, to_idx)
        except ValueError:
            return from_idx, to_idx, None

        if result == "remove":
            _, _, remove_idx = self.next_move(temp_game, removal_pending=True)

        return from_idx, to_idx, remove_idx


class SelfPlayTrainer:
    """Manages self-play training with policy gradient."""

    def __init__(
        self,
        model: ThePolicy,
        learning_rate: float = 1e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        device: torch.device | None = None,
    ):
        """Initializes the SelfPlayTrainer.

        Args:
            model: The policy model to be trained.
            learning_rate: The learning rate for the optimizer.
            gamma: The discount factor for future rewards.
            gae_lambda: The lambda factor for Generalized Advantage Estimation (GAE).
            entropy_coef: The coefficient for the entropy bonus in the loss function.
            value_coef: The coefficient for the value loss in the loss function.
            max_grad_norm: The maximum norm for gradient clipping.
            device: The torch device (CPU or CUDA) to run the training on.
        """
        self.model = model
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.agent = SelfPlayAgent(model, self.device)

    def collect_episode(
        self, temperature: float = 1.0, epsilon: float = 0.1
    ) -> tuple[List[dict], int]:
        """Play one full episode of self-play, collecting data for training.

        The agent plays against itself, and each step (state, action, reward, etc.)
        is recorded. Exploration is controlled by the temperature and epsilon parameters.

        Args:
            temperature: Controls the randomness of action selection.
            epsilon: The probability of choosing a random action.

        Returns:
            A tuple containing:
            - trajectory: A list of dictionaries, where each dictionary represents a step in the episode.
            - winner: The winner of the game (1, -1, or 0 for a draw).
        """
        env = Muehle()
        trajectory = []
        removal_pending = False
        removal_player = None
        winner = 0

        while not env.is_terminal():
            current_player = env.player
            reward = 0.0

            if removal_pending and removal_player != current_player:
                removal_pending = False
                removal_player = None

            almost_mills_player_before = env.count_almost_mills(current_player)

            legal_mask_np = ActionMapper.get_legal_mask(
                env, current_player, removal_pending
            )
            if not legal_mask_np.any():
                winner = -current_player
                break

            board_tensor, global_features = encode_data(
                env, current_player, removal_pending
            )

            action_idx, policy_logits, value = self.agent.select_action(
                board_tensor, global_features, legal_mask_np, temperature, epsilon
            )

            step_data = {
                "player": current_player,
                "board_tensor": board_tensor.cpu(),
                "global_features": global_features.cpu(),
                "action_idx": action_idx,
                "policy_logits": policy_logits.cpu(),
                "value": value.cpu(),
                "legal_mask": legal_mask_np,
            }

            try:
                source, target = ActionMapper.from_index(action_idx)
                if removal_pending:
                    is_breaking_mill = env.is_mill(target, -current_player)
                    result = env.remove_piece(target)
                    reward += 2.0
                    if is_breaking_mill:
                        reward += 0.5
                    removal_pending = False
                    removal_player = None
                    if result == 0:
                        env.player = cast(Literal[1, -1], -env.player)
                else:
                    is_blocking = False
                    for mill in env.mills:
                        if target in mill:
                            pieces = [env.board[p] for p in mill]
                            if (
                                pieces.count(-current_player) == 2
                                and pieces.count(0) == 1
                            ):
                                is_blocking = True
                                break

                    result = env.move(None if source == 24 else source, target)

                    if is_blocking:
                        reward += 0.7

                    if result != "remove":
                        almost_mills_player_after = env.count_almost_mills(
                            current_player
                        )
                        if almost_mills_player_after > almost_mills_player_before:
                            reward += 0.5

                    if result == "remove":
                        reward += 1.0
                        removal_pending = True
                        removal_player = current_player

                step_data["reward"] = reward
                trajectory.append(step_data)

                if env.is_terminal():
                    winner = env.done()
                    break

            except ValueError:
                winner = -current_player
                step_data["reward"] = -5.0
                trajectory.append(step_data)
                break

        if trajectory:
            if winner != 0:
                final_reward = 10.0 if trajectory[-1]["player"] == winner else -10.0
                trajectory[-1]["reward"] += final_reward
            elif env._truce():
                pass

        return trajectory, winner

    def compute_advantages(
        self, trajectory: List[dict]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Computes the Generalized Advantage Estimation (GAE) and returns for a given trajectory.

        Args:
            trajectory: A list of steps collected from an episode.

        Returns:
            A tuple containing:
            - advantages: A tensor of computed advantages for each step.
            - returns: A tensor of computed returns (discounted rewards) for each step.
        """
        if not trajectory:
            return torch.tensor([]), torch.tensor([])

        rewards = torch.tensor(
            [step["reward"] for step in trajectory], dtype=torch.float32
        )
        values = torch.stack([step["value"] for step in trajectory]).squeeze()

        advantages = torch.zeros_like(rewards)
        gae = 0
        for t in reversed(range(len(rewards))):
            next_value = values[t + 1] if t < len(rewards) - 1 else 0.0
            delta = rewards[t] + self.gamma * next_value - values[t]
            gae = delta + self.gamma * self.gae_lambda * gae
            advantages[t] = gae

        returns = advantages + values

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages, returns

    def update_model(self, trajectory: List[dict]) -> dict:
        """Updates the policy and value networks using the collected trajectory data.

        This method computes the loss and performs a gradient descent step.

        Args:
            trajectory: A list of steps collected from one or more episodes.

        Returns:
            A dictionary containing information about the losses (total, policy, value) and entropy.
        """
        if not trajectory:
            return {}

        board_batch = torch.stack([step["board_tensor"] for step in trajectory]).to(
            self.device
        )
        global_batch = torch.stack([step["global_features"] for step in trajectory]).to(
            self.device
        )
        actions = torch.tensor(
            [step["action_idx"] for step in trajectory], dtype=torch.long
        ).to(self.device)
        legal_mask_batch = torch.stack(
            [torch.tensor(step["legal_mask"], dtype=torch.bool) for step in trajectory]
        ).to(self.device)

        advantages, returns = self.compute_advantages(trajectory)
        advantages = advantages.to(self.device)
        returns = returns.to(self.device)

        self.model.train()
        policy_logits, values = self.model(board_batch, global_batch)
        values = values.squeeze()

        masked_logits = policy_logits.clone()
        masked_logits[~legal_mask_batch] = -1e9

        log_probs = F.log_softmax(masked_logits, dim=-1)
        action_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze()

        policy_loss = -(action_log_probs * advantages).mean()
        value_loss = F.mse_loss(values, returns)

        probs = F.softmax(masked_logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1).mean()
        loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()

        return {
            "total_loss": loss.item(),
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy.item(),
        }

    def train(
        self,
        num_episodes: int = 10000,
        episodes_per_update: int = 10,
        temperature: float = 1.0,
        epsilon: float = 0.1,
        save_every: int = 100,
        save_path: str = "policy_checkpoint.pth",
    ):
        """Runs the main training loop for a specified number of episodes.

        It collects self-play episodes, updates the model periodically, and saves checkpoints.

        Args:
            num_episodes: The total number of episodes to train for.
            episodes_per_update: The number of episodes to collect before each model update.
            temperature: The exploration temperature for action selection.
            epsilon: The exploration epsilon for action selection.
            save_every: The interval (in episodes) for saving model checkpoints.
            save_path: The path to save model checkpoints.
        """
        print(f"Starting training on {self.device}")
        print(f"Total parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        win_loss_draw = deque(maxlen=100)
        all_trajectories = []

        for episode in range(1, num_episodes + 1):
            trajectory, winner = self.collect_episode(temperature, epsilon)
            all_trajectories.extend(trajectory)
            win_loss_draw.append(winner)

            if episode % episodes_per_update == 0:
                loss_info = self.update_model(all_trajectories)
                all_trajectories.clear()

                p1_wins = np.mean([1 if r == 1 else 0 for r in win_loss_draw])
                p2_wins = np.mean([1 if r == -1 else 0 for r in win_loss_draw])
                draws = np.mean([1 if r == 0 else 0 for r in win_loss_draw])

                print(
                    f"Ep {episode:5d} | "
                    f"Loss: {loss_info.get('total_loss', 0):.3f} | "
                    f"P1 Win: {p1_wins:.2f}, P2 Win: {p2_wins:.2f}, Draw: {draws:.2f}"
                )

            if episode % save_every == 0:
                torch.save(
                    {
                        "episode": episode,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                    },
                    save_path,
                )
                print(f"Checkpoint saved at episode {episode}")

        torch.save(self.model.state_dict(), "policy_final.pth")
        print("Training complete! Model saved as 'policy_final.pth'")


def main():
    """Main training function."""
    model = ThePolicy()
    trainer = SelfPlayTrainer(
        model,
        learning_rate=1e-4,
        gamma=0.99,
        gae_lambda=0.95,
        entropy_coef=0.01,
        value_coef=0.5,
        max_grad_norm=0.5,
    )
    trainer.train(
        num_episodes=50000,
        episodes_per_update=20,
        temperature=1.0,
        epsilon=0.2,
        save_every=500,
    )


if __name__ == "__main__":
    main()
