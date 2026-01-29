import torch
import torch.nn as nn
import torch.nn.functional as F


# todo: maybe gnn, but ...
class ThePolicy(nn.Module):
    """The neural network model for the Muehle AI.

    This model takes the encoded board state and global game features as input
    and outputs policy logits (for action selection) and a state value (for evaluation).
    """

    def __init__(self):
        """Initializes the layers of the neural network."""
        super().__init__()

        self.in_board_fc = nn.Linear(3 * 24, 128)  # board, op_board, empty
        self.in_global_fc = nn.Linear(11, 32)  # global features

        self.fc1 = nn.Linear(128 + 32, 256)
        self.fc2 = nn.Linear(256, 256)

        # Policy head/600 logits
        # from -> to: 24*24
        # remove: 24
        self.policy_head = nn.Linear(256, 24 * 25)

        # training only
        self.value_fc = nn.Linear(256, 128)
        self.value_head = nn.Linear(128, 1)

    def forward(self, board, global_features):
        """Performs the forward pass of the model.

        Args:
            board: A tensor of shape (batch, 3, 24) representing the board state.
            global_features: A tensor of shape (batch, 11) with global game features.

        Returns:
            A tuple containing:
            - policy_logits: Raw logits for each possible action.
            - value: The predicted value of the current game state, between -1 and 1.
        """

        x_board = torch.flatten(board, start_dim=1)
        x_board = F.relu(self.in_board_fc(x_board))

        x_global = F.relu(self.in_global_fc(global_features))

        x = torch.cat([x_board, x_global], dim=-1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        policy_logits = self.policy_head(x)
        if True:
            v = F.relu(self.value_fc(x))
            value = torch.tanh(self.value_head(v))

            return policy_logits, value
        else:
            return policy_logits
