"""
Shared neural network architectures for DQN-based agents.

Contains the DuelingDQN network used by both the training script
(train_dqn.py) and the browser player (play_browser.py).
"""

import torch.nn as nn


class DuelingDQN(nn.Module):
    """
    Dueling DQN: separates state value V(s) from advantage A(s,a).
    Q(s,a) = V(s) + A(s,a) - mean(A)
    This helps when many actions have similar values (e.g., obstacle is far away).
    """
    def __init__(self, input_dim, action_size):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )
        self.value = nn.Linear(256, 1)
        self.advantage = nn.Linear(256, action_size)

    def forward(self, x):
        feat = self.feature(x)
        value = self.value(feat)
        advantage = self.advantage(feat)
        return value + advantage - advantage.mean(dim=1, keepdim=True)
