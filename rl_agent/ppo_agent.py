import torch
import torch.nn as nn
from torch.distributions import Categorical


class PPOAgent(nn.Module):
    """Actor-critic agent with a stochastic policy over discrete actions."""

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
        self.critic = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, state):
        # Ensure state is 2D: (batch_size, input_dim)
        if state.dim() == 1:
            state = state.unsqueeze(0)
        return self.actor(state), self.critic(state)

    def get_action(self, state, epsilon=0.0):
        """Sample an action from the policy's categorical distribution.

        Args:
            state (torch.Tensor): Current environment state.
            epsilon (float, optional): Unused; kept for compatibility.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                Sampled action, log probability of the action, and value estimate.
        """
        action_logits, value = self.forward(state)
        action_probs = torch.softmax(action_logits, dim=-1)
        dist = Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, value.squeeze(-1)
