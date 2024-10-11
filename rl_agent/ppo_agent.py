import torch
import torch.nn as nn
from torch.distributions import Categorical

class PPOAgent(nn.Module):
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
        action_logits, value = self.forward(state)
        action_probs = torch.softmax(action_logits, dim=-1)
        
        if torch.rand(1).item() < epsilon:
            action = torch.randint(0, action_probs.shape[-1], (action_probs.shape[0],))
        else:
            action = action_probs.argmax(dim=-1)
        
        log_prob = torch.log(action_probs.gather(-1, action.unsqueeze(-1)).squeeze(-1) + 1e-8)
        return action, log_prob, value.squeeze(-1)