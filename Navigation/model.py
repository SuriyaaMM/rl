from torch import nn
import torch
import numpy as np

class ConvActor(nn.Module):
    def __init__(self, height, width, num_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * height * width, 64),
            nn.ReLU(),
            nn.Linear(64, num_actions)
        )

    def forward(self, x):
        return torch.softmax(self.net(x), dim=-1)

# --- Critic Model ---
class ConvCritic(nn.Module):
    def __init__(self, height, width):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * height * width, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)
    
def to_grid_tensor(state, size, env_instance, device):
    # multi channel grid
    grid = np.zeros((3, size[0], size[1]), dtype=np.float32)
    # channel 0 is agent
    grid[0, state[0], state[1]] = 1.0
    # channel 1 is current checkpoint
    current_goal_pos = env_instance.checkpoints[env_instance.current_checkpoint_idx]
    grid[1, current_goal_pos[0], current_goal_pos[1]] = 1.0
    # channel 2 is all checkpoint
    for cp in env_instance.checkpoints:
        grid[2, cp[0], cp[1]] = 0.5
    # differentiate completed checkpoints
    for i in range(env_instance.current_checkpoint_idx):
        cp = env_instance.checkpoints[i]
        grid[2, cp[0], cp[1]] = 0.3
    
    # flatten grid into single channel
    combined_grid = np.zeros(size, dtype=np.float32)
    combined_grid += grid[0] * 3.0  # agent position
    combined_grid += grid[1] * 2.0  # current target
    combined_grid += grid[2] * 1.0  # checkpoints
    
    return torch.tensor(combined_grid).unsqueeze(0).unsqueeze(0).to(device)