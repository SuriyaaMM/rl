import torch
import torch.nn as nn
import numpy as np


class ConvActor(nn.Module):
    """Convolutional Actor Network for policy learning"""
    
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


class ConvCritic(nn.Module):
    """Convolutional Critic Network for value estimation"""
    
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
    """
    Convert state to grid tensor representation
    
    Args:
        state: Current agent position (x, y)
        size: Grid size (height, width)
        env_instance: Environment instance for checkpoint info
        device: PyTorch device
    
    Returns:
        Tensor representation of the grid state
    """
    # Initialize empty grid
    grid = np.zeros(size, dtype=np.float32)
    
    # Mark agent position
    grid[state] = 1.0 
    
    # Mark current active checkpoint as goal
    current_goal_pos = env_instance.checkpoints[env_instance.current_checkpoint_idx]
    grid[current_goal_pos] = 2.0 
    
    # Convert to tensor and add batch and channel dimensions
    return torch.tensor(grid).unsqueeze(0).unsqueeze(0).to(device)