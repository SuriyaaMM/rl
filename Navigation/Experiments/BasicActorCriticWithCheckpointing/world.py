import numpy as np


class GridWorld:
    """
    Grid World environment with sequential checkpoints
    
    The agent must visit checkpoints in order before reaching the final goal.
    """
    
    def __init__(self, size=(8, 8), checkpoints=[(2, 7), (6, 1), (7, 7)], max_iters=5000):
        """
        Initialize GridWorld environment
        
        Args:
            size: Grid dimensions (height, width)
            checkpoints: List of checkpoint positions to visit in order
            max_iters: Maximum steps per episode before termination
        """
        self.size = size
        self.checkpoints = checkpoints
        self.max_iters = max_iters
        self.reset()

    def reset(self):
        """Reset environment to initial state"""
        self.agent_pos = [0, 0]
        self.current_checkpoint_idx = 0
        self.iter = 0
        return self._get_state()

    def _get_state(self):
        """Get current state as tuple"""
        return tuple(self.agent_pos)

    def step(self, action):
        """
        Take a step in the environment
        
        Args:
            action: Integer action (0=up, 1=down, 2=left, 3=right)
            
        Returns:
            tuple: (next_state, reward, done)
        """
        x, y = self.agent_pos
        
        # Apply action with boundary checking
        if action == 0 and x > 0:  # Up
            x -= 1
        elif action == 1 and x < self.size[0] - 1:  # Down
            x += 1
        elif action == 2 and y > 0:  # Left
            y -= 1
        elif action == 3 and y < self.size[1] - 1:  # Right
            y += 1

        self.agent_pos = [x, y]
        self.iter += 1
        done = False
        
        # Small step penalty to encourage efficiency
        reward = -0.01 

        current_goal = self.checkpoints[self.current_checkpoint_idx]

        # Check if reached current checkpoint
        if (x, y) == current_goal:
            # Reward for reaching checkpoint
            reward = 1.0 if self.current_checkpoint_idx < len(self.checkpoints) - 1 else 2.0
            
            # Move to next checkpoint
            self.current_checkpoint_idx += 1
            
            # Check if all checkpoints completed
            if self.current_checkpoint_idx >= len(self.checkpoints):
                done = True
                self.iter = 0

        # Check for timeout
        elif self.iter >= self.max_iters:
            reward = -1
            done = True
            self.iter = 0

        return self._get_state(), reward, done