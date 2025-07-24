import numpy as np

class GridWorld:
    def __init__(self, size=(8, 8), checkpoints=[(2, 7), (6, 1), (7, 7)], max_iters=5000):
        self.size = size
        self.checkpoints = checkpoints
        self.max_iters = max_iters
        self.reset()

    def reset(self):
        self.agent_pos = [0, 0]
        self.current_checkpoint_idx = 0
        self.iter = 0
        return self._get_state()

    def _get_state(self):
        return tuple(self.agent_pos)

    def step(self, action):
        x, y = self.agent_pos
        if action == 0 and x > 0: x -= 1
        elif action == 1 and x < self.size[0] - 1: x += 1
        elif action == 2 and y > 0: y -= 1
        elif action == 3 and y < self.size[1] - 1: y += 1

        self.agent_pos = [x, y]
        self.iter += 1
        done = False
        # step penalty
        reward = -0.01 

        current_goal = self.checkpoints[self.current_checkpoint_idx]

        if (x, y) == current_goal:
            reward = 1.0 if self.current_checkpoint_idx < len(self.checkpoints) - 1 else 2.0
            self.current_checkpoint_idx += 1
            if self.current_checkpoint_idx >= len(self.checkpoints):
                done = True
                self.iter = 0

        elif self.iter >= self.max_iters:
            reward = -1
            done = True
            self.iter = 0

        return self._get_state(), reward, done
