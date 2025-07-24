import numpy as np
import matplotlib.pyplot as plt
from world import GridWorld
import random

size = (16, 16)
goal = (15, 15)
env = GridWorld(size=size, goal=goal)

# Fully preallocated Q-table
Q = np.random.standard_normal((size[0], size[1], 4))  # H x W x Actions
Q = Q + np.ones_like(Q)
# Hyperparameters
alpha = 3e-4
gamma = 0.25
epsilon = 0.03
num_epochs = 5000

rewards = []

for episode in range(num_epochs):
    state = env.reset()  # (x, y)
    total_reward = 0
    done = False

    while not done:
        x, y = state

        # Action selection: ε-greedy
        if np.random.rand() < epsilon:
            action = np.random.randint(4)
        else:
            action = np.argmax(Q[x, y])  # vectorized access

        # Step in environment
        next_state, reward, done = env.step(action)
        nx, ny = next_state
        total_reward += reward

        # Vectorized Q-update
        best_next = np.max(Q[nx, ny])  # max_a' Q(s', a')
        Q[x, y, action] += alpha * (reward + gamma * best_next - Q[x, y, action])

        state = next_state

    # Track every 100 episodes
    if episode % 100 == 0:
        rewards.append(total_reward)
        print(f"Episode {episode}: Reward = {total_reward}")


plt.scatter(range(0, num_epochs, 100), rewards)
plt.xlabel("Epochs")
plt.ylabel("Reward")
plt.title("Rewards vs Epoch")
plt.savefig(f"Exp_Rewards_vs_Epochs_{num_epochs}.png")
plt.show()


actions = ["↑", "↓", "←", "→"]

print("\nLearned Policy:")
for i in range(size[0]):
    row = ""
    for j in range(size[1]):
        s = (i, j)
        if s == (goal[0], goal[1]):
            row += " G  "
        else:
            best_a = int(np.argmax(Q[s]))
            row += f" {actions[best_a]}  "
    print(row)
