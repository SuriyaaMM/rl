import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from world import GridWorld
from model import ConvActor, ConvCritic, to_grid_tensor

# Environment setup
grid_size = (8, 8)
checkpoints = [(2, 7), (6, 1), (7, 7)]
num_actions = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = GridWorld(size=grid_size, checkpoints=checkpoints)
H, W = grid_size

# Initialize actor and critic
actor = ConvActor(H, W, num_actions).to(device)
critic = ConvCritic(H, W).to(device)

# Initialize optimizers for both networks
optimizerA = optim.Adam(actor.parameters(), lr=1e-3)
optimizerC = optim.Adam(critic.parameters(), lr=1e-3)

# Hyperparameters
gamma = 0.95
episodes = 500
entropy_beta = 0.1

# Metrics tracking
reward_log = []
actor_loss_log = []
critic_loss_log = []

# Training loop
for ep in range(episodes):
    state = env.reset()
    total_reward = 0.0
    done = False
    quit_iter = False

    log_probs = []
    values = []
    rewards = []

    while not done:
        # Preprocess the grid to state tensor
        state_tensor = to_grid_tensor(state, grid_size, env, device)
        
        # Calculate action probabilities
        action_probs = actor(state_tensor)
        dist = torch.distributions.Categorical(action_probs)
        
        # Sample an action from distribution
        action = dist.sample()
        
        # Calculate log probability
        log_prob = dist.log_prob(action)
        
        # Get critic's value estimate
        value = critic(state_tensor)
        
        # Take step in environment
        next_state, reward, done = env.step(action.item())
        
        # Check for premature episode termination
        if reward == -1 and done:
            quit_iter = True
        
        # Store trajectory data
        total_reward += reward
        log_probs.append(log_prob)
        values.append(value)
        rewards.append(reward)

        state = next_state
    
    # Calculate returns (episodic update)
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    
    # Convert to tensors
    returns = torch.tensor(returns).to(device)
    log_probs = torch.cat(log_probs).to(device)
    values = torch.cat(values).squeeze().to(device)

    # Calculate advantages
    advantage_critic = returns - values
    advantage_actor = advantage_critic.detach()
    
    # Update critic network
    critic_loss = advantage_critic.pow(2).mean()
    optimizerC.zero_grad()
    critic_loss.backward()
    optimizerC.step()
    
    # Update actor network
    entropy = dist.entropy().mean()
    actor_loss = (-log_probs * advantage_actor - entropy_beta * entropy).mean()
    optimizerA.zero_grad()
    actor_loss.backward()
    optimizerA.step()

    # Log metrics
    reward_log.append(total_reward)
    actor_loss_log.append(actor_loss.item())
    critic_loss_log.append(critic_loss.item())

    # Print progress
    if quit_iter:
        print(f"Episode {ep}, Quit Iteration (max steps)")
    if (ep + 1) % 10 == 0:
        print(f"Episode {ep}, Reward: {total_reward:.3f}, "
              f"Critic Loss: {critic_loss.item():.3f}, "
              f"Actor Loss: {actor_loss.item():.3f}")

# Visualization
plt.figure(figsize=(10, 6))
plt.plot(reward_log, label="Total Reward")
plt.plot(actor_loss_log, label="Actor Loss") 
plt.plot(critic_loss_log, label="Critic Loss")
plt.xlabel("Episode")
plt.ylabel("Metrics")
plt.title("Actor-Critic with CNN and Checkpoints")
plt.grid(True)
plt.legend()
plt.savefig(f"Exp_Rewards_vs_Epochs_AC_{episodes}.png")
plt.show()

# Learned policy visualization
actions = ["↑", "↓", "←", "→"]
print("\nLearned Policy:")
actor.eval()

with torch.no_grad():
    original_idx = env.current_checkpoint_idx
    
    for c_idx, cp_pos in enumerate(checkpoints):
        env.current_checkpoint_idx = c_idx 
        
        target_name = f"Checkpoint {c_idx+1}"
        if c_idx == len(checkpoints) - 1:
            target_name = "Final Goal"
            
        print(f"\nPolicy towards {target_name}: {cp_pos}")
        
        for i in range(grid_size[0]):
            row = ""
            for j in range(grid_size[1]):
                s = (i, j)
                if s == cp_pos:
                    row += f" C{c_idx+1} " if c_idx < len(checkpoints) - 1 else " G  "
                else:
                    state_tensor = to_grid_tensor(s, grid_size, env, device)
                    action_probs = actor(state_tensor)
                    best_a = int(torch.argmax(action_probs, dim=1).item())
                    row += f" {actions[best_a]}  "
            print(row)

    env.current_checkpoint_idx = original_idx