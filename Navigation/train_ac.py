import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from world import GridWorld
from a2c import ConvActor, ConvCritic, to_grid_tensor

# environment setup
grid_size = (8, 8)
checkpoints = [(2, 7), (6, 1), (7, 7)]
num_actions = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = GridWorld(size=grid_size, checkpoints=checkpoints)
H, W = grid_size

# initialize actor and critic with improved architecture
actor = ConvActor(H, W, num_actions).to(device)
critic = ConvCritic(H, W).to(device)

# improved optimizers with gradient clipping
optimizerA = optim.Adam(actor.parameters(), lr=3e-4, eps=1e-5)
optimizerC = optim.Adam(critic.parameters(), lr=1e-3, eps=1e-5)

# improved hyperparameters
gamma = 0.90
episodes = 100
entropy_beta = 0.001  
gae_lambda = 0.95   
grad_clip = 0.5 
value_loss_coeff = 0.5

# metrics
reward_log = []
actor_loss_log = []
critic_loss_log = []
entropy_log = []

# generalized advantage estimation
def compute_gae(rewards, values, next_value, gamma, gae_lambda):
    values = values + [next_value]
    gae = 0
    advantages = []
    
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] - values[step]
        gae = delta + gamma * gae_lambda * gae
        advantages.insert(0, gae)
    
    return advantages

# train loop
for ep in range(episodes):
    state = env.reset()
    total_reward = 0.0
    done = False
    quit_iter = False

    log_probs = []
    values = []
    rewards = []
    entropies = []

    while not done:
        # preprocess the grid to state
        state_tensor = to_grid_tensor(state, grid_size, env, device)
        
        # calculate action probabilities
        action_probs = actor(state_tensor)
        dist = torch.distributions.Categorical(action_probs)
        
        # sample an action from distribution
        action = dist.sample()
        
        # calculate log probability and entropy
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        
        # get critic's value
        value = critic(state_tensor)
        
        # advance in environment
        next_state, reward, done = env.step(action.item())
        
        # premature episode quitting
        if reward == -1 and done:
            quit_iter = True
        
        # store trajectory
        total_reward += reward
        log_probs.append(log_prob)
        values.append(value.squeeze())
        rewards.append(reward)
        entropies.append(entropy)

        state = next_state
    
    # calculate final value for GAE
    if done and not quit_iter:
        next_value = 0.0
    else:
        with torch.no_grad():
            final_state_tensor = to_grid_tensor(state, grid_size, env, device)
            next_value = critic(final_state_tensor).squeeze().item()
    
    # compute advantage
    value_list = [v.item() for v in values]
    advantages = compute_gae(rewards, value_list, next_value, gamma, gae_lambda)
    
    # convert to tensors
    advantages = torch.tensor(advantages, dtype=torch.float32).to(device)
    returns = advantages + torch.tensor(value_list, dtype=torch.float32).to(device)
    
    # normalize advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    log_probs = torch.stack(log_probs)
    values = torch.stack(values)
    entropies = torch.stack(entropies)
    
    # calculate losses
    critic_loss = value_loss_coeff * ((returns - values) ** 2).mean()
    actor_loss = -(log_probs * advantages).mean()
    entropy_loss = -entropy_beta * entropies.mean()
    total_actor_loss = actor_loss + entropy_loss
    
    # update critic
    optimizerC.zero_grad()
    critic_loss.backward()
    torch.nn.utils.clip_grad_norm_(critic.parameters(), grad_clip)
    optimizerC.step()
    
    # update actor
    optimizerA.zero_grad()
    total_actor_loss.backward()
    torch.nn.utils.clip_grad_norm_(actor.parameters(), grad_clip)
    optimizerA.step()

    # metrics
    reward_log.append(total_reward)
    actor_loss_log.append(total_actor_loss.item())
    critic_loss_log.append(critic_loss.item())
    entropy_log.append(entropies.mean().item())

    if quit_iter:
        print(f"Episode {ep}, Quit Iteration (max steps)")
    if (ep + 1) % 10 == 0:
        avg_reward = np.mean(reward_log[-10:])
        print(f"Episode {ep}, Reward: {total_reward:.3f}, Avg Reward: {avg_reward:.3f}, "
              f"Critic Loss: {critic_loss.item():.3f}, Actor Loss: {total_actor_loss.item():.3f}, "
              f"Entropy: {entropies.mean().item():.3f}")

# visualizing
plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
plt.plot(reward_log, alpha=0.7, label="Episode Reward")

# moving average
window = 20
if len(reward_log) >= window:
    moving_avg = np.convolve(reward_log, np.ones(window)/window, mode='valid')
    plt.plot(range(window-1, len(reward_log)), moving_avg, 'r-', label=f'{window}-Episode Moving Average')

plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Training Rewards")
plt.legend()
plt.grid(True)

plt.subplot(2, 2, 2)
plt.plot(actor_loss_log, label="Actor Loss")
plt.xlabel("Episode")
plt.ylabel("Loss")
plt.title("Actor Loss")
plt.legend()
plt.grid(True)

plt.subplot(2, 2, 3)
plt.plot(critic_loss_log, label="Critic Loss")
plt.xlabel("Episode")
plt.ylabel("Loss")
plt.title("Critic Loss")
plt.legend()
plt.grid(True)

plt.subplot(2, 2, 4)
plt.plot(entropy_log, label="Entropy")
plt.xlabel("Episode")
plt.ylabel("Entropy")
plt.title("Policy Entropy")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig(f"Improved_AC_Training_{episodes}.png", dpi=300)
plt.show()

# learning policy visualization
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