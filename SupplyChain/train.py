from torch.optim import Adam
from supply_chain_sim import SupplyChainSimulator
import simpy
from model import ActorAndCritic
from ppo import RolloutBuffer, ppo_update
import numpy as np
from plot import plot_episode_summary, plot_training_progress

# initialize environment & models
env = SupplyChainSimulator(env=simpy.Environment())
obs_dim = env.get_observation().shape[0]
num_warehouses = env.warehouse_count
num_stores = env.stores_count

model = ActorAndCritic(
    observation_dim=obs_dim, 
    num_warehouses=num_warehouses, 
    num_stores=num_stores)
optimizer = Adam(model.parameters(), lr=1e-4)

# ppo parameters
PPO_CLIP_EPSILON = 0.2
PPO_VF_COEF = 0.5
PPO_ENTROPY_COEF = 0.05
PPO_MAX_GRAD_NORM = 0.5
ROLLOUT_STEPS = 1024

# for visualization
rewards_history = []
avg_values_history = []

for episode in range(1000):
    # initialize new environment
    env = SupplyChainSimulator(env=simpy.Environment())
    obs = env.get_observation()

    initial_warehouse_levels = [w.level for w in env.warehouses]
    initial_store_levels = [s.level for s, _ in env.stores]

    buffer = RolloutBuffer()

    for step in range(ROLLOUT_STEPS):
        action, logprob, value = model.sample_action(obs)
        
        next_obs, reward, diagnostics = env.step(action)
        done = step == ROLLOUT_STEPS - 1

        buffer.store(obs, action, logprob, reward, value, done, diagnostics)
        obs = next_obs

        if step > 0 and step % 500 == 0:
            print(f"Step {step}: Op={action['operation']}, Target={action['target_id']}, Qty={action['quantity']:.1f}, Reward={reward:.2f}, Value={value.item():.2f}")

    final_warehouse_levels = [w.level for w in env.warehouses]
    final_store_levels = [s.level for s, _ in env.stores]

    total_reward = np.sum(buffer.rewards)
    rewards_history.append(total_reward)
    avg_value = np.mean(buffer.values)
    avg_values_history.append(avg_value)

    if len(rewards_history) >= 10:
        recent_avg = np.mean(rewards_history[-10:])
        print(f"[{episode}] Total Reward: {total_reward:.2f}, Recent Avg: {recent_avg:.2f}, Avg Value: {avg_value:.2f}\n")
    else:
        print(f"[{episode}] Total Reward: {total_reward:.2f}, Avg Value: {avg_value:.2f}\n")

    # Plotting functions are called before the buffer is cleared
    if episode % 50 == 0:
        plot_episode_summary(
            episode, buffer,
            initial_warehouse_levels, final_warehouse_levels,
            initial_store_levels, final_store_levels
        )

    if episode % 250 == 0 and episode > 0:
        plot_training_progress(rewards_history, avg_values_history, episode)
    
    # PPO update will use and then clear the buffer
    ppo_update(
        model,
        optimizer,
        buffer,
        clip_epsilon=PPO_CLIP_EPSILON,
        vf_coef=PPO_VF_COEF,
        entropy_coef=PPO_ENTROPY_COEF,
        max_grad_norm=PPO_MAX_GRAD_NORM
    )

print("Training completed!")
plot_training_progress(rewards_history, avg_values_history, len(rewards_history))