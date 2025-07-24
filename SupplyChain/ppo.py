import torch
from torch import nn
import numpy as np

from model import ActorAndCritic

class RolloutBuffer:
    def __init__(self):
        self.states = []
        self.operations = []
        self.target_ids = []
        self.quantities = []
        self.logprobs = []
        self.rewards = []
        self.values = []
        self.dones = []
        self.diagnostics = []

    def store(self, state, action, logprob, reward, value, done, diagnostics):
        self.states.append(state)
        self.operations.append(action["operation"])
        self.target_ids.append(action["target_id"])
        self.quantities.append(action["quantity"])
        self.logprobs.append(logprob.item())
        self.values.append(value.item())
        self.rewards.append(reward)
        self.dones.append(done)
        self.diagnostics.append(diagnostics)

    def clear(self):
        self.__init__()

def compute_returns_and_advantages(rewards, values, dones, gamma=0.99, lam=0.95):
    """Compute returns and advantages using GAE"""
    returns = []
    advantages = []
    gae = 0
    next_value = 0

    for t in reversed(range(len(rewards))):
        mask = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * mask - values[t]
        gae = delta + gamma * lam * mask * gae
        advantages.insert(0, gae)
        next_value = values[t]
        returns.insert(0, gae + values[t])

    advantages = torch.tensor(advantages, dtype=torch.float32)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    return torch.tensor(returns, dtype=torch.float32), advantages

def ppo_update(
    model: ActorAndCritic, 
    optimizer: torch.optim.Optimizer, 
    rollout: RolloutBuffer, 
    clip_epsilon=0.2, 
    vf_coef=0.5, 
    entropy_coef=0.01, 
    max_grad_norm=0.5, 
    ppo_epochs=4
):
    states = torch.tensor(np.array(rollout.states), dtype=torch.float32)
    operations = torch.tensor(rollout.operations, dtype=torch.int64)
    target_ids = torch.tensor(rollout.target_ids, dtype=torch.int64)
    quantities = torch.tensor(rollout.quantities, dtype=torch.float32)
    old_logprobs = torch.tensor(rollout.logprobs, dtype=torch.float32)
    old_values = torch.tensor(np.array(rollout.values), dtype=torch.float32)
    
    returns, advantages = compute_returns_and_advantages(
        np.array(rollout.rewards, dtype=np.float32),
        np.array(rollout.values, dtype=np.float32),
        np.array(rollout.dones, dtype=np.float32)
    )

    returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    
    total_loss = 0
    policy_losses, value_losses, entropy_losses = [], [], []

    for epoch in range(ppo_epochs):
        logprobs, values, entropies = model.evaluate_action(
            states, 
            operations, 
            target_ids, 
            quantities
        )

        ratios = torch.exp(logprobs - old_logprobs)
        
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        values_clipped = old_values + torch.clamp(values - old_values, -clip_epsilon, clip_epsilon)
        vf_loss_unclipped = nn.functional.mse_loss(values, returns)
        vf_loss_clipped = nn.functional.mse_loss(values_clipped, returns)
        value_loss = 0.5 * torch.max(vf_loss_unclipped, vf_loss_clipped)

        entropy_loss = -entropies.mean()
        loss = policy_loss + vf_coef * value_loss + entropy_coef * entropy_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        total_loss += loss.item()
        policy_losses.append(policy_loss.item())
        value_losses.append(value_loss.item())
        entropy_losses.append(entropy_loss.item())

    print(f"    Policy Loss: {np.mean(policy_losses):.4f}, "
            f"Value Loss: {np.mean(value_losses):.4f}, "
            f"Entropy Loss: {np.mean(entropy_losses):.4f}")
    
    rollout.clear()
    return total_loss / ppo_epochs