import torch
from torch import nn
from typing import Tuple
import torch.optim as optim
from torch.distributions import Categorical

import numpy as np 

import time
import logging

logging.basicConfig(
    level=logging.INFO,  # Show INFO and above (INFO, WARNING, ERROR, CRITICAL)
    format='%(asctime)s - %(levelname)s - %(message)s'
)

NUM_HOUSES = 15
NUM_LOCATIONS = NUM_HOUSES + 1
WAREHOUSE_IDX = 0
COORDINATE_DIM = 2
HIDDEN_SIZE = 256
LEARNING_RATE = 3e-4
EPISODES = 5000
GAMMA = 0.99 

class A2C(nn.Module):
    def __init__(
        self, 
        coordinate_dim: int, 
        hidden_size: int
    ):
        
        super().__init__()

        self.hidden_size = hidden_size
        self.node_embedding = nn.Linear(coordinate_dim, hidden_size)
        
        # context -> (current position + available nodes)
        self.context_encoder = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        
        # actor network
        self.actor_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        
        # critic network
        self.critic_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        
    def forward(
            self, 
            coordinates: torch.Tensor, 
            current_idx: int, 
            visited_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, ...] :
        # types for strong hinting
        node_embeds: torch.Tensor
        available_embed: torch.Tensor
        state_value: torch.Tensor

        # embed all the nodes -> (num_nodes, hidden)
        node_embeds = self.node_embedding(coordinates)
        # get this node's embedding
        current_embed = node_embeds[current_idx]

        # available nodes
        available_mask = ~visited_mask.bool()
        if available_mask.sum() > 0:
            available_embed = node_embeds[available_mask].mean(dim=0) 
        else:
            available_embed = torch.zeros_like(current_embed)
        
        # encode context
        context = torch.cat([current_embed, available_embed])
        context_encoded = self.context_encoder(context)
        
        # logits for actor network
        compatibility = torch.matmul(node_embeds, context_encoded)
        action_logits = compatibility.clone()
        # Invalidate visited nodes
        action_logits[visited_mask] = -float('inf')

        # critic output
        state_value = self.critic_head(context_encoded)
        
        return action_logits, state_value
    

def train_agent(
    distance_matrix: torch.Tensor, 
    coordinates: torch.Tensor
):
    model = A2C(coordinate_dim=COORDINATE_DIM, hidden_size=HIDDEN_SIZE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3000, gamma=0.8)
    
    logging.info("Starting Agent Training")
    best_distance = float('inf')
    running_reward = 0
    
    for episode in range(EPISODES):
        current_loc: int = WAREHOUSE_IDX
        visited_mask = torch.zeros(NUM_LOCATIONS, dtype=torch.bool)
        visited_mask[WAREHOUSE_IDX] = True
        
        log_probs, state_values, rewards = [], [], []
        states = []
        
        action_logits: torch.Tensor
        value: torch.Tensor
        action: torch.Tensor

        for step in range(NUM_HOUSES):
            action_logits, value = model(coordinates, current_loc, visited_mask)
            action_dist = Categorical(logits=action_logits)
            action = action_dist.sample()
            
            log_probs.append(action_dist.log_prob(action))
            state_values.append(value)

            reward = -distance_matrix[current_loc, int(action.item())]
            rewards.append(reward)
            
            # update 
            current_loc = int(action.item())
            next_mask = visited_mask.clone()
            next_mask[current_loc] = True
            visited_mask = next_mask
        
        # return to warehouse
        final_reward = -distance_matrix[current_loc, WAREHOUSE_IDX]
        rewards.append(final_reward)
        
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + GAMMA * G
            returns.insert(0, G)
        
        returns = torch.tensor(returns[:-1], dtype=torch.float32) 
        log_probs = torch.stack(log_probs)
        state_values = torch.cat(state_values).squeeze()
        
        # normalization
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        

        advantage = returns - state_values.detach()
        actor_loss = -(log_probs * advantage).mean()
        critic_loss = nn.functional.mse_loss(state_values, returns)
        entropy = -torch.sum(torch.exp(log_probs) * log_probs)
        entropy_bonus = 0.01 * entropy

        loss = actor_loss + 0.5 * critic_loss - entropy_bonus
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        scheduler.step()
        
        # metrics
        total_distance = -sum(rewards)
        running_reward = 0.95 * running_reward + 0.05 * (-total_distance)
        
        if total_distance < best_distance:
            best_distance = total_distance
        
        if episode % 1000 == 0:
            print(f"Episode {episode} | Distance: {total_distance:.2f} | "
                  f"Best: {best_distance:.2f} | Running Avg: {running_reward:.2f}")

    logging.info("Agent Training completed")
    return model

def test_agent(
    model: A2C, 
    distance_matrix: np.ndarray, 
    coordinates: torch.Tensor
):
    logging.info("Testing learned RL heuristic...")
    
    start_time = time.time()
    
    best_tour = None
    best_distance = float('inf')
    
    for trial in range(10):
        with torch.no_grad():
            current_loc: int = WAREHOUSE_IDX
            visited_mask = torch.zeros(NUM_LOCATIONS, dtype=torch.bool)
            visited_mask[WAREHOUSE_IDX] = True
            tour = [WAREHOUSE_IDX]
            total_distance = 0
            
            action_logits: torch.Tensor
            actin: torch.Tensor

            for _ in range(NUM_HOUSES):
                action_logits, _ = model(coordinates, current_loc, visited_mask)
                
                # greedy
                if trial < 5: 
                    action = action_logits.argmax()
                # stochastic
                else: 
                    action_dist = Categorical(logits=action_logits)
                    action = action_dist.sample()
                
                total_distance += distance_matrix[current_loc, int(action.item())]
                current_loc = int(action.item())
                visited_mask[current_loc] = True
                tour.append(current_loc)
            
            total_distance += distance_matrix[current_loc, WAREHOUSE_IDX]
            tour.append(WAREHOUSE_IDX)
            
            if total_distance < best_distance:
                best_distance = total_distance
                best_tour = tour
    
    rl_inference_time = time.time() - start_time
    logging.info(f"Agent inference time: {rl_inference_time:.4f} seconds")
    
    return best_tour, best_distance, rl_inference_time