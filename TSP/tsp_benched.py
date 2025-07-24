import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import matplotlib.pyplot as plt
from python_tsp.exact import solve_tsp_dynamic_programming

# --- Configuration ---
NUM_HOUSES = 15
NUM_LOCATIONS = NUM_HOUSES + 1
WAREHOUSE_IDX = 0
COORDINATE_DIM = 2
HIDDEN_SIZE = 256
LEARNING_RATE = 3e-4
EPISODES = 5000
GAMMA = 0.99  # Discount factor

def generate_data():
    print("Generating random locations and distance matrix...")
    locations = np.random.rand(NUM_LOCATIONS, 2) * 100
    distance_matrix = np.linalg.norm(locations[:, np.newaxis, :] - locations[np.newaxis, :, :], axis=2)
    return locations, torch.from_numpy(distance_matrix).float()

class AttentionTSP(nn.Module):
    def __init__(self, coordinate_dim, hidden_size, num_heads=8):
        super(AttentionTSP, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        
        # Embedding layers
        self.node_embedding = nn.Linear(coordinate_dim, hidden_size)
        self.current_embedding = nn.Linear(coordinate_dim, hidden_size)
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        
        # Output layers
        self.actor_head = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        
        self.critic_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        
    def forward(self, coordinates, current_idx, visited_mask):
        batch_size = coordinates.shape[0] if len(coordinates.shape) == 3 else 1
        if len(coordinates.shape) == 2:
            coordinates = coordinates.unsqueeze(0)
            current_idx = current_idx.unsqueeze(0)
            visited_mask = visited_mask.unsqueeze(0)
        
        # Embed all nodes
        node_embeds = self.node_embedding(coordinates)  # [batch, num_nodes, hidden]
        
        # Current node embedding
        current_coords = coordinates.gather(1, current_idx.unsqueeze(-1).expand(-1, -1, 2))
        current_embed = self.current_embedding(current_coords)  # [batch, 1, hidden]
        
        # Attention mechanism
        attended, _ = self.attention(current_embed, node_embeds, node_embeds)
        
        # Combine current and attended features
        combined = torch.cat([current_embed, attended], dim=-1)  # [batch, 1, hidden*2]
        
        # Actor: compute action logits
        action_logits = self.actor_head(combined).squeeze(-1)  # [batch, 1]
        
        # Expand to all nodes
        action_logits = action_logits.expand(batch_size, coordinates.shape[1])
        
        # Mask visited nodes
        action_logits = action_logits.masked_fill(visited_mask.bool(), -float('inf'))
        
        # Critic: compute state value
        state_value = self.critic_head(attended.mean(dim=1))  # [batch, 1]
        
        if batch_size == 1:
            action_logits = action_logits.squeeze(0)
            state_value = state_value.squeeze(0)
        
        return action_logits, state_value

class ImprovedActorCritic(nn.Module):
    def __init__(self, coordinate_dim, hidden_size):
        super(ImprovedActorCritic, self).__init__()
        self.hidden_size = hidden_size
        
        # Node embedding
        self.node_embedding = nn.Linear(coordinate_dim, hidden_size)
        
        # Context encoding (current position + available nodes)
        self.context_encoder = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        
        # Actor head
        self.actor_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        
        # Critic head
        self.critic_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        
    def forward(self, coordinates, current_idx, visited_mask):
        # Embed all nodes
        node_embeds = self.node_embedding(coordinates)  # [num_nodes, hidden]
        
        # Current node embedding
        current_embed = node_embeds[current_idx]  # [hidden]
        
        # Available nodes embedding (mean of unvisited)
        available_mask = ~visited_mask.bool()
        if available_mask.sum() > 0:
            available_embed = node_embeds[available_mask].mean(dim=0)  # [hidden]
        else:
            available_embed = torch.zeros_like(current_embed)
        
        # Context encoding
        context = torch.cat([current_embed, available_embed])  # [hidden*2]
        context_encoded = self.context_encoder(context)  # [hidden]
        
        # Actor: compute logits for each node
        action_logits = torch.zeros(coordinates.shape[0])
        for i in range(coordinates.shape[0]):
            if not visited_mask[i]:
                # Compute compatibility between current context and target node
                compatibility = torch.sum(context_encoded * node_embeds[i])
                action_logits[i] = compatibility
            else:
                action_logits[i] = -float('inf')
        
        # Critic: state value
        state_value = self.critic_head(context_encoded)
        
        return action_logits, state_value

def train_rl_agent(distance_matrix, coordinates):
    model = ImprovedActorCritic(coordinate_dim=COORDINATE_DIM, hidden_size=HIDDEN_SIZE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3000, gamma=0.8)
    
    print("\n🚀 Starting improved RL agent training...")
    best_distance = float('inf')
    running_reward = 0
    
    for episode in range(EPISODES):
        current_loc = WAREHOUSE_IDX
        visited_mask = torch.zeros(NUM_LOCATIONS, dtype=torch.bool)
        visited_mask[WAREHOUSE_IDX] = True
        
        log_probs, state_values, rewards = [], [], []
        states = []
        
        # Episode rollout
        for step in range(NUM_HOUSES):
            action_logits, value = model(coordinates, current_loc, visited_mask)
            action_dist = Categorical(logits=action_logits)
            action = action_dist.sample()
            
            log_probs.append(action_dist.log_prob(action))
            state_values.append(value)
            
            # Reward: negative distance (we want to minimize distance)
            reward = -distance_matrix[current_loc, action.item()]
            rewards.append(reward)
            
            # Update state
            current_loc = action.item()
            visited_mask[current_loc] = True
        
        # Return to warehouse
        final_reward = -distance_matrix[current_loc, WAREHOUSE_IDX]
        rewards.append(final_reward)
        
        # Calculate returns with discount
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + GAMMA * G
            returns.insert(0, G)
        
        returns = torch.tensor(returns[:-1], dtype=torch.float32)  # Exclude final return
        log_probs = torch.stack(log_probs)
        state_values = torch.cat(state_values).squeeze()
        
        # Normalize returns
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Calculate advantage
        advantage = returns - state_values.detach()
        
        # Actor loss (policy gradient)
        actor_loss = -(log_probs * advantage).mean()
        
        # Critic loss (value function)
        critic_loss = nn.functional.mse_loss(state_values, returns)
        
        # Entropy bonus for exploration
        entropy = -torch.sum(torch.exp(log_probs) * log_probs)
        entropy_bonus = 0.01 * entropy
        
        # Total loss
        loss = actor_loss + 0.5 * critic_loss - entropy_bonus
        
        # Optimization step
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)  # Gradient clipping
        optimizer.step()
        scheduler.step()
        
        # Track performance
        total_distance = -sum(rewards)
        running_reward = 0.95 * running_reward + 0.05 * (-total_distance)
        
        if total_distance < best_distance:
            best_distance = total_distance
        
        if episode % 1000 == 0:
            print(f"Episode {episode} | Distance: {total_distance:.2f} | "
                  f"Best: {best_distance:.2f} | Running Avg: {running_reward:.2f}")

    print("✅ Improved RL Training complete.")
    return model

def test_rl_agent(model, distance_matrix, coordinates):
    print("🧪 Testing learned RL heuristic...")
    
    best_tour = None
    best_distance = float('inf')
    
    # Run multiple trials and take the best
    for trial in range(10):
        with torch.no_grad():
            current_loc = WAREHOUSE_IDX
            visited_mask = torch.zeros(NUM_LOCATIONS, dtype=torch.bool)
            visited_mask[WAREHOUSE_IDX] = True
            tour = [WAREHOUSE_IDX]
            total_distance = 0
            
            for _ in range(NUM_HOUSES):
                action_logits, _ = model(coordinates, current_loc, visited_mask)
                
                # Use both greedy and stochastic selection
                if trial < 5:  # Greedy trials
                    action = action_logits.argmax()
                else:  # Stochastic trials
                    action_dist = Categorical(logits=action_logits)
                    action = action_dist.sample()
                
                total_distance += distance_matrix[current_loc, action.item()]
                current_loc = action.item()
                visited_mask[current_loc] = True
                tour.append(current_loc)
            
            total_distance += distance_matrix[current_loc, WAREHOUSE_IDX]
            tour.append(WAREHOUSE_IDX)
            
            if total_distance < best_distance:
                best_distance = total_distance
                best_tour = tour
    
    return best_tour, best_distance.item()

def solve_exact(distance_matrix):
    print("🔍 Solving for the exact global minimum...")
    distance_matrix_np = distance_matrix.numpy()
    permutation, distance = solve_tsp_dynamic_programming(distance_matrix_np)
    full_tour = permutation + [permutation[0]]
    return full_tour, distance

def local_search_2opt(tour, distance_matrix):
    """Apply 2-opt local search to improve the tour"""
    improved = True
    best_distance = sum(distance_matrix[tour[i], tour[i+1]] for i in range(len(tour)-1))
    
    while improved:
        improved = False
        for i in range(1, len(tour)-2):
            for j in range(i+1, len(tour)-1):
                # Try swapping edges
                new_tour = tour[:i] + tour[i:j+1][::-1] + tour[j+1:]
                new_distance = sum(distance_matrix[new_tour[k], new_tour[k+1]] for k in range(len(new_tour)-1))
                
                if new_distance < best_distance:
                    tour = new_tour
                    best_distance = new_distance
                    improved = True
                    break
            if improved:
                break
    
    return tour, best_distance

def visualize_results(locations, rl_tour, rl_dist, opt_tour, opt_dist):
    """Plots the locations and the two paths for comparison."""
    print("🎨 Generating visualization...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle('TSP Solution Comparison', fontsize=16)

    # --- Plot RL Heuristic Path ---
    ax1.set_title(f'RL Heuristic Path\nTotal Distance: {rl_dist:.2f}')
    for i in range(len(rl_tour) - 1):
        start_node = rl_tour[i]
        end_node = rl_tour[i+1]
        ax1.plot(*zip(locations[start_node], locations[end_node]), 'b-', linewidth=2)
    ax1.scatter(locations[:, 0], locations[:, 1], c='lightblue', s=100, edgecolors='black')
    ax1.scatter(locations[WAREHOUSE_IDX, 0], locations[WAREHOUSE_IDX, 1], c='red', s=200, marker='*')
    for i, txt in enumerate(range(NUM_LOCATIONS)):
        ax1.annotate(txt, (locations[i, 0], locations[i, 1]), ha='center', va='center', fontweight='bold')
    ax1.set_xlabel("X Coordinate")
    ax1.set_ylabel("Y Coordinate")
    ax1.grid(True, alpha=0.3)

    # --- Plot Optimal Path ---
    ax2.set_title(f'Optimal Path (Exact Solver)\nTotal Distance: {opt_dist:.2f}')
    for i in range(len(opt_tour) - 1):
        start_node = opt_tour[i]
        end_node = opt_tour[i+1]
        ax2.plot(*zip(locations[start_node], locations[end_node]), 'g-', linewidth=2)
    ax2.scatter(locations[:, 0], locations[:, 1], c='lightgreen', s=100, edgecolors='black')
    ax2.scatter(locations[WAREHOUSE_IDX, 0], locations[WAREHOUSE_IDX, 1], c='red', s=200, marker='*')
    for i, txt in enumerate(range(NUM_LOCATIONS)):
        ax2.annotate(txt, (locations[i, 0], locations[i, 1]), ha='center', va='center', fontweight='bold')
    ax2.set_xlabel("X Coordinate")
    ax2.set_ylabel("Y Coordinate")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

if __name__ == "__main__":
    locations, distance_matrix = generate_data()
    coordinates = torch.from_numpy(locations).float()
    
    rl_model = train_rl_agent(distance_matrix, coordinates)
    rl_tour, rl_distance = test_rl_agent(rl_model, distance_matrix, coordinates)
    
    # Apply 2-opt improvement
    print("🔧 Applying 2-opt local search...")
    rl_tour_improved, rl_distance_improved = local_search_2opt(rl_tour, distance_matrix)
    
    optimal_tour, optimal_distance = solve_exact(distance_matrix)
    
    print("\n--- Final Results ---")
    print(f"RL Heuristic Path Distance:           {rl_distance:.2f}")
    print(f"RL + 2-opt Improved Distance:         {rl_distance_improved:.2f}")
    print(f"Optimal Path Distance:                {optimal_distance:.2f}")
    
    optimality_gap = ((rl_distance_improved - optimal_distance) / optimal_distance) * 100
    print(f"Optimality Gap (after 2-opt): {optimality_gap:.2f}%")
    
    visualize_results(locations, rl_tour_improved, rl_distance_improved, optimal_tour, optimal_distance)