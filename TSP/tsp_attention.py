import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import matplotlib.pyplot as plt
from python_tsp.exact import solve_tsp_dynamic_programming
import math

# -------------------------------------------------------------------
# ## 1. Configuration & Data Generation 🗺️
# -------------------------------------------------------------------

# --- Configuration ---
NUM_HOUSES = 15
NUM_LOCATIONS = NUM_HOUSES + 1
WAREHOUSE_IDX = 0

# Model Hyperparameters
EMBED_DIM = 128
NUM_HEADS = 8
NUM_ENCODER_LAYERS = 3
HIDDEN_DIM = 512

# Training Hyperparameters
LEARNING_RATE = 1e-4
EPISODES = 5000


def generate_data():
    """Generates random coordinates and a distance matrix."""
    print("Generating random locations and distance matrix...")
    locations = np.random.rand(NUM_LOCATIONS, 2) * 100
    distance_matrix = np.linalg.norm(locations[:, np.newaxis, :] - locations[np.newaxis, :, :], axis=2)
    return torch.from_numpy(locations).float(), torch.from_numpy(distance_matrix).float()


# -------------------------------------------------------------------
# ## 2. The Attention Model (Encoder-Decoder) 🧠
# -------------------------------------------------------------------

class AttentionModel(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers):
        super().__init__()
        
        self.initial_embed = nn.Linear(2, embed_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=HIDDEN_DIM, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.decoder_mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        
        self.value_head = nn.Sequential(
            nn.Linear(embed_dim, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, 1)
        )

    # Note the updated signature: we now accept current_loc
    def forward(self, locations, current_loc, visited_mask):
        initial_embeddings = self.initial_embed(locations)
        encoder_output = self.encoder(initial_embeddings.unsqueeze(0))

        # LOGIC FIX: Use current_loc to get the last node's embedding accurately
        last_node_embedding = encoder_output[:, current_loc, :].unsqueeze(1)
        
        _, attn_weights = self.decoder_mha(query=last_node_embedding, key=encoder_output, value=encoder_output)
        
        attn_weights = attn_weights.squeeze(1)

        # INDEXERROR FIX: Use masked_fill_ with an unsqueezed mask
        attn_weights = attn_weights.masked_fill(visited_mask.unsqueeze(0).bool(), -float('inf'))

        action_dist = Categorical(logits=attn_weights)
        
        graph_embedding = encoder_output.mean(dim=1)
        state_value = self.value_head(graph_embedding)
        
        return action_dist, state_value

# -------------------------------------------------------------------
# ## 3. Training, Solving, and Visualization Functions ⚙️
# -------------------------------------------------------------------

def train_rl_agent(locations, distance_matrix):
    """The main RL training loop for the attention model."""
    model = AttentionModel(embed_dim=EMBED_DIM, num_heads=NUM_HEADS, num_layers=NUM_ENCODER_LAYERS)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    print("\n🚀 Starting RL agent training (Attention Model)...")

    for episode in range(EPISODES):
        current_loc = WAREHOUSE_IDX
        visited_mask = torch.zeros(NUM_LOCATIONS, dtype=torch.uint8)
        visited_mask[WAREHOUSE_IDX] = 1
        
        log_probs, rewards = [], []
        state_value = None 

        for _ in range(NUM_HOUSES):
            # Pass current_loc to the model
            action_dist, value = model(locations, current_loc, visited_mask)
            if state_value is None:
                state_value = value

            action = action_dist.sample()
            log_probs.append(action_dist.log_prob(action))
            reward = -distance_matrix[current_loc, action.item()]
            rewards.append(reward)
            current_loc = action.item()
            visited_mask[current_loc] = 1

        final_reward = -distance_matrix[current_loc, WAREHOUSE_IDX]
        rewards.append(final_reward)

        total_reward = torch.tensor(rewards).sum()
        log_probs = torch.stack(log_probs)
        
        advantage = total_reward - state_value.squeeze().detach()
        
        actor_loss = -(log_probs * advantage).mean()
        critic_loss = nn.functional.mse_loss(state_value.squeeze(), total_reward)

        loss = actor_loss + critic_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if episode % 500 == 0:
            print(f"Episode {episode} | Approx. Distance: {-total_reward:.2f}")

    print("✅ RL Training complete.")
    return model

def test_rl_agent(model, locations, distance_matrix):
    """Uses the trained RL agent to find a single, good path."""
    print("🧪 Testing learned RL heuristic...")
    with torch.no_grad():
        current_loc = WAREHOUSE_IDX
        visited_mask = torch.zeros(NUM_LOCATIONS, dtype=torch.uint8)
        visited_mask[WAREHOUSE_IDX] = 1
        tour = [WAREHOUSE_IDX]
        total_distance = 0
        for _ in range(NUM_HOUSES):
            # Pass current_loc to the model
            action_dist, _ = model(locations, current_loc, visited_mask)
            action = action_dist.probs.argmax()
            total_distance += distance_matrix[current_loc, action.item()]
            current_loc = action.item()
            visited_mask[current_loc] = 1
            tour.append(current_loc)
        total_distance += distance_matrix[current_loc, WAREHOUSE_IDX]
        tour.append(WAREHOUSE_IDX)
    return tour, total_distance.item()

def solve_exact(distance_matrix):
    """Finds the guaranteed optimal solution using a dedicated solver."""
    print("🔍 Solving for the exact global minimum...")
    permutation, distance = solve_tsp_dynamic_programming(distance_matrix)
    full_tour = permutation + [permutation[0]]
    return full_tour, distance

def visualize_results(locations, rl_tour, rl_dist, opt_tour, opt_dist):
    """Plots the locations and the two paths for comparison."""
    print("🎨 Generating visualization...")
    locations_np = locations.numpy()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle('TSP Solution Comparison (Attention Model)', fontsize=16)

    # Plot RL Heuristic Path
    ax1.set_title(f'RL Heuristic Path\nTotal Distance: {rl_dist:.2f}')
    for i in range(len(rl_tour) - 1):
        ax1.plot(*zip(locations_np[rl_tour[i]], locations_np[rl_tour[i+1]]), 'b-')
    ax1.scatter(locations_np[:, 0], locations_np[:, 1], c='lightblue', s=100)
    ax1.scatter(locations_np[WAREHOUSE_IDX, 0], locations_np[WAREHOUSE_IDX, 1], c='red', s=200, marker='*')
    for i in range(NUM_LOCATIONS): ax1.annotate(i, (locations_np[i, 0], locations_np[i, 1]), ha='center', va='center')
    ax1.set_xlabel("X Coordinate"); ax1.set_ylabel("Y Coordinate")

    # Plot Optimal Path
    ax2.set_title(f'Optimal Path (Exact Solver)\nTotal Distance: {opt_dist:.2f}')
    for i in range(len(opt_tour) - 1):
        ax2.plot(*zip(locations_np[opt_tour[i]], locations_np[opt_tour[i+1]]), 'g-')
    ax2.scatter(locations_np[:, 0], locations_np[:, 1], c='lightgreen', s=100)
    ax2.scatter(locations_np[WAREHOUSE_IDX, 0], locations_np[WAREHOUSE_IDX, 1], c='red', s=200, marker='*')
    for i in range(NUM_LOCATIONS): ax2.annotate(i, (locations_np[i, 0], locations_np[i, 1]), ha='center', va='center')
    ax2.set_xlabel("X Coordinate")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

# -------------------------------------------------------------------
# ## 4. Main Execution Block
# -------------------------------------------------------------------

if __name__ == "__main__":
    locations, distance_matrix = generate_data()
    rl_model = train_rl_agent(locations, distance_matrix)
    rl_tour, rl_distance = test_rl_agent(rl_model, locations, distance_matrix)
    optimal_tour, optimal_distance = solve_exact(distance_matrix)
    
    print("\n--- Final Results ---")
    print(f"RL Heuristic Path Distance: {rl_distance:.2f}")
    print(f"Optimal Path Distance:      {optimal_distance:.2f}")
    
    optimality_gap = ((rl_distance - optimal_distance) / optimal_distance) * 100
    print(f"Optimality Gap: {optimality_gap:.2f}%")
    
    visualize_results(locations, rl_tour, rl_distance, optimal_tour, optimal_distance)