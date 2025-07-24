import torch
import numpy as np
import matplotlib.pyplot as plt
from python_tsp.exact import solve_tsp_dynamic_programming
from model import A2C, train_agent, test_agent
from road_network import generate_network

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

def solve_exact(distance_matrix: np.ndarray):
    logging.info("Solving for the exact global minimum...")

    start_time = time.time()
    permutation, distance = solve_tsp_dynamic_programming(distance_matrix)
    full_tour = permutation + [permutation[0]]
    
    exact_solve_time = time.time() - start_time
    logging.info(f"Exact solver time: {exact_solve_time:.4f} seconds")
    
    return full_tour, distance, exact_solve_time

def local_search_2opt(tour, distance_matrix):
    start_time = time.time()
    
    improved = True
    best_distance = sum(distance_matrix[tour[i], tour[i+1]] for i in range(len(tour)-1))
    
    while improved:
        improved = False
        for i in range(1, len(tour)-2):
            for j in range(i+1, len(tour)-1):
                new_tour = tour[:i] + tour[i:j+1][::-1] + tour[j+1:]
                new_distance = sum(distance_matrix[new_tour[k], new_tour[k+1]] for k in range(len(new_tour)-1))
                
                if new_distance < best_distance:
                    tour = new_tour
                    best_distance = new_distance
                    improved = True
                    break
            if improved:
                break
    
    local_search_time = time.time() - start_time
    logging.info(f"2-opt local search time: {local_search_time:.4f} seconds")
    
    return tour, best_distance, local_search_time

def benchmark_scalability_2(
    model: A2C
):    
    logging.info("\n----- Benchmarking scalability across different problem sizes... -----")
    
    sizes = [10, 15, 20, 50, 100, 300, 500]
    rl_times = []
    exact_times = []
    
    for size in sizes:
        print(f"\n--- Testing with {size} nodes ---")
        # generate problem
        locations = np.random.rand(size, 2) * 100
        distance_matrix = np.linalg.norm(locations[:, np.newaxis, :] - locations[np.newaxis, :, :], axis=2)
        distance_matrix = torch.from_numpy(distance_matrix).float()
        coordinates = torch.from_numpy(locations).float()
        model = model

        # agent inference
        start_time = time.time()
        with torch.no_grad():
            current_loc = 0
            visited_mask = torch.zeros(size, dtype=torch.bool)
            visited_mask[0] = True
            
            for _ in range(size-1):
                action_logits, _ = model(coordinates, current_loc, visited_mask)
                action = action_logits.argmax()
                current_loc = action.item()
                visited_mask[current_loc] = True
        
        rl_time = time.time() - start_time
        rl_times.append(rl_time)
        
        # exact solver
        start_time = time.time()
        exact_time = 0.0
        if size < 20:
            try:
                solve_tsp_dynamic_programming(distance_matrix.numpy())
                exact_time = time.time() - start_time
                exact_times.append(exact_time)
            except:
                exact_times.append(float('inf'))
                exact_time = float('inf')
        
        logging.info(f"Agent inference: {rl_time:.4f}s | Exact solver: {exact_time:.4f}s | "
              f"Speedup: {exact_time/rl_time:.1f}x")
    
    return sizes, rl_times, exact_times

def visualize_results(locations, rl_tour, rl_dist, opt_tour, opt_dist):
    """Plots the locations and the two paths for comparison."""
    print("Generating visualization...")
    
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

def plot_timing_comparison(sizes, rl_times, exact_times):
    """Plot timing comparison between RL and exact solver"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Linear scale
    ax1.plot(sizes, rl_times, 'b-o', label='RL Inference', linewidth=2, markersize=8)
    ax1.plot(sizes, exact_times, 'r-s', label='Exact Solver', linewidth=2, markersize=8)
    ax1.set_xlabel('Problem Size (Number of Nodes)')
    ax1.set_ylabel('Time (seconds)')
    ax1.set_title('Inference Time Comparison (Linear Scale)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Log scale
    ax2.semilogy(sizes, rl_times, 'b-o', label='RL Inference', linewidth=2, markersize=8)
    ax2.semilogy(sizes, exact_times, 'r-s', label='Exact Solver', linewidth=2, markersize=8)
    ax2.set_xlabel('Problem Size (Number of Nodes)')
    ax2.set_ylabel('Time (seconds) - Log Scale')
    ax2.set_title('Inference Time Comparison (Log Scale)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":

    locations, distance_matrix = generate_network()
    distance_matrix_tensor = torch.from_numpy(distance_matrix).float()
    coordinates = torch.from_numpy(locations).float()
    
    logging.info("=== Training Agent ===")
    rl_model = train_agent(distance_matrix_tensor, coordinates)
    rl_tour, rl_distance, rl_time = test_agent(rl_model, distance_matrix, coordinates)
    
    logging.info("Applying 2-opt local search...")
    rl_tour_improved, rl_distance_improved, local_search_time = local_search_2opt(rl_tour, distance_matrix)
    optimal_tour, optimal_distance, exact_time = solve_exact(distance_matrix)
    
    logging.info("--- TIMING ANALYSIS ---")
    total_rl_time = rl_time + local_search_time
    speedup = exact_time / rl_time
    speedup_with_local_search = exact_time / total_rl_time
    
    logging.info(f"RL Inference Time:           {rl_time:.4f} seconds")
    logging.info(f"2-opt Local Search Time:     {local_search_time:.4f} seconds")
    logging.info(f"Total RL + 2-opt Time:       {total_rl_time:.4f} seconds")
    logging.info(f"Exact Solver Time:           {exact_time:.4f} seconds")
    logging.info(f"Speedup (RL only):           {speedup:.1f}x faster")
    logging.info(f"Speedup (RL + 2-opt):        {speedup_with_local_search:.1f}x faster")
    logging.info("\n--- SOLUTION QUALITY ---")
    logging.info(f"RL Heuristic Path Distance:           {rl_distance:.2f}")
    logging.info(f"RL + 2-opt Improved Distance:         {rl_distance_improved:.2f}")
    logging.info(f"Optimal Path Distance:                {optimal_distance:.2f}")
    
    optimality_gap = ((rl_distance_improved - optimal_distance) / optimal_distance) * 100
    logging.info(f"Optimality Gap (after 2-opt):         {optimality_gap:.2f}%")
    
    logging.info(f"\n--- QUALITY-SPEED TRADEOFF ---")
    logging.info(f"RL achieves {100-optimality_gap:.1f}% of optimal quality")
    logging.info(f"at {speedup_with_local_search:.1f}x the speed!")
    
    visualize_results(locations, rl_tour_improved, rl_distance_improved, optimal_tour, optimal_distance)
    
    # Run scalability benchmark
    print("\n" + "="*50)
    sizes, rl_times, exact_times = benchmark_scalability(rl_model)
    plot_timing_comparison(sizes, rl_times, exact_times)
    
    logging.info("\n--- SCALABILITY SUMMARY ---")
    for i, size in enumerate(sizes):
        if exact_times[i] != float('inf'):
            speedup = exact_times[i] / rl_times[i]
            logging.info(f"{size} nodes: RL is {speedup:.1f}x faster")
        else:
            logging.info(f"{size} nodes: RL is {rl_times[i]:.4f}s, Exact solver too slow")