import torch
import numpy as np
import matplotlib.pyplot as plt
from heuristic_search import calculate_tour_length, two_opt_improvement, solve_with_or_tools, solve_exact
from model import A2C, train_agent, test_agent
from road_network import generate_network

import time
import logging
from typing import List, Tuple, Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Constants
NUM_HOUSES = 15
NUM_LOCATIONS = NUM_HOUSES + 1
WAREHOUSE_IDX = 0
COORDINATE_DIM = 2
HIDDEN_SIZE = 256
LEARNING_RATE = 3e-4
EPISODES = 5000
GAMMA = 0.99 



def get_rl_solution(model: A2C, coordinates: torch.Tensor, distance_matrix: np.ndarray) -> Tuple[List[int], float, float]:
    """Get solution from RL model"""
    size = len(coordinates)
    
    start_time = time.time()
    with torch.no_grad():
        current_loc = 0
        visited_mask = torch.zeros(size, dtype=torch.bool)
        visited_mask[0] = True
        rl_tour = [0]
        
        for _ in range(size-1):
            action_logits, _ = model(coordinates, current_loc, visited_mask)
            action = action_logits.argmax()
            current_loc = action.item()
            visited_mask[current_loc] = True
            rl_tour.append(current_loc)
        
        rl_tour.append(0)  # Return to start
    
    rl_time = time.time() - start_time
    rl_distance = calculate_tour_length(rl_tour, distance_matrix)
    
    return rl_tour, rl_distance, rl_time

def benchmark_comprehensive(
    model: A2C,
    sizes: List[int] = [10, 20, 50, 100, 200, 500],
    or_tools_time_limit: int = 30,
    max_2opt_iterations: int = 1000
):
    """Comprehensive benchmark: RL vs RL+2opt vs OR-Tools vs Exact"""
    
    logging.info("\n----- Comprehensive TSP Benchmark -----")
    
    results = {
        'sizes': [],
        'rl_times': [],
        'rl_lengths': [],
        'rl_2opt_times': [],
        'rl_2opt_lengths': [],
        'ortools_times': [],
        'ortools_lengths': [],
        'exact_times': [],
        'exact_lengths': [],
        'rl_vs_ortools_gap': [],
        'rl_2opt_vs_ortools_gap': [],
        'rl_vs_exact_gap': [],
        'rl_2opt_vs_exact_gap': [],
        'hybrid_speedup_vs_ortools': [],
        'hybrid_speedup_vs_exact': [],
        'quality_improvement': []
    }
    
    for size in sizes:
        print(f"\n{'='*50}")
        print(f"Testing with {size} nodes")
        print(f"{'='*50}")
        
        # Generate problem
        np.random.seed(42)  # For reproducibility
        locations = np.random.rand(size, 2) * 100
        distance_matrix = np.linalg.norm(
            locations[:, np.newaxis, :] - locations[np.newaxis, :, :], axis=2)
        coordinates = torch.from_numpy(locations).float()
        
        # 1. RL Agent solution
        rl_tour, rl_length, rl_time = get_rl_solution(model, coordinates, distance_matrix)
        
        # 2. RL + 2-opt solution
        start_time = time.time()
        rl_tour_copy = rl_tour[:-1]  # Remove duplicate start node for 2-opt
        improved_tour, improved_length, two_opt_time = two_opt_improvement(
            rl_tour_copy, distance_matrix, max_2opt_iterations)
        improved_tour.append(improved_tour[0])  # Complete the tour
        rl_2opt_time = rl_time + two_opt_time
        
        # 3. OR-Tools solution
        ortools_tour, ortools_length, ortools_time = solve_with_or_tools(
            distance_matrix, or_tools_time_limit)
        
        # 4. Exact solution (only for small instances)
        exact_tour, exact_length, exact_time = [], float('inf'), float('inf')
        if size <= 15:  # Reduced threshold for exact solver
            try:
                exact_tour, exact_length, exact_time = solve_exact(distance_matrix)
            except Exception as e:
                logging.warning(f"Exact solver failed: {e}")
        
        # Calculate metrics
        def safe_gap(actual, reference):
            if reference != float('inf') and reference > 0:
                return ((actual - reference) / reference) * 100
            return float('inf')
        
        rl_vs_ortools_gap = safe_gap(rl_length, ortools_length)
        rl_2opt_vs_ortools_gap = safe_gap(improved_length, ortools_length)
        rl_vs_exact_gap = safe_gap(rl_length, exact_length)
        rl_2opt_vs_exact_gap = safe_gap(improved_length, exact_length)
        
        def safe_speedup(ref_time, actual_time):
            if actual_time > 0 and ref_time != float('inf'):
                return ref_time / actual_time
            return float('inf')
        
        hybrid_speedup_vs_ortools = safe_speedup(ortools_time, rl_2opt_time)
        hybrid_speedup_vs_exact = safe_speedup(exact_time, rl_2opt_time)
        
        quality_improvement = safe_gap(rl_length, improved_length)
        quality_improvement = -quality_improvement if quality_improvement != float('inf') else 0
        
        # Store results
        results['sizes'].append(size)
        results['rl_times'].append(rl_time)
        results['rl_lengths'].append(rl_length)
        results['rl_2opt_times'].append(rl_2opt_time)
        results['rl_2opt_lengths'].append(improved_length)
        results['ortools_times'].append(ortools_time)
        results['ortools_lengths'].append(ortools_length)
        results['exact_times'].append(exact_time)
        results['exact_lengths'].append(exact_length)
        results['rl_vs_ortools_gap'].append(rl_vs_ortools_gap)
        results['rl_2opt_vs_ortools_gap'].append(rl_2opt_vs_ortools_gap)
        results['rl_vs_exact_gap'].append(rl_vs_exact_gap)
        results['rl_2opt_vs_exact_gap'].append(rl_2opt_vs_exact_gap)
        results['hybrid_speedup_vs_ortools'].append(hybrid_speedup_vs_ortools)
        results['hybrid_speedup_vs_exact'].append(hybrid_speedup_vs_exact)
        results['quality_improvement'].append(quality_improvement)
        
        # Log results
        print(f"Method           | Time (s) | Length  | vs OR-Tools | vs Exact")
        print(f"RL Only          | {rl_time:8.4f} | {rl_length:7.2f} | {rl_vs_ortools_gap:8.2f}% | {rl_vs_exact_gap:7.2f}%")
        print(f"RL + 2-opt       | {rl_2opt_time:8.4f} | {improved_length:7.2f} | {rl_2opt_vs_ortools_gap:8.2f}% | {rl_2opt_vs_exact_gap:7.2f}%")
        print(f"OR-Tools         | {ortools_time:8.4f} | {ortools_length:7.2f} | {0:8.2f}% | {safe_gap(ortools_length, exact_length):7.2f}%")
        if exact_time != float('inf'):
            print(f"Exact            | {exact_time:8.4f} | {exact_length:7.2f} | {-safe_gap(ortools_length, exact_length):8.2f}% | {0:7.2f}%")
        
        print(f"Speedup (RL+2opt vs OR-Tools): {hybrid_speedup_vs_ortools:.2f}x")
        if exact_time != float('inf'):
            print(f"Speedup (RL+2opt vs Exact): {hybrid_speedup_vs_exact:.2f}x")
        print(f"Quality improvement from 2-opt: {quality_improvement:.2f}%")
    
    return results

def analyze_comprehensive_results(results):
    """Analyze and print comprehensive summary statistics"""
    print("\n" + "="*80)
    print("COMPREHENSIVE BENCHMARK SUMMARY")
    print("="*80)
    
    # Filter out invalid results
    valid_ortools_indices = [i for i, gap in enumerate(results['rl_2opt_vs_ortools_gap']) 
                             if gap != float('inf')]
    valid_exact_indices = [i for i, gap in enumerate(results['rl_2opt_vs_exact_gap']) 
                           if gap != float('inf')]
    
    if valid_ortools_indices:
        ortools_gaps = [results['rl_2opt_vs_ortools_gap'][i] for i in valid_ortools_indices]
        ortools_speedups = [results['hybrid_speedup_vs_ortools'][i] for i in valid_ortools_indices]
        
        print("VS OR-TOOLS:")
        print(f"  Average optimality gap: {np.mean(ortools_gaps):.2f}%")
        print(f"  Median optimality gap: {np.median(ortools_gaps):.2f}%")
        print(f"  Best optimality gap: {np.min(ortools_gaps):.2f}%")
        print(f"  Average speedup: {np.mean(ortools_speedups):.2f}x")
        print(f"  Median speedup: {np.median(ortools_speedups):.2f}x")
        
        better_than_ortools = sum(1 for gap in ortools_gaps if gap < 0)
        print(f"  Cases where RL+2-opt beat OR-Tools: {better_than_ortools}/{len(ortools_gaps)}")
    
    if valid_exact_indices:
        exact_gaps = [results['rl_2opt_vs_exact_gap'][i] for i in valid_exact_indices]
        exact_speedups = [results['hybrid_speedup_vs_exact'][i] for i in valid_exact_indices]
        
        print("\nVS EXACT SOLVER:")
        print(f"  Average optimality gap: {np.mean(exact_gaps):.2f}%")
        print(f"  Median optimality gap: {np.median(exact_gaps):.2f}%")
        print(f"  Best optimality gap: {np.min(exact_gaps):.2f}%")
        print(f"  Average speedup: {np.mean(exact_speedups):.2f}x")
        print(f"  Median speedup: {np.median(exact_speedups):.2f}x")
    
    quality_improvements = [imp for imp in results['quality_improvement'] if imp != float('inf')]
    if quality_improvements:
        print(f"\n2-OPT IMPROVEMENT:")
        print(f"  Average quality improvement: {np.mean(quality_improvements):.2f}%")
        print(f"  Median quality improvement: {np.median(quality_improvements):.2f}%")
        print(f"  Best quality improvement: {np.max(quality_improvements):.2f}%")

def plot_comprehensive_results(results):
    """Plot comprehensive benchmark results"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 12))
    
    sizes = results['sizes']
    
    # Runtime comparison
    ax1.loglog(sizes, results['rl_times'], 'o-', label='RL Only', color='blue', alpha=0.7)
    ax1.loglog(sizes, results['rl_2opt_times'], 's-', label='RL + 2-opt', color='green', linewidth=2)
    
    # Filter valid times for plotting
    valid_ortools = [(s, t) for s, t in zip(sizes, results['ortools_times']) if t != float('inf')]
    valid_exact = [(s, t) for s, t in zip(sizes, results['exact_times']) if t != float('inf')]
    
    if valid_ortools:
        or_sizes, or_times = zip(*valid_ortools)
        ax1.loglog(or_sizes, or_times, '^-', label='OR-Tools', color='red')
    
    if valid_exact:
        ex_sizes, ex_times = zip(*valid_exact)
        ax1.loglog(ex_sizes, ex_times, 'd-', label='Exact', color='purple')
    
    ax1.set_xlabel('Problem Size (nodes)')
    ax1.set_ylabel('Runtime (seconds)')
    ax1.set_title('Runtime Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Solution quality
    ax2.plot(sizes, results['rl_lengths'], 'o-', label='RL Only', color='blue', alpha=0.7)
    ax2.plot(sizes, results['rl_2opt_lengths'], 's-', label='RL + 2-opt', color='green', linewidth=2)
    
    if valid_ortools:
        or_sizes, or_lengths = zip(*[(s, l) for s, l in zip(sizes, results['ortools_lengths']) if l != float('inf')])
        ax2.plot(or_sizes, or_lengths, '^-', label='OR-Tools', color='red')
    
    if valid_exact:
        ex_sizes, ex_lengths = zip(*[(s, l) for s, l in zip(sizes, results['exact_lengths']) if l != float('inf')])
        ax2.plot(ex_sizes, ex_lengths, 'd-', label='Exact', color='purple')
    
    ax2.set_xlabel('Problem Size (nodes)')
    ax2.set_ylabel('Tour Length')
    ax2.set_title('Solution Quality')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Speedup vs OR-Tools
    valid_speedups = [(s, sp) for s, sp in zip(sizes, results['hybrid_speedup_vs_ortools']) if sp != float('inf')]
    if valid_speedups:
        sp_sizes, speedups = zip(*valid_speedups)
        ax3.semilogx(sp_sizes, speedups, 'o-', color='green', linewidth=2)
        ax3.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='Break-even')
        ax3.set_xlabel('Problem Size (nodes)')
        ax3.set_ylabel('Speedup Factor')
        ax3.set_title('Speed Advantage (RL+2-opt vs OR-Tools)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # Optimality gap
    valid_gaps = [(s, g) for s, g in zip(sizes, results['rl_2opt_vs_ortools_gap']) if g != float('inf')]
    if valid_gaps:
        gap_sizes, gaps = zip(*valid_gaps)
        ax4.plot(gap_sizes, gaps, 's-', label='RL + 2-opt vs OR-Tools', color='green', linewidth=2)
        
        valid_exact_gaps = [(s, g) for s, g in zip(sizes, results['rl_2opt_vs_exact_gap']) if g != float('inf')]
        if valid_exact_gaps:
            ex_gap_sizes, ex_gaps = zip(*valid_exact_gaps)
            ax4.plot(ex_gap_sizes, ex_gaps, 'd-', label='RL + 2-opt vs Exact', color='purple', linewidth=2)
        
        ax4.axhline(y=0, color='red', linestyle='--', alpha=0.7, label='Optimal')
        ax4.set_xlabel('Problem Size (nodes)')
        ax4.set_ylabel('Optimality Gap (%)')
        ax4.set_title('Solution Quality Gap')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def visualize_solutions(locations, solutions_dict):
    """Visualize multiple solutions for comparison"""
    n_solutions = len(solutions_dict)
    cols = min(n_solutions, 3)
    rows = (n_solutions + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 5*rows))
    if n_solutions == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes.reshape(1, -1)
    
    for idx, (method, (tour, distance)) in enumerate(solutions_dict.items()):
        row, col = idx // cols, idx % cols
        ax = axes[row, col] if rows > 1 else axes[col]
        
        # Plot tour
        if tour and len(tour) > 1:
            for i in range(len(tour) - 1):
                start_node = tour[i]
                end_node = tour[i+1]
                ax.plot(*zip(locations[start_node], locations[end_node]), 
                       'b-', linewidth=2, alpha=0.7)
        
        # Plot nodes
        ax.scatter(locations[:, 0], locations[:, 1], c='lightblue', s=100, edgecolors='black')
        ax.scatter(locations[0, 0], locations[0, 1], c='red', s=200, marker='*')
        
        # Add node labels
        for i in range(len(locations)):
            ax.annotate(str(i), (locations[i, 0], locations[i, 1]), 
                       ha='center', va='center', fontweight='bold')
        
        ax.set_title(f'{method}\nDistance: {distance:.2f}')
        ax.set_xlabel("X Coordinate")
        ax.set_ylabel("Y Coordinate")
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(n_solutions, rows * cols):
        row, col = idx // cols, idx % cols
        axes[row, col].set_visible(False)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Generate network and train model
    locations, distance_matrix = generate_network()
    distance_matrix_tensor = torch.from_numpy(distance_matrix).float()
    coordinates = torch.from_numpy(locations).float()
    
    logging.info("=== Training Agent ===")
    rl_model = train_agent(distance_matrix_tensor, coordinates)
    
    # Test on original problem
    logging.info("\n=== Testing on Original Problem ===")
    rl_tour, rl_distance, rl_time = get_rl_solution(rl_model, coordinates, distance_matrix)
    
    # Apply 2-opt improvement
    rl_tour_for_2opt = rl_tour[:-1]  # Remove duplicate start
    improved_tour, improved_distance, two_opt_time = two_opt_improvement(rl_tour_for_2opt, distance_matrix)
    improved_tour.append(improved_tour[0])  # Complete the tour
    
    # Get OR-Tools solution
    ortools_tour, ortools_distance, ortools_time = solve_with_or_tools(distance_matrix, 30)
    
    # Get exact solution
    try:
        exact_tour, exact_distance, exact_time = solve_exact(distance_matrix)
    except:
        exact_tour, exact_distance, exact_time = [], float('inf'), float('inf')
    
    # Display results
    print("\n" + "="*60)
    print("ORIGINAL PROBLEM RESULTS")
    print("="*60)
    print(f"Method           | Time (s) | Distance | Gap vs Exact")
    print(f"RL Only          | {rl_time:8.4f} | {rl_distance:8.2f} | {((rl_distance-exact_distance)/exact_distance*100) if exact_distance != float('inf') else float('inf'):7.2f}%")
    print(f"RL + 2-opt       | {rl_time + two_opt_time:8.4f} | {improved_distance:8.2f} | {((improved_distance-exact_distance)/exact_distance*100) if exact_distance != float('inf') else float('inf'):7.2f}%")
    print(f"OR-Tools         | {ortools_time:8.4f} | {ortools_distance:8.2f} | {((ortools_distance-exact_distance)/exact_distance*100) if exact_distance != float('inf') else float('inf'):7.2f}%")
    if exact_distance != float('inf'):
        print(f"Exact            | {exact_time:8.4f} | {exact_distance:8.2f} | {0:7.2f}%")
    
    # Visualize solutions
    solutions = {
        'RL Only': (rl_tour, rl_distance),
        'RL + 2-opt': (improved_tour, improved_distance),
    }
    
    if ortools_distance != float('inf'):
        solutions['OR-Tools'] = (ortools_tour, ortools_distance)
    
    if exact_distance != float('inf'):
        solutions['Exact'] = (exact_tour, exact_distance)
    
    visualize_solutions(locations, solutions)
    
    # Run comprehensive benchmark
    print("\n" + "="*60)
    print("RUNNING COMPREHENSIVE BENCHMARK")
    print("="*60)
    
    results = benchmark_comprehensive(
        rl_model, 
        sizes=[10, 15, 20, 50, 100, 200],
        or_tools_time_limit=30,
        max_2opt_iterations=1000
    )
    
    analyze_comprehensive_results(results)
    plot_comprehensive_results(results)