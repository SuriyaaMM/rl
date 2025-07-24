import tempfile
import os
import subprocess
import time 
import logging
from typing import List, Tuple, Optional

import torch
import numpy as np
import matplotlib.pyplot as plt
from python_tsp.exact import solve_tsp_dynamic_programming
from model import A2C, train_agent, test_agent
from road_network import generate_network

def write_tsp_file(distance_matrix: np.ndarray, filename: str):
    """Write distance matrix to TSPLIB format file"""
    n = len(distance_matrix)
    with open(filename, 'w') as f:
        f.write(f"NAME: benchmark\n")
        f.write(f"TYPE: TSP\n")
        f.write(f"COMMENT: Generated for benchmarking\n")
        f.write(f"DIMENSION: {n}\n")
        f.write(f"EDGE_WEIGHT_TYPE: EXPLICIT\n")
        f.write(f"EDGE_WEIGHT_FORMAT: FULL_MATRIX\n")
        f.write(f"EDGE_WEIGHT_SECTION\n")
        
        for i in range(n):
            for j in range(n):
                f.write(f"{int(distance_matrix[i][j] * 1000)} ")
            f.write("\n")
        f.write("EOF\n")

def solve_with_lkh(distance_matrix: np.ndarray, lkh_path: str = "LKH") -> Tuple[List[int], float, float]:
    """
    Solve TSP using LKH solver
    Returns: (tour, tour_length, solve_time)
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        tsp_file = os.path.join(temp_dir, "problem.tsp")
        tour_file = os.path.join(temp_dir, "tour.txt")
        param_file = os.path.join(temp_dir, "params.par")
        
        # Write TSP file
        write_tsp_file(distance_matrix, tsp_file)
        
        # Write parameter file
        with open(param_file, 'w') as f:
            f.write(f"PROBLEM_FILE = {tsp_file}\n")
            f.write(f"TOUR_FILE = {tour_file}\n")
            f.write(f"RUNS = 1\n")
            f.write(f"TIME_LIMIT = 300\n")  # 5 minutes max
        
        start_time = time.time()
        try:
            # Run LKH
            result = subprocess.run([lkh_path, param_file], 
                                  capture_output=True, text=True, timeout=300)
            solve_time = time.time() - start_time
            
            if result.returncode != 0:
                raise Exception(f"LKH failed: {result.stderr}")
            
            # Parse tour file
            tour = []
            with open(tour_file, 'r') as f:
                lines = f.readlines()
                tour_started = False
                for line in lines:
                    if line.strip() == "TOUR_SECTION":
                        tour_started = True
                        continue
                    if tour_started and line.strip() != "-1" and line.strip() != "EOF":
                        tour.append(int(line.strip()) - 1)  # Convert to 0-indexed
            
            # Calculate tour length
            tour_length = 0
            for i in range(len(tour)):
                tour_length += distance_matrix[tour[i]][tour[(i + 1) % len(tour)]]
            
            return tour, tour_length, solve_time
            
        except (subprocess.TimeoutExpired, Exception) as e:
            logging.warning(f"LKH solver failed: {e}")
            return [], float('inf'), float('inf')

def solve_with_or_tools(distance_matrix: np.ndarray) -> Tuple[List[int], float, float]:
    """
    Alternative: solve with OR-Tools if LKH is not available
    """
    try:
        from ortools.constraint_solver import routing_enums_pb2
        from ortools.constraint_solver import pywrapcp
        
        start_time = time.time()
        
        # Create distance callback
        def distance_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return int(distance_matrix[from_node][to_node] * 1000)
        
        # Create routing model
        manager = pywrapcp.RoutingIndexManager(len(distance_matrix), 1, 0)
        routing = pywrapcp.RoutingModel(manager)
        
        transit_callback_index = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
        
        # Set search parameters
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
        search_parameters.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
        search_parameters.time_limit.seconds = 30
        
        # Solve
        solution = routing.SolveWithParameters(search_parameters)
        solve_time = time.time() - start_time
        
        if solution:
            tour = []
            index = routing.Start(0)
            while not routing.IsEnd(index):
                tour.append(manager.IndexToNode(index))
                index = solution.Value(routing.NextVar(index))
            
            tour_length = solution.ObjectiveValue() / 1000.0
            return tour, tour_length, solve_time
        else:
            return [], float('inf'), float('inf')
            
    except ImportError:
        logging.warning("OR-Tools not available")
        return [], float('inf'), float('inf')

def calculate_tour_length(tour: List[int], distance_matrix: np.ndarray) -> float:
    """Calculate total tour length"""
    if not tour:
        return float('inf')
    
    length = 0
    for i in range(len(tour)):
        length += distance_matrix[tour[i]][tour[(i + 1) % len(tour)]]
    return length

def benchmark_scalability(
    model,
    sizes: List[int] = [10, 15, 20, 50, 100, 300, 500, 1000],
    use_lkh: bool = True,
    lkh_path: str = "LKH"
):    
    logging.info("\n----- Benchmarking scalability with solution quality comparison -----")
    
    results = {
        'sizes': [],
        'rl_times': [],
        'rl_lengths': [],
        'exact_times': [],
        'exact_lengths': [],
        'solver_times': [],
        'solver_lengths': [],
        'optimality_gaps': []
    }
    
    for size in sizes:
        print(f"\n--- Testing with {size} nodes ---")
        
        # Generate problem
        locations = np.random.rand(size, 2) * 100
        distance_matrix = np.linalg.norm(
            locations[:, np.newaxis, :] - locations[np.newaxis, :, :], axis=2)
        distance_matrix_torch = torch.from_numpy(distance_matrix).float()
        coordinates = torch.from_numpy(locations).float()
        
        # RL Agent inference
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
        
        rl_time = time.time() - start_time
        rl_length = calculate_tour_length(rl_tour, distance_matrix)
        
        # Exact solver (only for small instances)
        exact_time = float('inf')
        exact_length = float('inf')
        if size <= 15:  # Reduced threshold for DP
            try:
                start_time = time.time()
                # Use your existing DP solver here
                # exact_tour, exact_length = solve_tsp_dynamic_programming(distance_matrix)
                exact_time = time.time() - start_time
            except:
                pass
        
        # High-quality solver (LKH or OR-Tools)
        solver_time = float('inf')
        solver_length = float('inf')
        
        if use_lkh:
            _, solver_length, solver_time = solve_with_lkh(distance_matrix, lkh_path)
        
        # Fallback to OR-Tools if LKH fails
        if solver_length == float('inf'):
            _, solver_length, solver_time = solve_with_or_tools(distance_matrix)
        
        # Calculate optimality gap
        if solver_length != float('inf') and solver_length > 0:
            optimality_gap = ((rl_length - solver_length) / solver_length) * 100
        else:
            optimality_gap = float('inf')
        
        # Store results
        results['sizes'].append(size)
        results['rl_times'].append(rl_time)
        results['rl_lengths'].append(rl_length)
        results['exact_times'].append(exact_time)
        results['exact_lengths'].append(exact_length)
        results['solver_times'].append(solver_time)
        results['solver_lengths'].append(solver_length)
        results['optimality_gaps'].append(optimality_gap)
        
        # Log results
        speedup_vs_solver = solver_time / rl_time if rl_time > 0 else float('inf')
        
        logging.info(f"Size {size:4d} | "
                    f"RL: {rl_time:.4f}s ({rl_length:.2f}) | "
                    f"Solver: {solver_time:.4f}s ({solver_length:.2f}) | "
                    f"Speedup: {speedup_vs_solver:.1f}x | "
                    f"Gap: {optimality_gap:.2f}%")
    
    return results

def plot_scalability_results(results):
    """Plot the benchmarking results"""
    import matplotlib.pyplot as plt
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    sizes = results['sizes']
    
    # Runtime comparison
    ax1.loglog(sizes, results['rl_times'], 'o-', label='RL Agent', color='blue')
    valid_solver_times = [t for t in results['solver_times'] if t != float('inf')]
    valid_sizes_solver = [s for s, t in zip(sizes, results['solver_times']) if t != float('inf')]
    if valid_solver_times:
        ax1.loglog(valid_sizes_solver, valid_solver_times, 's-', label='LKH/OR-Tools', color='red')
    
    ax1.set_xlabel('Problem Size (nodes)')
    ax1.set_ylabel('Runtime (seconds)')
    ax1.set_title('Runtime Scalability')
    ax1.legend()
    ax1.grid(True)
    
    # Speedup
    speedups = [st/rt if rt > 0 and st != float('inf') else 0 
               for rt, st in zip(results['rl_times'], results['solver_times'])]
    ax2.semilogx(sizes, speedups, 'o-', color='green')
    ax2.set_xlabel('Problem Size (nodes)')
    ax2.set_ylabel('Speedup Factor')
    ax2.set_title('Speed Advantage of RL Agent')
    ax2.grid(True)
    
    # Solution quality comparison
    ax3.plot(sizes, results['rl_lengths'], 'o-', label='RL Agent', color='blue')
    valid_solver_lengths = [l for l in results['solver_lengths'] if l != float('inf')]
    valid_sizes_lengths = [s for s, l in zip(sizes, results['solver_lengths']) if l != float('inf')]
    if valid_solver_lengths:
        ax3.plot(valid_sizes_lengths, valid_solver_lengths, 's-', label='LKH/OR-Tools', color='red')
    
    ax3.set_xlabel('Problem Size (nodes)')
    ax3.set_ylabel('Tour Length')
    ax3.set_title('Solution Quality Comparison')
    ax3.legend()
    ax3.grid(True)
    
    # Optimality gap
    valid_gaps = [g for g in results['optimality_gaps'] if g != float('inf')]
    valid_sizes_gaps = [s for s, g in zip(sizes, results['optimality_gaps']) if g != float('inf')]
    if valid_gaps:
        ax4.plot(valid_sizes_gaps, valid_gaps, 'o-', color='orange')
        ax4.set_xlabel('Problem Size (nodes)')
        ax4.set_ylabel('Optimality Gap (%)')
        ax4.set_title('Solution Quality Gap')
        ax4.grid(True)
    
    plt.tight_layout()
    plt.show()