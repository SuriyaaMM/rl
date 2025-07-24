import numpy as np
import time
import logging

from typing import List, Tuple

from python_tsp.exact import solve_tsp_dynamic_programming

def two_opt_improvement(
    tour: List[int], 
    distance_matrix: np.ndarray, 
    max_iterations: int = 100
) -> Tuple[List[int], float, float]:
    """
    Apply 2-opt improvement to a tour
    Returns: (improved_tour, tour_length, improvement_time)
    """
    def calculate_tour_length(tour_path: List[int]) -> float:
        if len(tour_path) < 2:
            return 0.0
        length = 0
        for i in range(len(tour_path)-1):
            length += distance_matrix[tour_path[i], tour_path[i+1]]
        return length
    
    def two_opt_swap(tour_path: List[int], i: int, k: int) -> List[int]:
        """Perform 2-opt swap between positions i and k"""
        new_tour = tour_path[:i] + tour_path[i:k+1][::-1] + tour_path[k+1:]
        return new_tour
    
    start_time = time.time()
    
    current_tour = tour.copy()
    current_length = calculate_tour_length(current_tour)
    n = len(current_tour)
    
    improved = True
    iteration = 0
    
    while improved and iteration < max_iterations:
        improved = False
        iteration += 1
        
        for i in range(n-1):
            for k in range(i + 2, n):
                if k == n - 1 and i == 0:
                    continue
                
                new_tour = two_opt_swap(current_tour, i, k)
                new_length = calculate_tour_length(new_tour)
                
                if new_length < current_length:
                    current_tour = new_tour
                    current_length = new_length
                    improved = True
                    break
            
            if improved:
                break
    
    improvement_time = time.time() - start_time
    logging.info(f"2-opt improvement time: {improvement_time:.4f} seconds")
    
    return current_tour, current_length, improvement_time

def solve_exact(
    distance_matrix: np.ndarray
):
    """Solve TSP using exact dynamic programming"""
    logging.info("Solving for the exact global minimum...")
    
    start_time = time.time()
    permutation, distance = solve_tsp_dynamic_programming(distance_matrix)
    full_tour = permutation + [permutation[0]]
    
    exact_solve_time = time.time() - start_time
    logging.info(f"Exact solver time: {exact_solve_time:.4f} seconds")
    
    return full_tour, distance, exact_solve_time


def solve_with_or_tools(
    distance_matrix: np.ndarray, 
    time_limit: int = 30
) -> Tuple[List[int], float, float]:
    """
    Solve TSP with OR-Tools
    Returns: (tour, tour_length, solve_time)
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
        search_parameters.time_limit.seconds = time_limit
        
        # Solve
        solution = routing.SolveWithParameters(search_parameters)
        solve_time = time.time() - start_time
        
        if solution:
            tour = []
            index = routing.Start(0)
            while not routing.IsEnd(index):
                tour.append(manager.IndexToNode(index))
                index = solution.Value(routing.NextVar(index))
            tour.append(tour[0])  # Complete the tour
            
            tour_length = solution.ObjectiveValue() / 1000.0
            return tour, tour_length, solve_time
        else:
            return [], float('inf'), float('inf')
            
    except ImportError:
        logging.error("OR-Tools not available. Please install with: pip install ortools")
        return [], np.nan, np.nan

def calculate_tour_length(
    tour: List[int], 
    distance_matrix: np.ndarray
) -> float:
    """Calculate total tour length"""
    if not tour or len(tour) < 2:
        return float('inf')
    
    length = 0
    for i in range(len(tour)-1):
        length += distance_matrix[tour[i], tour[i+1]]
    return length