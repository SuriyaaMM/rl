import numpy as np
from typing import List, Tuple

def generate_network(num_locations: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates a single random TSP problem instance.
    """
    locations = np.random.rand(num_locations, 2) * 100
    distance_matrix = np.linalg.norm(locations[:, np.newaxis, :] - locations[np.newaxis, :, :], axis=2)
    return locations, distance_matrix

def generate_task_distribution(
    num_tasks: int,
    min_nodes: int,
    max_nodes: int
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Generates a distribution of TSP tasks with varying sizes.
    """
    task_batch = []
    for _ in range(num_tasks):
        num_nodes = np.random.randint(min_nodes, max_nodes + 1)
        locations, distance_matrix = generate_network(num_nodes)
        task_batch.append((locations, distance_matrix))
        
    return task_batch