import torch
import numpy as np
import time
import logging
import sys
import datetime
import os 
import pandas as pd

from typing import Tuple, List

from meta_model import MetaA2C, run_meta_episode, meta_train
from road_network import generate_task_distribution, generate_network
from heuristic_search import two_opt_improvement, solve_with_or_tools, solve_exact, calculate_tour_length

# static parameters
COORDINATE_DIM = 2
FILE_SUFFIX_DYNAMIC = f"[{datetime.datetime.now()}]TSPMeta"
FILE_SUFFIX = f"TSPMeta"

# hyper parameters
SEED = 69
LR = 3e-4
OR_TIME_LIMIT = 15
META_EPISODES = 500
TASKS_PER_EPISODE = 25
MIN_NODES_TRAIN = 15
MAX_NODES_TRAIN = 20
META_LR = 3e-4
META_STEPS = 3
ADAPTATION_STEPS = 5

# setup logging
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()
logger.setLevel(level=logging.INFO)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)

file_handler = logging.FileHandler(f"{FILE_SUFFIX_DYNAMIC}Log.txt")
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def get_meta_rl_solution(
    meta_model: MetaA2C,
    coordinates: torch.Tensor,
    distance_matrix: np.ndarray,
    device: torch.device,
    adaptation_steps: int = ADAPTATION_STEPS,
    inner_lr: float = META_LR,
    gradient_clip_value: float = 1.0
) -> Tuple[List[int], float, float]:
    """
    Get a solution from the meta-learned model after adaptation.
    """
    start_time = time.time()
    
    adapted_model = MetaA2C(COORDINATE_DIM).to(device)
    adapted_model.load_state_dict(meta_model.state_dict())
    
    dist_matrix_tensor = torch.from_numpy(distance_matrix).float().to(device)

    # --- Adaptation Loop ---
    for _ in range(adaptation_steps):
        loss, _ = run_meta_episode(adapted_model, coordinates, dist_matrix_tensor,device)
        # Prevent backprop if loss is NaN
        if torch.isnan(loss):
            continue
            
        grads = torch.autograd.grad(loss, list(adapted_model.parameters()))

        # ADD GRADIENT CLIPPING HERE
        with torch.no_grad():
            for param, grad in zip(adapted_model.parameters(), grads):
                grad_clipped = torch.clamp(grad, -gradient_clip_value, gradient_clip_value)
                param -= inner_lr * grad_clipped
    
    # --- Inference with the adapted model ---
    with torch.no_grad():
        current_loc = 0
        visited_mask = torch.zeros(len(coordinates), dtype=torch.bool).to(device)
        visited_mask[0] = True
        rl_tour = [0]
        
        for _ in range(len(coordinates) - 1):
            action_logits, _ = adapted_model(coordinates, current_loc, visited_mask)
            # If logits are NaN after adaptation, we can't proceed.
            if torch.any(torch.isnan(action_logits)):
                logging.error("Failed to generate a valid tour due to NaN logits after adaptation.")
                return [], float('inf'), time.time() - start_time

            action = action_logits.argmax()
            current_loc = action.item()
            visited_mask[current_loc] = True
            rl_tour.append(current_loc)
        
        rl_tour.append(0)
    
    rl_time = time.time() - start_time
    rl_distance = calculate_tour_length(rl_tour, distance_matrix)
    
    return rl_tour, rl_distance, rl_time

if __name__ == "__main__":

    model_path = f"{FILE_SUFFIX}Model.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    meta_model = MetaA2C(coordinate_dim=COORDINATE_DIM).to(device)

    if os.path.exists(model_path):
        logging.info("="*50)
        logging.info(f"Loading existing model from {model_path}")
        logging.info("="*50)
        meta_model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        # If no model exists, run the training process
        logging.info("="*50)
        logging.info("Starting Meta-Training")
        logging.info("="*50)
        
        meta_optimizer = torch.optim.Adam(meta_model.parameters(), lr=LR)
        
        for episode in range(META_EPISODES):
            task_batch = generate_task_distribution(TASKS_PER_EPISODE, MIN_NODES_TRAIN, MAX_NODES_TRAIN)
            loss = meta_train(
                meta_model=meta_model,
                task_batch=task_batch,
                coordinate_dim=COORDINATE_DIM,
                meta_optimizer=meta_optimizer,
                meta_lr=META_LR,
                meta_steps=META_STEPS,
                device=device
            )
            
            if episode % 10 == 0:
                logging.info(f"Meta-Episode {episode} | Meta-Loss: {loss:.4f}")
                
        logging.info("="*50)
        logging.info("Completed Meta-Training")
        logging.info("="*50)

        torch.save(meta_model.state_dict(), model_path)
        logging.info(f"Model saved to {model_path}")
    
    test_sizes = [10, 100, 500]

    df_rows = []

    hyperparameters = {
        "lr" : META_LR,
        "Episodes" : META_EPISODES,
        "Steps" : META_STEPS,
        "AdaptationSteps" : ADAPTATION_STEPS,
        "TasksPerEpisode" : TASKS_PER_EPISODE,
        "MinNodexUsedForTraining" : MIN_NODES_TRAIN,
        "MaxNodesUsedForTraining" : MAX_NODES_TRAIN,
        "OrToolsTimeLimit" : OR_TIME_LIMIT
    }
    
    for size in test_sizes:
        logging.info("="*50)
        logging.info(f"Testing with {size} nodes")
        logging.info("="*50)
        
        np.random.seed(SEED) 
        locations, distance_matrix = generate_network(size)
        coordinates = torch.from_numpy(locations).float().to(device)
        
        # original metal-rl solution
        meta_rl_tour, meta_rl_dist, meta_rl_time = get_meta_rl_solution(
            meta_model, 
            coordinates, 
            distance_matrix,
            device
        )

        # metal-rl enhanced with 2-opt
        improved_tour, improved_dist, two_opt_time = two_opt_improvement(
            meta_rl_tour[:-1], 
            distance_matrix
        )
        improved_tour.append(improved_tour[0])
        
        # google's or tools
        ortools_tour, ortools_dist, ortools_time = solve_with_or_tools(distance_matrix, OR_TIME_LIMIT)
        
        # exact solution using dp
        exact_tour, exact_dist, exact_time = [], np.nan, np.nan
        if size <= 15:
            exact_tour, exact_dist, exact_time = solve_exact(distance_matrix)

        # gap with respect to dp & or tools
        meta_rl_gap_wrtdp = ((meta_rl_dist - exact_dist) / exact_dist * 100) if exact_dist != float('inf') else np.nan
        meta_rl_gap_wrtor =  ((meta_rl_dist - exact_dist) / ortools_dist * 100)
        meta_rl_2opt_gap_wrtdp = ((improved_dist - exact_dist) / exact_dist * 100) if exact_dist != float('inf') else np.nan
        meta_rl_2opt_gap_wrtor =  ((improved_dist - ortools_dist) / ortools_dist * 100)
        ortools_gap_wrtdp = ((ortools_dist - exact_dist) / exact_dist * 100) if exact_dist != float('inf') else np.nan

        meta_rl_data = {
            "Nodes" : size,
            "Method" : "Meta-RL",
            "Time" : meta_rl_time,
            "Distance" : meta_rl_dist,
            "OptimalityGap_WRTDp" :  meta_rl_gap_wrtdp,
            "OptimalityGap_WRTOr" : meta_rl_gap_wrtor,
        }
        meta_rl_2opt_data = {
            "Nodes" : size,
            "Method" : "Meta-RL-2opt",
            "Time" : meta_rl_time + two_opt_time,
            "Distance" : improved_dist,
            "OptimalityGap_WRTDp" :  meta_rl_gap_wrtdp,
            "OptimalityGap_WRTOr" : meta_rl_gap_wrtor,
        }
        or_tools_data = {
            "Nodes" : size,
            "Method" : "Or-Tools",
            "Time" : ortools_time,
            "Distance" : ortools_dist,
            "OptimalityGap_WRTDp" :  ortools_gap_wrtdp,
            "OptimalityGap_WRTOr" : np.nan,
        }
        exact_solution_data = {
            "Nodes" : size,
            "Method" : "Exact-DP",
            "Time" : exact_time,
            "Distance" : exact_dist,
            "OptimalityGap_WRTDp" :  np.nan,
            "OptimalityGap_WRTOr" : np.nan,
        }
        meta_rl_data.update(hyperparameters)
        meta_rl_2opt_data.update(hyperparameters)
        or_tools_data.update(hyperparameters)
        exact_solution_data.update(hyperparameters)
        
        df_rows += [meta_rl_data, meta_rl_2opt_data, or_tools_data, exact_solution_data]

        # for printing early
        df = pd.DataFrame(df_rows)
        logging.info(df)
        df.to_csv(f"{FILE_SUFFIX_DYNAMIC}Data.csv")
        logging.info(f"Saved Data as Markdown to {FILE_SUFFIX_DYNAMIC}Data.csv")

    