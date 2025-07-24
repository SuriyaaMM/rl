# Reinforcment Learning 

# Supply Chain Optimization
- Si=mulates a Warehouse-Store based model using Simpy & uses Actor-Critic model with PPO to solve the problem.
- Logs can be found in `logs/`

# Navigation 
- Simulates a Simple Grid with Obstacles and solves it using Q-Learning 

# Travelling Salesman Problem
- Simulates the Travelling Salesman Problem
- A Actor-Critic model with PPO solves an reasonable solution which is paired with 2-opt search to increases its effectiveness 

## Testing with 50 Nodes
Exact solution for 50 nodes isn't feasible in my computer.
| Method         | Time (s) | Length  | vs OR-Tools | vs Exact|
|----------------|----------|---------|-------------|---------|
| RL Only        | 0.0053   | 1276.09 | +127.94%    |   nan   |
| RL + 2-opt     | 0.0453   | 576.27  |  +2.93%     |   nan   |
| OR-Tools       | 30.0010  | 559.84  | +0.00%      |   nan   |

**2-opt improvement time:** 0.0400 seconds  
**Speedup (RL+2opt vs OR-Tools):** 662.28×  

## Testing with 100 Nodes

| Method         | Time (s) | Length  | vs OR-Tools | vs Exact|
|----------------|----------|---------|-------------|---------|
| RL Only        | 0.0259   | 4660.41 | +340.16%    |   nan   |
| RL + 2-opt     | 35.1928  | 1118.37 |  +5.63%     |   nan   |
| OR-Tools       | 30.0004  | 1058.79 | +0.00%      |   nan   |

**2-opt improvement time:** 35.1669 seconds  
**Speedup (RL+2opt vs OR-Tools):** 0.85×  
