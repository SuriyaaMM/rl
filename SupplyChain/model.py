import torch
import torch.nn as nn
from typing import Tuple
import numpy as np

class ActorAndCritic(nn.Module):

    def __init__(self, observation_dim: int, num_warehouses: int, num_stores: int):
        super().__init__()
        self.num_warehouses = num_warehouses
        self.num_stores = num_stores

        # common neural network
        self.common_nn = nn.Sequential(
            nn.Linear(observation_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        # actor neural network
        self.actor_operation_nn = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )
        # actor's warehouse id neural network
        self.actor_target_id_warehouse_nn = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_warehouses)
        )
        # actor's store id neural network
        self.actor_target_id_store_nn = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_stores)
        )
        # actor's quantity mean neural network
        self.actor_quantity_mean = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        # actor's quantity std neural network
        self.actor_quantity_std = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Softplus()
        )

        # critic network
        self.critic_nn = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        # xavier initialization
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def forward(
        self, 
        state: torch.Tensor
    ) -> Tuple[torch.Tensor, ...]:
        # types
        features: torch.Tensor
        operation_logits: torch.Tensor
        warehouse_id_logits: torch.Tensor
        store_id_logits: torch.Tensor
        quantity_mean: torch.Tensor
        quantity_std: torch.Tensor
        value: torch.Tensor


        features = self.common_nn(state)
        # get logits for all the operations
        operation_logits = self.actor_operation_nn(features)
        warehouse_id_logits = self.actor_target_id_warehouse_nn(features)
        store_id_logits = self.actor_target_id_store_nn(features)
        quantity_mean = self.actor_quantity_mean(features)
        quantity_std = self.actor_quantity_std(features)
        value = self.critic_nn(features)

        return operation_logits, warehouse_id_logits, store_id_logits, quantity_mean, quantity_std, value

    def sample_action(
        self, 
        state: np.ndarray, 
        max_quantity_scale: float = 50.0
    ) -> Tuple[dict, torch.Tensor, torch.Tensor]:
        # types
        state_tensor: torch.Tensor
        # convert to Tensor
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        # without updating gradients
        with torch.no_grad():
            operation_logits, warehouse_id_logits, store_id_logits, quantity_mean, quantity_std, value = self.forward(state_tensor)

        # initialize operations distribution
        operation_dist = torch.distributions.Categorical(logits=operation_logits)
        # sample an operation
        operation = operation_dist.sample()
        # logprob for evaluation
        logprob_operation = operation_dist.log_prob(operation)

        # warehouse is the target id
        if operation.item() == 0: 
            # initialize warehouse id distribution
            target_id_dist = torch.distributions.Categorical(logits=warehouse_id_logits)
            # sample an warehouse id
            target_id = target_id_dist.sample()
        # store is the target id
        else:
            # initialize store id distribution
            target_id_dist = torch.distributions.Categorical(logits=store_id_logits)
            # sample an store id
            target_id = target_id_dist.sample()
        # logprob for evaluation
        logprob_target_id = target_id_dist.log_prob(target_id)

        # quantity sampling (minimum = 1)
        scaled_quantity_mean = quantity_mean * max_quantity_scale + 1.0 
        # clamp std for stability
        quantity_std_clamped = torch.clamp(quantity_std, min=0.5, max=max_quantity_scale * 0.2)
        # initialize quantity distribytion
        quantity_dist = torch.distributions.Normal(loc=scaled_quantity_mean, scale=quantity_std_clamped)
        # sample an quantity
        quantity_sample = quantity_dist.sample()
        # clamp within the limit
        quantity = torch.clamp(quantity_sample, min=1.0, max=max_quantity_scale)
        # logprob for evaluation
        logprob_quantity = quantity_dist.log_prob(quantity_sample).squeeze(-1)
        # aggeregated log probability
        total_logprob = logprob_operation + logprob_target_id + logprob_quantity
        # construct the action dictionary
        action = {
            "operation": operation.item(),
            "quantity": quantity.item(),
            "target_id": target_id.item()
        }
        return action, total_logprob, value.squeeze(-1)

    def evaluate_action(
        self, 
        states: torch.Tensor, 
        operations: torch.Tensor, 
        target_ids: torch.Tensor, 
        quantities: torch.Tensor, 
        max_quantity_scale: float = 50.0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        operation_logits, warehouse_id_logits, store_id_logits, quantity_mean, quantity_std, values = self.forward(states)

        # initialize operations distribution
        operation_dist = torch.distributions.Categorical(logits=operation_logits)
        # evaluation metrics
        logprob_operation = operation_dist.log_prob(operations)
        entropy_operation = operation_dist.entropy()

        batch_size = states.shape[0]
        logprob_target_ids = torch.zeros(batch_size, dtype=torch.float32)
        entropy_target_ids = torch.zeros(batch_size, dtype=torch.float32)
        
        warehouse_mask = (operations == 0)
        store_mask = (operations == 1)
        # evaluate warehouse id part
        if warehouse_mask.any():
            # extract warehouse id parts from the target ids
            warehouse_indices = warehouse_mask.nonzero(as_tuple=True)[0]
            warehouse_target_ids = target_ids[warehouse_indices]
            warehouse_logits = warehouse_id_logits[warehouse_indices]
            # initialize warehouse id distribution
            warehouse_dist = torch.distributions.Categorical(logits=warehouse_logits)
            # evaluation metrics
            logprob_target_ids[warehouse_indices] = warehouse_dist.log_prob(warehouse_target_ids)
            entropy_target_ids[warehouse_indices] = warehouse_dist.entropy()
        # evaluate store id part
        if store_mask.any():
            # extract store id part from the target ids
            store_indices = store_mask.nonzero(as_tuple=True)[0]
            store_target_ids = target_ids[store_indices]
            store_logits = store_id_logits[store_indices]
            # initialzie store id distribution
            store_dist = torch.distributions.Categorical(logits=store_logits)
            # evaluation metrics
            logprob_target_ids[store_indices] = store_dist.log_prob(store_target_ids)
            entropy_target_ids[store_indices] = store_dist.entropy()

        
        scaled_quantity_mean = quantity_mean * max_quantity_scale + 1.0
        quantity_std_clamped = torch.clamp(quantity_std, min=0.5, max=max_quantity_scale * 0.2)
        # initialize quantity distribution
        quantity_dist = torch.distributions.Normal(loc=scaled_quantity_mean, scale=quantity_std_clamped)
        # evaluation metrics
        logprob_quantity = quantity_dist.log_prob(quantities.unsqueeze(-1)).squeeze(-1)
        entropy_quantity = quantity_dist.entropy().squeeze(-1)
        # aggeregated evaluation metrics
        logprob_total = logprob_operation + logprob_target_ids + logprob_quantity
        entropy_total = entropy_operation + entropy_target_ids + entropy_quantity

        return logprob_total, values.squeeze(-1), entropy_total