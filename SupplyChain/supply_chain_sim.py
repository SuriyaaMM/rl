import simpy
import numpy as np

class SupplyChainSimulator:
    def __init__(self, env: simpy.Environment):
        self.env = env
        self.step_count = 0
        # warehouses & stores 
        self.warehouse_count = 10
        self.stores_count = 40
        # warehouse limits
        self.warehouse_low = 100
        self.warehouse_high = 5000
        # store limits
        self.store_low = 10
        self.store_high = 100
        # distance of store from warehouse
        self.warehouse_distance_min = 5
        self.warehouse_distance_max = 500

        # initialize warehouses
        self.warehouses: list[simpy.Container] = []
        for _ in range(self.warehouse_count):
            capacity = np.random.randint(self.warehouse_low, self.warehouse_high)
            # 40-60 %
            initial = np.random.randint(int(capacity * 0.4), int(capacity * 0.6))
            self.warehouses.append(simpy.Container(env=self.env, capacity=capacity, init=initial))
        # initialize stores
        self.stores: list[tuple[simpy.Container, int]] = []
        self.stores_warehouse_map = {}
        for i in range(self.stores_count):
            nearest_warehouse = np.random.randint(0, self.warehouse_count)
            nearest_distance = np.random.randint(self.warehouse_distance_min, self.warehouse_distance_max)
            capacity = np.random.randint(self.store_low, self.store_high)
            # 20-50% initially
            initial = np.random.randint(int(capacity * 0.2), int(capacity * 0.5))
            demand = np.random.randint(5, max(6, int(capacity * 0.3)))
            self.stores.append((simpy.Container(env=self.env, capacity=capacity, init=initial), demand))
            self.stores_warehouse_map[i] = (nearest_warehouse, nearest_distance)

    def get_observation(self) -> np.ndarray:
        """
        Gets the current state of the environment
        """
        warehouse_levels = [w.level / w.capacity for w in self.warehouses]
        store_observations = []
        for i in range(self.stores_count):
            store, demand = self.stores[i]
            supplying_warehouse_id, _ = self.stores_warehouse_map[i]
            supplying_warehouse = self.warehouses[supplying_warehouse_id]
            
            # normalize
            store_level_norm = store.level / store.capacity
            store_demand_norm = demand / self.store_high
            supplying_warehouse_level_norm = supplying_warehouse.level / supplying_warehouse.capacity
            
            store_observations.extend([store_level_norm, store_demand_norm, supplying_warehouse_level_norm])
        
        return np.array(warehouse_levels + store_observations)

    def step(self, action: dict) -> tuple[np.ndarray, float, dict]:
        # types
        warehouse: simpy.Container
        store: simpy.Container

        self.step_count += 1
        # get the operation
        operation = action["operation"]
        # get the quantity
        quantity = max(1.0, action["quantity"])
        # get the target_id
        target_id = action["target_id"]
        
        reward = 0.0
        diagnostics = {
            'sold': 0, 'demand': 0.0, 
            'wasted_production': 0, 'wasted_shipping': 0
        }
        
        # action: produce to warehouse
        if operation == 0:
            # penalty for wrong target_id
            if not (0 <= target_id < self.warehouse_count):
                reward -= 0.5
            else:
                warehouse = self.warehouses[target_id]
                space = warehouse.capacity - warehouse.level
                added = min(quantity, space)
                # penalty for producing 
                if added > 0:
                    warehouse.put(added)
                    reward -= 0.01 * added
                # track wasted production for plotting
                diagnostics['wasted_production'] = quantity - added
                # penalty for surplus produce
                if quantity > space:
                    reward -= 0.1
        # action: ship to store
        elif operation == 1:
            # penalty for wrong store
            if not (0 <= target_id < self.stores_count):
                reward -= 0.5
            else:
                store, _ = self.stores[target_id]
                warehouse_id, _ = self.stores_warehouse_map[target_id]
                warehouse = self.warehouses[warehouse_id]
                available = warehouse.level
                shipped = min(quantity, available)
                # penalty for keeping the store empty
                if shipped > 0:
                    warehouse.get(shipped)
                    store.put(shipped)
                    reward -= 0.01 * shipped
                # track wasted shipping for plotting
                diagnostics['wasted_shipping'] = quantity - shipped
                # penalty for not keeping the store supplied
                if quantity > available:
                    reward -= 0.1

        # sales action
        for i, (store, demand) in enumerate(self.stores):
            # track total demand for fulfillment rate
            diagnostics['demand'] += demand
            available = store.level
            sold = min(demand, available)
            # reward for selling
            if sold > 0:
                store.get(sold)
                reward += 0.85 * sold
                # track total sales for fulfillment rate
                diagnostics['sold'] += sold

            stockout = demand - sold
            # penalty for stockout
            if stockout > 0:
                reward -= 0.01 * stockout
        
            fill_ratio = store.level / store.capacity
            # penalty for poor inventory
            if not (0.2 < fill_ratio < 0.8):
                reward -= 0.01
        
        # get environment state and reward
        observation = self.get_observation()
        return observation, reward, diagnostics