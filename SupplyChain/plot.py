import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_episode_summary(
    episode: int,
    buffer,
    initial_warehouse_levels: list,
    final_warehouse_levels: list,
    initial_store_levels: list,
    final_store_levels: list,
    save_dir="logs"
):
    """
    Generates a comprehensive summary plot for a single episode.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # --- Data Extraction ---
    operations = np.array(buffer.operations)
    quantities = np.array(buffer.quantities)
    rewards = np.array(buffer.rewards)
    diagnostics = buffer.diagnostics
    
    # --- FIX: Replace undefined 'q' with 'buffer.quantities[i]' ---
    produce_actions = [
        (buffer.target_ids[i], buffer.quantities[i]) 
        for i, op in enumerate(operations) if op == 0
    ]
    ship_actions = [
        (buffer.target_ids[i], buffer.quantities[i])
        for i, op in enumerate(operations) if op == 1
    ]

    # --- Metrics Calculation ---
    total_demand = sum(d['demand'] for d in diagnostics)
    total_sold = sum(d['sold'] for d in diagnostics)
    fulfillment_rate = (total_sold / total_demand * 100) if total_demand > 0 else 0
    
    wasted_prod = sum(d['wasted_production'] for d in diagnostics)
    wasted_ship = sum(d['wasted_shipping'] for d in diagnostics)
    
    # --- Plotting (4x2 grid) ---
    fig, axs = plt.subplots(4, 2, figsize=(20, 28))
    fig.suptitle(f"Episode {episode}: Comprehensive Summary", fontsize=24, y=0.97)

    # 1. Warehouse Inventory Change
    bar_width = 0.35
    w_indices = np.arange(len(initial_warehouse_levels))
    axs[0, 0].bar(w_indices - bar_width/2, initial_warehouse_levels, bar_width, label='Initial', color='darkorange')
    axs[0, 0].bar(w_indices + bar_width/2, final_warehouse_levels, bar_width, label='Final', color='dodgerblue')
    axs[0, 0].set_title("Warehouse Inventory Levels (Start vs. End)", fontsize=14)
    axs[0, 0].set_xlabel("Warehouse ID")
    axs[0, 0].set_ylabel("Stock Level")
    axs[0, 0].set_xticks(w_indices)
    axs[0, 0].legend()

    # 2. Store Inventory Change
    s_indices = np.arange(len(initial_store_levels))
    axs[0, 1].bar(s_indices - bar_width/2, initial_store_levels, bar_width, label='Initial', color='gold')
    axs[0, 1].bar(s_indices + bar_width/2, final_store_levels, bar_width, label='Final', color='teal')
    axs[0, 1].set_title("Store Inventory Levels (Start vs. End)", fontsize=14)
    axs[0, 1].set_xlabel("Store ID")
    axs[0, 1].legend()

    # 3. Demand Fulfillment Plot
    fulfillment_per_step = [(d['sold'] / d['demand'] * 100) if d['demand'] > 0 else 100 for d in diagnostics]
    axs[1, 0].plot(fulfillment_per_step, color='seagreen', label='Step Fulfillment %')
    axs[1, 0].axhline(y=100, color='red', linestyle='--', label='100% Target')
    axs[1, 0].set_title(f"Demand Fulfillment (Overall: {fulfillment_rate:.2f}%)", fontsize=14)
    axs[1, 0].set_xlabel("Step in Episode")
    axs[1, 0].set_ylabel("Fulfillment Rate (%)")
    axs[1, 0].set_ylim(0, 110)
    axs[1, 0].legend()

    # 4. Wasted Quantity Analysis Plot
    wasted_labels = [f'Overproduction\n({wasted_prod:.0f})', f'Failed Shipments\n({wasted_ship:.0f})']
    wasted_sizes = [wasted_prod, wasted_ship]
    if sum(wasted_sizes) > 0:
        axs[1, 1].pie(wasted_sizes, labels=wasted_labels, autopct='%1.1f%%', startangle=90, colors=['lightcoral', 'sandybrown'])
    axs[1, 1].set_title("Analysis of Wasted Quantity", fontsize=14)

    # 5. Operation Type Distribution
    op_counts = [len(produce_actions), len(ship_actions)]
    op_labels = [f'Produce ({op_counts[0]})', f'Ship ({op_counts[1]})']
    if sum(op_counts) > 0:
        axs[2, 0].pie(op_counts, labels=op_labels, autopct='%1.1f%%', startangle=90, colors=['skyblue', 'lightgreen'])
    axs[2, 0].set_title("Distribution of Operations", fontsize=14)
    
    # 6. Action Targets
    if produce_actions:
        sns.histplot(data=[p[0] for p in produce_actions], ax=axs[2, 1], discrete=True, color='skyblue', label='Production Targets')
    if ship_actions:
        sns.histplot(data=[s[0] for s in ship_actions], ax=axs[2, 1], discrete=True, color='lightgreen', label='Shipment Targets')
    axs[2, 1].set_title("Distribution of Action Targets", fontsize=14)
    if produce_actions or ship_actions: axs[2, 1].legend()

    # 7. Quantity Distribution
    sns.histplot(data=quantities, ax=axs[3, 0], kde=True, bins=30, color='purple')
    axs[3, 0].set_title("Distribution of Action Quantities", fontsize=14)
    axs[3, 0].axvline(quantities.mean(), color='red', linestyle='--', label=f'Mean: {quantities.mean():.2f}')
    axs[3, 0].legend()

    # 8. Reward Timeline
    running_avg = np.convolve(rewards, np.ones(50)/50, mode='valid')
    axs[3, 1].plot(rewards, label='Step Reward', color='gray', alpha=0.4)
    axs[3, 1].plot(running_avg, label='50-Step Moving Avg', color='crimson')
    axs[3, 1].set_title("Reward Timeline During Episode", fontsize=14)
    axs[3, 1].legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(save_dir, f"episode_{episode}_summary.png"))
    plt.close()

def plot_training_progress(rewards_history, avg_values_history, current_episode, save_dir="logs"):
    os.makedirs(save_dir, exist_ok=True)
    episodes = range(1, len(rewards_history) + 1)
    fig, axs = plt.subplots(2, 1, figsize=(12, 10))
    fig.suptitle(f"Overall Training Progress (Up to Episode {current_episode})", fontsize=16)

    axs[0].plot(episodes, rewards_history, label="Total Reward per Episode", color='green')
    if len(rewards_history) > 10:
        running_avg = np.convolve(rewards_history, np.ones(10)/10, mode='valid')
        axs[0].plot(range(10, len(rewards_history) + 1), running_avg, label='10-Episode Moving Avg', color='darkgreen')
    axs[0].set_title("Total Reward per Episode")
    axs[0].legend()

    axs[1].plot(episodes, avg_values_history, label="Average Predicted Value (Critic)", color='purple')
    axs[1].set_title("Critic's Average State Value Prediction per Episode")
    axs[1].legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(save_dir, f"training_progress_ep{current_episode}.png"))
    plt.close()