import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_results(data_path="Data.csv"):
    """
    Loads TSP results from a CSV file and generates visualizations.

    Args:
        data_path (str): The path to the input CSV file.
    """
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at '{data_path}'")
        return

    # Load and preprocess the data
    df = pd.read_csv(data_path, index_col=0)
    df.dropna(subset=['Time', 'Distance'], inplace=True)

    # --- Plot 1: Solution Quality vs. Computation Time ---
    print("Generating Plot 1: Solution Quality vs. Time...")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig1, ax1 = plt.subplots(figsize=(12, 8))

    # Filter out methods that are not the primary solvers for clarity
    plot_df = df[df['Method'].isin(['Meta-RL-2opt', 'Or-Tools', 'Exact-DP'])]

    sns.scatterplot(
        data=plot_df,
        x='Time',
        y='Distance',
        hue='Method',
        size='Nodes',
        sizes=(50, 500),
        style='Method',
        ax=ax1,
        palette='viridis',
        edgecolor='black',
        alpha=0.8
    )

    ax1.set_title('Solution Quality vs. Computation Time Trade-off', fontsize=16, weight='bold')
    ax1.set_xlabel('Time (seconds) [Log Scale]', fontsize=12)
    ax1.set_ylabel('Distance (Tour Length)', fontsize=12)
    ax1.set_xscale('linear') # Use a log scale for time to better see differences
    ax1.legend(title='Solver Method')
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)

    plt.tight_layout()
    plt.savefig('quality_vs_time.png', dpi=300)
    print("Saved 'quality_vs_time.png'")

    # --- Plot 2: Scalability Analysis (Distance vs. Problem Size) ---
    print("\nGenerating Plot 2: Scalability Analysis...")
    fig2, ax2 = plt.subplots(figsize=(12, 8))

    sns.lineplot(
        data=plot_df,
        x='Nodes',
        y='Distance',
        hue='Method',
        style='Method',
        markers=True,
        dashes=False,
        ax=ax2,
        palette='viridis',
        linewidth=2.5
    )

    ax2.set_title('Scalability: Tour Distance vs. Problem Size', fontsize=16, weight='bold')
    ax2.set_xlabel('Problem Size (Number of Nodes)', fontsize=12)
    ax2.set_ylabel('Distance (Tour Length)', fontsize=12)
    ax2.legend(title='Solver Method')
    ax2.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax2.set_xticks(plot_df['Nodes'].unique()) # Ensure ticks for each problem size tested

    plt.tight_layout()
    plt.savefig('scalability_analysis.png', dpi=300)
    plt.show()
    print("Saved 'scalability_analysis.png'")

if __name__ == "__main__":
    # Assuming the data is in "Data.csv" as requested
    plot_results(data_path="[2025-07-19 16:46:05.585506]TSPMetaData.csv")