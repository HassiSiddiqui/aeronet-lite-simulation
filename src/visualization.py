import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import base64
from io import BytesIO
from typing import List, Tuple, Dict, Any
from grid_model import GRID_SIZE, GridCell, ZONE_COLORS

def fig_to_base64(fig) -> str:
    """Converts a matplotlib figure to a base64 string."""
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return encoded

def ensure_figures_dir():
    os.makedirs("report/figures", exist_ok=True)

def plot_zone_map(grid: List[List[GridCell]]):
    ensure_figures_dir()
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Create a simple color grid
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            color = ZONE_COLORS.get(grid[r][c].zone, "#FFFFFF")
            ax.add_patch(plt.Rectangle((c, GRID_SIZE - 1 - r), 1, 1, facecolor=color, edgecolor="black"))
            
            # Add markers
            if grid[r][c].is_hub:
                ax.text(c + 0.5, GRID_SIZE - 1 - r + 0.5, 'H', ha='center', va='center', color='white', fontweight='bold', fontsize=12)
            elif grid[r][c].is_charging:
                ax.text(c + 0.5, GRID_SIZE - 1 - r + 0.5, 'C', ha='center', va='center', color='white', fontweight='bold', fontsize=12)
            elif grid[r][c].zone == "Hospital":
                ax.text(c + 0.5, GRID_SIZE - 1 - r + 0.5, '+', ha='center', va='center', color='white', fontweight='bold', fontsize=16)
            elif grid[r][c].is_medical_pickup:
                ax.text(c + 0.5, GRID_SIZE - 1 - r + 0.5, 'M', ha='center', va='center', color='black', fontweight='bold', fontsize=12)
            elif grid[r][c].no_fly:
                ax.text(c + 0.5, GRID_SIZE - 1 - r + 0.5, 'X', ha='center', va='center', color='red', fontweight='bold', fontsize=14)

    ax.set_xlim(0, GRID_SIZE)
    ax.set_ylim(0, GRID_SIZE)
    ax.set_xticks(range(GRID_SIZE))
    ax.set_yticks(range(GRID_SIZE))
    ax.set_xticklabels(range(GRID_SIZE))
    ax.set_yticklabels(reversed(range(GRID_SIZE)))
    ax.grid(False)
    ax.set_title("City Zone Map")
    
    fig.savefig("report/figures/zone_map.png", bbox_inches="tight")
    return fig

def plot_route_map(grid: List[List[GridCell]], routes: List[List[Tuple[int, int]]], drone_positions: List[Tuple[int, int]]):
    ensure_figures_dir()
    fig, ax = plt.subplots(figsize=(6, 6))
    
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            color = "#DDDDDD" if not grid[r][c].no_fly else "#FF9999"
            ax.add_patch(plt.Rectangle((c, GRID_SIZE - 1 - r), 1, 1, facecolor=color, edgecolor="white"))
            
    # Draw routes
    colors = ['blue', 'green', 'purple', 'orange', 'cyan']
    for i, route in enumerate(routes):
        if not route: continue
        c_color = colors[i % len(colors)]
        x_coords = [p[1] + 0.5 for p in route]
        y_coords = [GRID_SIZE - 1 - p[0] + 0.5 for p in route]
        ax.plot(x_coords, y_coords, color=c_color, marker='.', linestyle='-', linewidth=2, alpha=0.7)
        
    # Draw drone positions
    for i, pos in enumerate(drone_positions):
        ax.plot(pos[1] + 0.5, GRID_SIZE - 1 - pos[0] + 0.5, marker='o', markersize=10, color='red', markeredgecolor='black')
        ax.text(pos[1] + 0.5, GRID_SIZE - 1 - pos[0] + 0.5, f"D{i+1}", ha='center', va='center', color='white', fontsize=8)
        
    ax.set_xlim(0, GRID_SIZE)
    ax.set_ylim(0, GRID_SIZE)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Live Drone Routes")
    
    fig.savefig("report/figures/route_map.png", bbox_inches="tight")
    return fig

def plot_demand_heatmap(grid: List[List[GridCell]]):
    ensure_figures_dir()
    fig, ax = plt.subplots(figsize=(6, 6))
    
    demand_matrix = np.zeros((GRID_SIZE, GRID_SIZE))
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            demand_matrix[r][c] = grid[r][c].demand
            
    sns.heatmap(demand_matrix, cmap="YlOrRd", ax=ax, annot=True, cbar=True, fmt=".1f")
    ax.set_title("Demand Heatmap")
    
    fig.savefig("report/figures/demand_heatmap.png", bbox_inches="tight")
    return fig

def plot_fitness_history(history: List[float]):
    ensure_figures_dir()
    fig, ax = plt.subplots(figsize=(8, 4))
    
    ax.plot(range(len(history)), history, color="cyan", linewidth=2)
    ax.set_facecolor("#1E1E1E")
    fig.patch.set_facecolor("#1E1E1E")
    ax.set_title("GA Fitness Over Generations", color="white")
    ax.set_xlabel("Generation", color="white")
    ax.set_ylabel("Fitness", color="white")
    ax.tick_params(colors="white")
    
    fig.savefig("report/figures/fitness_history.png", bbox_inches="tight")
    return fig

def plot_forecast_results(y_true, y_pred, model_name: str):
    ensure_figures_dir()
    fig, ax = plt.subplots(figsize=(8, 4))
    
    indices = np.arange(min(100, len(y_true))) # plot subset for clarity
    ax.plot(indices, y_true[:100], label="Actual", color="blue", alpha=0.7)
    ax.plot(indices, y_pred[:100], label="Predicted", color="red", linestyle="--", alpha=0.7)
    ax.set_title(f"Demand Forecast ({model_name})")
    ax.set_xlabel("Time (Hours)")
    ax.set_ylabel("Deliveries")
    ax.legend()
    
    fig.savefig(f"report/figures/demand_forecast_{model_name}.png", bbox_inches="tight")
    # For README requirement we need specifically `demand_forecast.png`
    fig.savefig("report/figures/demand_forecast.png", bbox_inches="tight")
    return fig

def plot_confusion_matrix(cm, labels: List[str]):
    ensure_figures_dir()
    fig, ax = plt.subplots(figsize=(6, 5))
    
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_title("Anomaly Detection Confusion Matrix")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    
    fig.savefig("report/figures/anomaly_confusion.png", bbox_inches="tight")
    return fig
