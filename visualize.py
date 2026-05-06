"""
Visualization module for plots and figures.
Includes loss curves, prediction plots, and explanation visualizations.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import networkx as nx
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import torch
from torch_geometric.data import Data
import pandas as pd


def set_plot_style():
    """Set matplotlib style for publication-quality figures."""
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")


def save_figure(fig: plt.Figure, filepath: str, dpi: int = 300):
    """
    Save figure to file.
    
    Args:
        fig: Matplotlib figure
        filepath: Path to save
        dpi: Resolution
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def plot_training_curves(history: Dict, save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot training and validation loss curves.
    
    Args:
        history: Training history dictionary
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    set_plot_style()
    fig, ax = plt.subplots(figsize=(10, 6))
    
    epochs = range(len(history['train_loss']))
    
    ax.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax.plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss (MSE)', fontsize=12)
    ax.set_title('Training and Validation Loss Curves', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    if save_path:
        save_figure(fig, save_path)
    
    return fig


def plot_predictions_vs_actual(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot predictions vs actual centrality scores.
    
    Args:
        y_true: Ground truth scores
        y_pred: Predicted scores
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    set_plot_style()
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Scatter plot
    ax.scatter(y_true, y_pred, alpha=0.6, s=50, edgecolors='k', linewidth=0.5)
    
    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    ax.set_xlabel('Actual Centrality Score', fontsize=12)
    ax.set_ylabel('Predicted Score', fontsize=12)
    ax.set_title('Predictions vs Actual Centrality Scores', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    if save_path:
        save_figure(fig, save_path)
    
    return fig


def plot_top_k_comparison(
    top_k_analysis: Dict,
    k: int = 10,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot comparison of top-k nodes: predicted vs actual.
    
    Args:
        top_k_analysis: Analysis results from TopKAnalyzer
        k: Number of top nodes
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    set_plot_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # True top-k
    true_nodes = [str(idx) for idx, _ in top_k_analysis['true_top_k']]
    true_scores = [score for _, score in top_k_analysis['true_top_k']]
    
    ax1.barh(range(len(true_nodes)), true_scores, color='steelblue', edgecolor='black')
    ax1.set_yticks(range(len(true_nodes)))
    ax1.set_yticklabels(true_nodes)
    ax1.set_xlabel('Centrality Score', fontsize=11)
    ax1.set_title(f'Top {k} Actual Influential Nodes', fontsize=12, fontweight='bold')
    ax1.invert_yaxis()
    ax1.grid(axis='x', alpha=0.3)
    
    # Predicted top-k
    pred_nodes = [str(idx) for idx, _ in top_k_analysis['pred_top_k']]
    pred_scores = [score for _, score in top_k_analysis['pred_top_k']]
    colors = ['green' if int(pred_nodes[i]) in top_k_analysis['overlap_indices']
              else 'lightcoral' for i in range(len(pred_nodes))]
    
    ax2.barh(range(len(pred_nodes)), pred_scores, color=colors, edgecolor='black')
    ax2.set_yticks(range(len(pred_nodes)))
    ax2.set_yticklabels(pred_nodes)
    ax2.set_xlabel('Predicted Score', fontsize=11)
    ax2.set_title(f'Top {k} Predicted Influential Nodes', fontsize=12, fontweight='bold')
    ax2.invert_yaxis()
    ax2.grid(axis='x', alpha=0.3)
    
    # Add legend
    correct = mpatches.Patch(color='green', label='Correct Prediction')
    incorrect = mpatches.Patch(color='lightcoral', label='Incorrect Prediction')
    fig.legend(handles=[correct, incorrect], loc='upper center', bbox_to_anchor=(0.5, -0.02), ncol=2)
    
    plt.tight_layout()
    
    if save_path:
        save_figure(fig, save_path)
    
    return fig


def plot_explanation_subgraph(
    data: Data,
    target_node: int,
    important_nodes: List[int],
    explanation: Dict,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Visualize explanation subgraph with important nodes highlighted.
    
    Args:
        data: PyTorch Geometric data
        target_node: Node to explain
        important_nodes: Important neighboring nodes
        explanation: Explanation dictionary
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    set_plot_style()
    
    # Create NetworkX graph
    edge_index = data.edge_index.numpy()
    G = nx.Graph()
    G.add_nodes_from(range(data.num_nodes))
    G.add_edges_from(list(zip(edge_index[0], edge_index[1])))
    
    # Create subgraph with target and important nodes
    subgraph_nodes = [target_node] + important_nodes
    H = G.subgraph(subgraph_nodes)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Layout
    pos = nx.spring_layout(H, k=0.5, iterations=50, seed=42)
    
    # Draw edges
    nx.draw_networkx_edges(H, pos, ax=ax, edge_color='gray', width=1.5, alpha=0.6)
    
    # Draw nodes
    node_colors = ['#FF6B6B' if node == target_node else '#4ECDC4' for node in H.nodes()]
    nx.draw_networkx_nodes(H, pos, node_color=node_colors, node_size=800,
                          ax=ax, edgecolors='black', linewidths=2)
    
    # Draw labels
    labels = {node: str(node) for node in H.nodes()}
    nx.draw_networkx_labels(H, pos, labels, ax=ax, font_size=10, font_weight='bold')
    
    ax.set_title(f'Explanation Subgraph for Node {target_node}\nPredicted Score: {explanation["prediction"]:.4f}',
                fontsize=13, fontweight='bold')
    ax.axis('off')
    
    # Legend
    target_patch = mpatches.Patch(color='#FF6B6B', label='Target Node')
    neighbor_patch = mpatches.Patch(color='#4ECDC4', label='Important Neighbors')
    ax.legend(handles=[target_patch, neighbor_patch], loc='upper left', fontsize=10)
    
    if save_path:
        save_figure(fig, save_path)
    
    return fig


def plot_node_importance_map(
    importance_scores: np.ndarray,
    top_k: int = 10,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot node importance scores across the graph.
    
    Args:
        importance_scores: Importance score for each node
        top_k: Number of top nodes to highlight
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    set_plot_style()
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Sort nodes by importance
    sorted_indices = np.argsort(-importance_scores)
    
    # Create bar plot
    colors = ['#FF6B6B' if i in sorted_indices[:top_k] else '#95E1D3'
              for i in range(len(importance_scores))]
    
    ax.bar(range(len(importance_scores)), importance_scores[sorted_indices],
          color=colors, edgecolor='black', linewidth=0.5)
    
    ax.set_xlabel('Node (sorted by importance)', fontsize=12)
    ax.set_ylabel('Importance Score', fontsize=12)
    ax.set_title(f'Node Importance Map (Top {top_k} highlighted)', fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Legend
    top_k_patch = mpatches.Patch(color='#FF6B6B', label=f'Top {top_k} Nodes')
    other_patch = mpatches.Patch(color='#95E1D3', label='Other Nodes')
    ax.legend(handles=[top_k_patch, other_patch], fontsize=10)
    
    if save_path:
        save_figure(fig, save_path)
    
    return fig


def plot_metrics_table(
    metrics: Dict,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create a table visualization of metrics.
    
    Args:
        metrics: Dictionary of metrics
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    set_plot_style()
    
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('off')
    
    # Create table data
    rows = []
    for key, value in metrics.items():
        if isinstance(value, float):
            rows.append([key, f'{value:.4f}'])
        else:
            rows.append([key, str(value)])
    
    # Create table
    table = ax.table(cellText=rows,
                    colLabels=['Metric', 'Value'],
                    cellLoc='center',
                    loc='center',
                    colWidths=[0.4, 0.4])
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2)
    
    # Style header
    for i in range(2):
        table[(0, i)].set_facecolor('#4ECDC4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(rows) + 1):
        for j in range(2):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#F0F0F0')
            else:
                table[(i, j)].set_facecolor('#FFFFFF')
    
    ax.set_title('Evaluation Metrics', fontsize=13, fontweight='bold', pad=20)
    
    if save_path:
        save_figure(fig, save_path)
    
    return fig


def create_result_summary_plots(
    history: Dict,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metrics: Dict,
    top_k_analysis: Dict,
    output_dir: str = './outputs'
) -> Dict[str, str]:
    """
    Create all result summary plots and save them.
    
    Args:
        history: Training history
        y_true: Ground truth scores
        y_pred: Predictions
        metrics: Evaluation metrics
        top_k_analysis: Top-k analysis results
        output_dir: Directory to save plots
        
    Returns:
        Dictionary mapping plot names to file paths
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    saved_files = {}
    
    # Plot loss curves
    plot_training_curves(
        history,
        save_path=f'{output_dir}/training_curves.png'
    )
    saved_files['training_curves'] = f'{output_dir}/training_curves.png'
    
    # Plot predictions vs actual
    plot_predictions_vs_actual(
        y_true,
        y_pred,
        save_path=f'{output_dir}/predictions_vs_actual.png'
    )
    saved_files['predictions_vs_actual'] = f'{output_dir}/predictions_vs_actual.png'
    
    # Plot top-k comparison
    plot_top_k_comparison(
        top_k_analysis,
        k=len(top_k_analysis['true_top_k']),
        save_path=f'{output_dir}/top_k_comparison.png'
    )
    saved_files['top_k_comparison'] = f'{output_dir}/top_k_comparison.png'
    
    # Plot metrics table
    plot_metrics_table(
        metrics,
        save_path=f'{output_dir}/metrics_table.png'
    )
    saved_files['metrics_table'] = f'{output_dir}/metrics_table.png'
    
    return saved_files
