import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from torch_geometric.utils import to_networkx
from pathlib import Path


def plot_graph_with_scores(data, scores: torch.Tensor, title="Graph", node_size=80, save_path=None):
    G = to_networkx(data, to_undirected=True)
    s = scores.view(-1).detach().cpu().numpy()
    s = (s - s.min()) / (s.max() - s.min() + 1e-9)

    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(8, 6))
    nx.draw_networkx_nodes(G, pos, node_color=s, cmap="viridis", node_size=node_size)
    nx.draw_networkx_edges(G, pos, alpha=0.3, width=0.8)
    plt.title(title)
    plt.axis("off")
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=220, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_feature_importance(feature_scores, title="Feature Attribution", save_path=None):
    feature_scores = np.asarray(feature_scores)
    plt.figure(figsize=(8, 4))
    plt.bar(np.arange(len(feature_scores)), feature_scores)
    plt.xlabel("Feature Index")
    plt.ylabel("Attribution")
    plt.title(title)
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=220, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_fidelity_curves(k_values, insert_curve, delete_curve, save_path=None):
    plt.figure(figsize=(7, 4))
    plt.plot(k_values, insert_curve, marker="o", label="Insertion")
    plt.plot(k_values, delete_curve, marker="s", label="Deletion")
    plt.xlabel("Top-k edges")
    plt.ylabel("Normalized prediction")
    plt.title("Fidelity Curves")
    plt.legend()
    plt.grid(alpha=0.2)
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=220, bbox_inches="tight")
        plt.close()
    else:
        plt.show()