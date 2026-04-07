import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from torch_geometric.utils import to_networkx


def plot_graph_with_scores(data, scores: torch.Tensor, title="Graph", node_size=80):
    G = to_networkx(data, to_undirected=True)
    s = scores.view(-1).detach().cpu().numpy()
    s = (s - s.min()) / (s.max() - s.min() + 1e-9)

    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(8, 6))
    nx.draw_networkx_nodes(G, pos, node_color=s, cmap="viridis", node_size=node_size)
    nx.draw_networkx_edges(G, pos, alpha=0.3, width=0.8)
    plt.title(title)
    plt.axis("off")
    plt.show()