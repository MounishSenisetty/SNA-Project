import numpy as np
import torch
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx
from torch_geometric.datasets import Planetoid


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_synthetic_ba_graph(n: int, m: int, feature_dim: int) -> Data:
    """
    Barabasi-Albert graph (scale-free), good for influence-like patterns.
    """
    G = nx.barabasi_albert_graph(n=n, m=m, seed=42)
    data = from_networkx(G)

    # Undirected edges are represented both directions in PyG
    # Ensure edge_index is present
    assert data.edge_index is not None

    # Random node features (you can replace with real features)
    x = torch.randn((data.num_nodes, feature_dim), dtype=torch.float32)
    data.x = x

    return data


def load_cora() -> Data:
    dataset = Planetoid(root="data/Planetoid", name="Cora")
    data = dataset[0]
    # Cora already has x and edge_index
    return data


def make_train_val_test_masks(num_nodes: int, train_ratio=0.6, val_ratio=0.2, seed=42):
    rng = np.random.default_rng(seed)
    idx = np.arange(num_nodes)
    rng.shuffle(idx)

    n_train = int(train_ratio * num_nodes)
    n_val = int(val_ratio * num_nodes)

    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    return train_mask, val_mask, test_mask