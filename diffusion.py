import numpy as np
import torch
from tqdm import tqdm


def build_adj_list(edge_index: torch.Tensor, num_nodes: int):
    """
    edge_index: [2, E]
    Returns adjacency list as Python list of lists.
    """
    adj = [[] for _ in range(num_nodes)]
    src = edge_index[0].cpu().numpy()
    dst = edge_index[1].cpu().numpy()
    for s, d in zip(src, dst):
        if d not in adj[s]:
            adj[s].append(d)
    return adj


def independent_cascade_spread(adj, seed: int, p: float, max_steps: int, rng: np.random.Generator):
    active = set([seed])
    newly_active = set([seed])

    steps = 0
    while newly_active and steps < max_steps:
        next_new = set()
        for u in newly_active:
            for v in adj[u]:
                if v in active:
                    continue
                if rng.random() < p:
                    next_new.add(v)
        active |= next_new
        newly_active = next_new
        steps += 1

    return len(active)


def generate_ic_labels(edge_index: torch.Tensor, num_nodes: int, p: float, mc_runs: int, max_steps: int, seed: int = 42):
    """
    For each node v, run IC mc_runs times and average the final spread size.
    Returns y: tensor [num_nodes, 1]
    """
    adj = build_adj_list(edge_index, num_nodes)
    rng = np.random.default_rng(seed)

    labels = np.zeros(num_nodes, dtype=np.float32)
    for v in tqdm(range(num_nodes), desc="Generating IC labels"):
        spreads = []
        for _ in range(mc_runs):
            spreads.append(independent_cascade_spread(adj, v, p, max_steps, rng))
        labels[v] = float(np.mean(spreads))

    y = torch.tensor(labels, dtype=torch.float32).view(-1, 1)
    return y