import networkx as nx
import numpy as np
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from torch_geometric.utils import to_networkx


def classical_centrality_scores(data):
    G = to_networkx(data, to_undirected=True)
    degree = np.array([v for _, v in G.degree()], dtype=np.float32)
    pagerank = np.array(list(nx.pagerank(G).values()), dtype=np.float32)
    betweenness = np.array(list(nx.betweenness_centrality(G).values()), dtype=np.float32)
    closeness = np.array(list(nx.closeness_centrality(G).values()), dtype=np.float32)
    try:
        eigen = np.array(list(nx.eigenvector_centrality_numpy(G).values()), dtype=np.float32)
    except Exception:
        eigen = degree.copy()
    return {
        "degree": degree,
        "pagerank": pagerank,
        "betweenness": betweenness,
        "closeness": closeness,
        "eigenvector": eigen,
    }


def heuristic_features(data, k_hop=2):
    G = to_networkx(data, to_undirected=True)
    n = data.num_nodes
    deg = np.zeros(n, dtype=np.float32)
    hop = np.zeros(n, dtype=np.float32)
    spread1 = np.zeros(n, dtype=np.float32)
    p = 0.1

    for v in range(n):
        deg[v] = float(G.degree(v))
        nodes_k = nx.single_source_shortest_path_length(G, v, cutoff=k_hop)
        hop[v] = float(len(nodes_k) - 1)
        spread1[v] = p * deg[v]

    return np.stack([deg, hop, spread1], axis=1)


def ml_baselines(X, y, train_mask, test_mask):
    out = {}
    train_idx = np.where(train_mask.cpu().numpy())[0]
    test_idx = np.where(test_mask.cpu().numpy())[0]

    models = {
        "linear": LinearRegression(),
        "ridge": Ridge(alpha=1.0),
        "lasso": Lasso(alpha=0.001),
        "rf": RandomForestRegressor(n_estimators=200, random_state=0),
    }

    for name, model in models.items():
        model.fit(X[train_idx], y[train_idx])
        pred = model.predict(X[test_idx])
        out[name] = pred.astype(np.float32)
    return out


def non_learning_baselines(data, test_mask):
    idx = np.where(test_mask.cpu().numpy())[0]
    scores = classical_centrality_scores(data)
    h = heuristic_features(data)
    scores["khop_size"] = h[:, 1]
    scores["expected_one_step_spread"] = h[:, 2]
    return {k: v[idx] for k, v in scores.items()}


def build_tabular_features(data):
    x = data.x.detach().cpu().numpy()
    h = heuristic_features(data)
    return np.concatenate([x, h], axis=1)
