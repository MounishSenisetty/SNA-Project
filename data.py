import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.datasets import KarateClub, Planetoid
from torch_geometric.utils import from_networkx, to_undirected


@dataclass
class DatasetBundle:
    data: Data
    name: str
    metadata: Dict
    gt_explanations: Optional[Dict] = None


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _attach_features(data: Data, feature_dim: int, seed: int):
    if getattr(data, "x", None) is None:
        gen = torch.Generator().manual_seed(seed)
        data.x = torch.randn((data.num_nodes, feature_dim), generator=gen, dtype=torch.float32)
    return data


def _finalize_graph(
    G: nx.Graph,
    feature_dim: int,
    seed: int,
    directed: bool,
    weighted: bool,
) -> Data:
    if weighted:
        rng = np.random.default_rng(seed)
        for u, v in G.edges():
            G[u][v]["weight"] = float(rng.uniform(0.2, 1.0))
    data = from_networkx(G)
    if not directed:
        data.edge_index = to_undirected(data.edge_index, num_nodes=data.num_nodes)
    data = _attach_features(data, feature_dim=feature_dim, seed=seed)
    return data


def generate_synthetic_graph(cfg: Dict, seed: int) -> DatasetBundle:
    gcfg = cfg["graph"]
    gtype = gcfg["type"]
    n = int(gcfg["n"])
    directed = bool(gcfg.get("directed", False))
    weighted = bool(gcfg.get("weighted", False))
    feature_dim = int(cfg["feature_dim"])

    if gtype == "ba":
        G = nx.barabasi_albert_graph(n=n, m=int(gcfg.get("m", 3)), seed=seed)
    elif gtype == "er":
        G = nx.erdos_renyi_graph(n=n, p=float(gcfg.get("p", 0.03)), seed=seed, directed=directed)
    elif gtype == "ws":
        G = nx.watts_strogatz_graph(
            n=n,
            k=int(gcfg.get("k", 6)),
            p=float(gcfg.get("rewire_p", 0.2)),
            seed=seed,
        )
    elif gtype == "sbm":
        sizes = gcfg.get("sizes", [n // 3, n // 3, n - 2 * (n // 3)])
        p_in = float(gcfg.get("p_in", 0.08))
        p_out = float(gcfg.get("p_out", 0.01))
        probs = [[p_in if i == j else p_out for j in range(len(sizes))] for i in range(len(sizes))]
        G = nx.stochastic_block_model(sizes=sizes, p=probs, seed=seed)
    elif gtype == "motif":
        return generate_motif_dataset(cfg=cfg, seed=seed)
    else:
        raise ValueError(f"Unsupported synthetic graph type: {gtype}")

    if directed and not G.is_directed():
        G = G.to_directed()

    data = _finalize_graph(G, feature_dim=feature_dim, seed=seed, directed=directed, weighted=weighted)
    meta = graph_metadata(data, name=f"synthetic-{gtype}")
    return DatasetBundle(data=data, name=f"synthetic-{gtype}", metadata=meta)


def generate_motif_dataset(cfg: Dict, seed: int) -> DatasetBundle:
    gcfg = cfg["graph"]
    n = int(gcfg.get("n", 300))
    feature_dim = int(cfg["feature_dim"])
    motif_count = int(gcfg.get("motif_count", 8))
    spokes = int(gcfg.get("motif_spokes", 6))

    G = nx.erdos_renyi_graph(n=n, p=float(gcfg.get("base_p", 0.02)), seed=seed)
    rng = np.random.default_rng(seed)
    important_edges = set()
    important_nodes = set()

    centers = rng.choice(np.arange(n), size=min(motif_count, n), replace=False)
    for c in centers:
        candidates = [v for v in range(n) if v != c]
        picked = rng.choice(candidates, size=min(spokes, len(candidates)), replace=False)
        important_nodes.add(int(c))
        for v in picked:
            u_i = int(c)
            v_i = int(v)
            G.add_edge(u_i, v_i)
            important_edges.add(tuple(sorted((u_i, v_i))))
            important_nodes.add(v_i)

    data = _finalize_graph(G, feature_dim=feature_dim, seed=seed, directed=False, weighted=False)
    labels = np.zeros(data.num_nodes, dtype=np.float32)
    for node in range(data.num_nodes):
        nbrs = set(G.neighbors(node))
        motif_hits = sum((tuple(sorted((node, nb))) in important_edges) for nb in nbrs)
        labels[node] = float(motif_hits + 0.1 * len(nbrs))
    data.y = torch.tensor(labels, dtype=torch.float32).view(-1, 1)

    meta = graph_metadata(data, name="synthetic-motif")
    meta["motif_count"] = motif_count
    meta["motif_spokes"] = spokes
    gt = {
        "important_edges": [list(e) for e in sorted(important_edges)],
        "important_nodes": sorted(important_nodes),
    }
    return DatasetBundle(data=data, name="synthetic-motif", metadata=meta, gt_explanations=gt)


def load_real_dataset(cfg: Dict, seed: int) -> DatasetBundle:
    dname = cfg["name"].lower()
    root = cfg.get("root", "data")
    if dname in {"cora", "citeseer", "pubmed"}:
        canonical = {
            "cora": "Cora",
            "citeseer": "CiteSeer",
            "pubmed": "PubMed",
        }[dname]
        ds = Planetoid(root=str(Path(root) / "Planetoid"), name=canonical)
        data = ds[0]
    elif dname == "karate":
        ds = KarateClub()
        data = ds[0]
    else:
        raise ValueError(f"Unsupported real dataset: {dname}")

    data = _attach_features(data, feature_dim=int(cfg.get("feature_dim", 16)), seed=seed)
    meta = graph_metadata(data, name=dname)
    return DatasetBundle(data=data, name=dname, metadata=meta)


def make_train_val_test_masks(
    num_nodes: int,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if train_ratio <= 0 or val_ratio <= 0 or train_ratio + val_ratio >= 1:
        raise ValueError("Ratios must satisfy train_ratio>0, val_ratio>0, train+val<1.")

    rng = np.random.default_rng(seed)
    idx = np.arange(num_nodes)
    rng.shuffle(idx)

    n_train = int(train_ratio * num_nodes)
    n_val = int(val_ratio * num_nodes)

    train_idx = idx[:n_train]
    val_idx = idx[n_train : n_train + n_val]
    test_idx = idx[n_train + n_val :]

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True
    return train_mask, val_mask, test_mask


def graph_metadata(data: Data, name: str) -> Dict:
    n = int(data.num_nodes)
    e = int(data.edge_index.size(1))
    avg_deg = float(e / max(n, 1))
    x = data.x.detach().cpu().numpy()
    y = data.y.detach().cpu().numpy() if getattr(data, "y", None) is not None else None

    out = {
        "name": name,
        "num_nodes": n,
        "num_edges": e,
        "avg_degree": avg_deg,
        "feature_dim": int(data.x.size(-1)),
        "feature_mean": float(np.mean(x)),
        "feature_std": float(np.std(x)),
    }
    if y is not None:
        out.update(
            {
                "label_mean": float(np.mean(y)),
                "label_std": float(np.std(y)),
                "label_min": float(np.min(y)),
                "label_max": float(np.max(y)),
            }
        )
    return out


def save_metadata(path: Path, metadata: Dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)