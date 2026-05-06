import hashlib
import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from joblib import Parallel, delayed
from tqdm import tqdm


def build_adj_list(edge_index: torch.Tensor, num_nodes: int):
    adj = [[] for _ in range(num_nodes)]
    src = edge_index[0].detach().cpu().numpy()
    dst = edge_index[1].detach().cpu().numpy()
    for s, d in zip(src, dst):
        adj[int(s)].append(int(d))
    return [list(set(vs)) for vs in adj]


def _ic_spread(
    adj,
    seed_node: int,
    p: float,
    max_steps: int,
    rng: np.random.Generator,
    p_by_edge: Optional[Dict] = None,
):
    active = {seed_node}
    frontier = {seed_node}
    steps = 0
    while frontier and steps < max_steps:
        nxt = set()
        for u in frontier:
            for v in adj[u]:
                if v in active:
                    continue
                if p_by_edge is None:
                    prob = p
                else:
                    prob = float(p_by_edge.get((u, v), p_by_edge.get((v, u), p)))
                if rng.random() < prob:
                    nxt.add(v)
        active |= nxt
        frontier = nxt
        steps += 1
    return len(active)


def _lt_spread(adj, seed_node: int, max_steps: int, rng: np.random.Generator):
    n = len(adj)
    thresholds = rng.uniform(0.2, 0.8, size=n)
    active = {seed_node}
    steps = 0
    while steps < max_steps:
        changed = False
        for v in range(n):
            if v in active:
                continue
            nbrs = adj[v]
            if not nbrs:
                continue
            frac_active = np.mean([1.0 if u in active else 0.0 for u in nbrs])
            if frac_active >= thresholds[v]:
                active.add(v)
                changed = True
        if not changed:
            break
        steps += 1
    return len(active)


def _sir_spread(adj, seed_node: int, beta: float, gamma: float, max_steps: int, rng: np.random.Generator):
    susceptible = set(range(len(adj)))
    infected = {seed_node}
    recovered = set()
    susceptible.remove(seed_node)

    steps = 0
    while infected and steps < max_steps:
        new_inf = set()
        new_rec = set()
        for u in infected:
            for v in adj[u]:
                if v in susceptible and rng.random() < beta:
                    new_inf.add(v)
            if rng.random() < gamma:
                new_rec.add(u)

        susceptible -= new_inf
        infected |= new_inf
        infected -= new_rec
        recovered |= new_rec
        steps += 1
    return len(recovered | infected)


def _cache_key(edge_index: torch.Tensor, num_nodes: int, method: str, params: Dict, seed: int) -> str:
    payload = {
        "num_nodes": int(num_nodes),
        "edge_index": edge_index.detach().cpu().numpy().tolist(),
        "method": method,
        "params": params,
        "seed": int(seed),
    }
    b = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(b).hexdigest()[:24]


def generate_diffusion_labels(
    edge_index: torch.Tensor,
    num_nodes: int,
    method: str,
    params: Dict,
    mc_runs: int,
    max_steps: int,
    seed: int = 42,
    n_jobs: int = 1,
    cache_dir: str = "cache/labels",
    use_cache: bool = True,
):
    method = method.lower()
    cache_root = Path(cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)

    key = _cache_key(edge_index=edge_index, num_nodes=num_nodes, method=method, params=params, seed=seed)
    cache_file = cache_root / f"{key}.npy"
    meta_file = cache_root / f"{key}.json"

    if use_cache and cache_file.exists():
        labels = np.load(cache_file)
        return torch.tensor(labels, dtype=torch.float32).view(-1, 1)

    adj = build_adj_list(edge_index=edge_index, num_nodes=num_nodes)

    def label_for_node(v: int):
        local_rng = np.random.default_rng(seed + v)
        spreads = []
        for _ in range(mc_runs):
            if method == "ic":
                spreads.append(
                    _ic_spread(
                        adj=adj,
                        seed_node=v,
                        p=float(params.get("p", 0.1)),
                        max_steps=max_steps,
                        rng=local_rng,
                        p_by_edge=params.get("p_by_edge"),
                    )
                )
            elif method == "lt":
                spreads.append(_lt_spread(adj=adj, seed_node=v, max_steps=max_steps, rng=local_rng))
            elif method == "sir":
                spreads.append(
                    _sir_spread(
                        adj=adj,
                        seed_node=v,
                        beta=float(params.get("beta", 0.3)),
                        gamma=float(params.get("gamma", 0.2)),
                        max_steps=max_steps,
                        rng=local_rng,
                    )
                )
            else:
                raise ValueError(f"Unsupported diffusion method: {method}")
        return float(np.mean(spreads))

    if n_jobs > 1:
        labels = Parallel(n_jobs=n_jobs)(delayed(label_for_node)(v) for v in range(num_nodes))
    else:
        labels = [label_for_node(v) for v in tqdm(range(num_nodes), desc=f"Generating {method.upper()} labels")]

    labels_np = np.array(labels, dtype=np.float32)
    if use_cache:
        np.save(cache_file, labels_np)
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "method": method,
                    "params": params,
                    "mc_runs": mc_runs,
                    "max_steps": max_steps,
                    "seed": seed,
                    "num_nodes": num_nodes,
                },
                f,
                indent=2,
            )

    return torch.tensor(labels_np, dtype=torch.float32).view(-1, 1)


def generate_ic_labels(edge_index: torch.Tensor, num_nodes: int, p: float, mc_runs: int, max_steps: int, seed: int = 42):
    return generate_diffusion_labels(
        edge_index=edge_index,
        num_nodes=num_nodes,
        method="ic",
        params={"p": p},
        mc_runs=mc_runs,
        max_steps=max_steps,
        seed=seed,
    )