import numpy as np
import torch

from baselines import classical_centrality_scores
from diffusion import generate_diffusion_labels


def build_targets(data, cfg, seed: int):
    task_cfg = cfg["task"]
    target_mode = task_cfg.get("target", "diffusion")

    targets = []
    names = []

    if target_mode in {"diffusion", "multi"}:
        dcfg = cfg["diffusion"]
        y_diff = generate_diffusion_labels(
            edge_index=data.edge_index,
            num_nodes=data.num_nodes,
            method=dcfg.get("method", "ic"),
            params=dcfg.get("params", {"p": 0.1}),
            mc_runs=int(dcfg.get("mc_runs", 40)),
            max_steps=int(dcfg.get("max_steps", 50)),
            seed=seed,
            n_jobs=int(dcfg.get("n_jobs", 1)),
            cache_dir=dcfg.get("cache_dir", "cache/labels"),
            use_cache=bool(dcfg.get("use_cache", True)),
        )
        targets.append(y_diff)
        names.append("diffusion")

    if target_mode in {"classical", "multi"}:
        c_names = task_cfg.get("classical_targets", ["degree", "pagerank"])
        cs = classical_centrality_scores(data)
        for c in c_names:
            if c not in cs:
                raise ValueError(f"Unknown classical centrality: {c}")
            arr = cs[c]
            targets.append(torch.tensor(arr, dtype=torch.float32).view(-1, 1))
            names.append(c)

    if not targets:
        raise ValueError("No targets configured.")

    y = torch.cat(targets, dim=1)
    y_np = y.detach().cpu().numpy()
    mean = np.mean(y_np, axis=0, keepdims=True)
    std = np.std(y_np, axis=0, keepdims=True) + 1e-9
    y_norm = torch.tensor((y_np - mean) / std, dtype=torch.float32)
    return y_norm, {"target_names": names, "norm_mean": mean.tolist(), "norm_std": std.tolist()}
