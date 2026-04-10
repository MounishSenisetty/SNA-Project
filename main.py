import yaml
import torch
import torch.optim as optim
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime

from data import (
    generate_synthetic_graph,
    graph_metadata,
    load_real_dataset,
    make_train_val_test_masks,
    save_metadata,
    set_seed,
)
from model import build_model
from train import fit_model
from evaluate import aggregate_seed_metrics, evaluate
from explain import (
    deletion_insertion_fidelity,
    explain_gradient_saliency,
    explain_node,
    explain_node_pgexplainer,
    jaccard_at_k,
    random_explanation,
)
from viz import plot_feature_importance, plot_fidelity_curves, plot_graph_with_scores
from baselines import build_tabular_features, ml_baselines, non_learning_baselines
from targets import build_targets
from utils import bootstrap_ci, ensure_dir, write_csv, write_json


def load_cfg(config_path: Path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_dataset(cfg, seed):
    dcfg = cfg["data"]
    if dcfg["source"] == "synthetic":
        return generate_synthetic_graph(cfg=dcfg["synthetic"], seed=seed)
    if dcfg["source"] == "real":
        return load_real_dataset(cfg=dcfg["real"], seed=seed)
    raise ValueError("data.source must be synthetic or real")


def save_explanation_artifacts(run_dir: Path, node_id: int, explanation, prefix: str):
    out = {}
    node_mask = getattr(explanation, "node_mask", None)
    edge_mask = getattr(explanation, "edge_mask", None)
    if node_mask is not None:
        node_scores = node_mask.detach().cpu().numpy()
        np.save(run_dir / f"{prefix}_node_mask.npy", node_scores)
        out["top_features"] = np.argsort(-node_scores.reshape(-1)).tolist()[:20]
    if edge_mask is not None:
        edge_scores = edge_mask.detach().cpu().numpy()
        np.save(run_dir / f"{prefix}_edge_mask.npy", edge_scores)
        out["top_edges_indices"] = np.argsort(-edge_scores).tolist()[:50]
    write_json(run_dir / f"{prefix}_node_{node_id}.json", out)


def run_single_seed(cfg, model_name: str, seed: int, out_dir: Path, no_viz: bool):
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() and cfg["train"].get("device", "auto") != "cpu" else "cpu"

    bundle = resolve_dataset(cfg, seed=seed)
    data = bundle.data
    y, target_meta = build_targets(data=data, cfg=cfg, seed=seed)
    data.y = y

    train_mask, val_mask, test_mask = make_train_val_test_masks(
        data.num_nodes,
        train_ratio=float(cfg["split"]["train_ratio"]),
        val_ratio=float(cfg["split"]["val_ratio"]),
        seed=seed,
    )

    data = data.to(device)
    model = build_model(
        name=model_name,
        in_dim=data.x.size(-1),
        hidden_dim=int(cfg["model"]["hidden_dim"]),
        num_layers=int(cfg["model"]["num_layers"]),
        dropout=float(cfg["model"]["dropout"]),
        out_dim=data.y.size(-1),
    ).to(device)

    optimizer = optim.Adam(
        model.parameters(),
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(cfg["train"].get("lr_decay", 0.5)),
        patience=int(cfg["train"].get("lr_patience", 10)),
    )

    fit_info = fit_model(
        model=model,
        data=data,
        train_mask=train_mask,
        val_mask=val_mask,
        optimizer=optimizer,
        epochs=int(cfg["train"]["epochs"]),
        patience=int(cfg["train"]["early_stopping_patience"]),
        scheduler=scheduler,
        grad_clip=cfg["train"].get("grad_clip"),
        verbose_every=int(cfg["train"].get("log_every", 20)),
    )

    metrics = evaluate(
        model=model,
        data=data,
        mask=test_mask,
        topk_values=tuple(cfg["metrics"].get("topk", [10, 20, 50])),
    )
    metrics.update(
        {
            "seed": seed,
            "model": model_name,
            "dataset": bundle.name,
            "target_names": target_meta["target_names"],
            "train_seconds": fit_info["train_seconds"],
        }
    )

    # Baseline metrics on primary target only.
    with torch.no_grad():
        pred = model(data.x, data.edge_index).view(data.num_nodes, -1).detach().cpu().numpy()[:, 0]
        true = data.y.detach().cpu().numpy()[:, 0]

    baseline_rows = []
    for bname, bpred in non_learning_baselines(data.cpu(), test_mask).items():
        idx = test_mask.cpu().numpy()
        mse = float(np.mean((bpred - true[idx]) ** 2))
        baseline_rows.append({"baseline": bname, "mse": mse, "seed": seed})

    X = build_tabular_features(data.cpu())
    preds_ml = ml_baselines(X=X, y=true, train_mask=train_mask, test_mask=test_mask)
    idx = test_mask.cpu().numpy()
    for bname, bpred in preds_ml.items():
        mse = float(np.mean((bpred - true[idx]) ** 2))
        baseline_rows.append({"baseline": bname, "mse": mse, "seed": seed})

    run_dir = ensure_dir(out_dir / f"seed_{seed}" / model_name)
    write_json(run_dir / "metrics.json", metrics)
    write_json(run_dir / "fit_info.json", fit_info)
    write_json(run_dir / "dataset_metadata.json", graph_metadata(data.cpu(), bundle.name))
    if bundle.gt_explanations is not None:
        write_json(run_dir / "ground_truth_explanations.json", bundle.gt_explanations)
    save_metadata(run_dir / "task_metadata.json", target_meta)
    write_csv(run_dir / "baseline_metrics.csv", baseline_rows)
    torch.save(model.state_dict(), run_dir / "checkpoint.pt")

    # Explanations
    node_id = int(cfg["explain"].get("node_id", 0))
    node_id = min(max(node_id, 0), data.num_nodes - 1)
    methods = cfg["explain"].get("methods", ["gnnexplainer", "gradient", "random"])
    explain_rows = []
    for method in methods:
        method_l = method.lower()
        if method_l == "gnnexplainer":
            exp = explain_node(model, data, node_id=node_id, device=device)
            edge_mask = exp.edge_mask
            node_mask = exp.node_mask
            save_explanation_artifacts(run_dir, node_id=node_id, explanation=exp, prefix="gnnexplainer")
        elif method_l == "pgexplainer":
            exp = explain_node_pgexplainer(model, data, node_id=node_id, device=device)
            edge_mask = exp.edge_mask
            node_mask = exp.node_mask
            save_explanation_artifacts(run_dir, node_id=node_id, explanation=exp, prefix="pgexplainer")
        elif method_l == "gradient":
            exp = explain_gradient_saliency(model, data, node_id=node_id, device=device)
            edge_mask = exp["edge_mask"]
            node_mask = exp["node_mask"]
        elif method_l == "random":
            exp = random_explanation(data, device=device)
            edge_mask = exp["edge_mask"]
            node_mask = exp["node_mask"]
        else:
            continue

        k_values = cfg["explain"].get("fidelity_k", [5, 10, 20, 40])
        fid = deletion_insertion_fidelity(
            model=model,
            data=data,
            node_id=node_id,
            edge_mask=edge_mask,
            k_values=k_values,
        )
        st = {
            "jaccard_edge_topk": jaccard_at_k(edge_mask, random_explanation(data, device=device)["edge_mask"], k=20),
            "jaccard_feat_topk": jaccard_at_k(node_mask, random_explanation(data, device=device)["node_mask"], k=20),
        }
        explain_rows.append({"method": method_l, **fid, **st, "seed": seed})
        plot_fidelity_curves(
            k_values=k_values,
            insert_curve=fid["insert_curve"],
            delete_curve=fid["delete_curve"],
            save_path=run_dir / f"fidelity_{method_l}.png",
        )
        if node_mask is not None:
            plot_feature_importance(
                node_mask[node_id].detach().cpu().numpy()
                if node_mask.dim() > 1
                else node_mask.detach().cpu().numpy(),
                title=f"Feature attribution ({method_l})",
                save_path=run_dir / f"feature_attr_{method_l}.png",
            )

    write_csv(run_dir / "explanations.csv", explain_rows)

    if not no_viz:
        plot_graph_with_scores(data.cpu(), torch.tensor(true), title="True influence", save_path=run_dir / "true_scores.png")
        plot_graph_with_scores(data.cpu(), torch.tensor(pred), title="Predicted influence", save_path=run_dir / "pred_scores.png")

    rows = [{"node_id": i, "y_true": float(true[i]), "y_pred": float(pred[i])} for i in range(len(true))]
    write_csv(run_dir / "predictions.csv", rows)
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Research-grade explainable centrality experiments")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to YAML config")
    parser.add_argument("--no-viz", action="store_true", help="Disable visualization generation")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    cfg = load_cfg(config_path)

    exp_name = cfg.get("experiment_name", f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    out_root = ensure_dir(Path(cfg.get("output_root", "runs")) / exp_name)
    write_json(out_root / "resolved_config.json", cfg)

    seeds = cfg.get("seeds", [cfg.get("seed", 42)])
    models = cfg["model"].get("names", [cfg["model"].get("name", "gcn")])

    all_metrics = []
    for model_name in models:
        model_seed_metrics = []
        for seed in seeds:
            print(f"\n=== Running model={model_name} seed={seed} ===")
            m = run_single_seed(cfg=cfg, model_name=model_name, seed=int(seed), out_dir=out_root, no_viz=args.no_viz)
            all_metrics.append(m)
            model_seed_metrics.append(m)

        agg = aggregate_seed_metrics(model_seed_metrics)
        for k, v in list(agg.items()):
            vals = [row[k] for row in model_seed_metrics if isinstance(row.get(k), (int, float))]
            ci = bootstrap_ci(vals) if vals else {"low": None, "high": None}
            v["ci95"] = ci
        write_json(out_root / f"aggregate_{model_name}.json", agg)

    write_csv(out_root / "all_metrics.csv", all_metrics)
    print(f"\nCompleted experiments. Artifacts written to: {out_root}")


if __name__ == "__main__":
    main()