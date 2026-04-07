import yaml
import torch
import torch.optim as optim
from pathlib import Path

from data import set_seed, load_synthetic_ba_graph, load_cora, make_train_val_test_masks
from diffusion import generate_ic_labels
from model import GCNRegressor
from train import train_one_epoch, eval_loss
from evaluate import evaluate
from explain import explain_node
from viz import plot_graph_with_scores


def main():
    config_path = Path(__file__).resolve().parent / "config.yaml"
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- Load data ----
    if cfg["data"]["type"] == "synthetic":
        n = cfg["data"]["synthetic"]["n"]
        m = cfg["data"]["synthetic"]["m"]
        feat_dim = cfg["data"]["synthetic"]["feature_dim"]
        data = load_synthetic_ba_graph(n=n, m=m, feature_dim=feat_dim)
    elif cfg["data"]["type"] == "cora":
        data = load_cora()
    else:
        raise ValueError("Unknown data.type")

    # ---- Generate diffusion-based centrality labels ----
    y = generate_ic_labels(
        edge_index=data.edge_index,
        num_nodes=data.num_nodes,
        p=cfg["diffusion"]["p"],
        mc_runs=cfg["diffusion"]["mc_runs"],
        max_steps=cfg["diffusion"]["max_steps"],
        seed=cfg["seed"],
    )
    data.y = y

    # ---- Train/val/test split ----
    train_mask, val_mask, test_mask = make_train_val_test_masks(
        data.num_nodes, train_ratio=0.6, val_ratio=0.2, seed=cfg["seed"]
    )

    data = data.to(device)

    # ---- Model ----
    in_dim = data.x.size(-1)
    model = GCNRegressor(
        in_dim=in_dim,
        hidden_dim=cfg["model"]["hidden_dim"],
        num_layers=cfg["model"]["num_layers"],
        dropout=cfg["model"]["dropout"],
    ).to(device)

    optimizer = optim.Adam(
        model.parameters(),
        lr=cfg["model"]["lr"],
        weight_decay=cfg["model"]["weight_decay"],
    )

    # ---- Training loop ----
    best_val = float("inf")
    best_state = None

    for epoch in range(1, cfg["model"]["epochs"] + 1):
        loss = train_one_epoch(model, data, optimizer, train_mask)
        val = eval_loss(model, data, val_mask)

        if val < best_val:
            best_val = val
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 20 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d} | train MSE {loss:.4f} | val MSE {val:.4f}")

    # Restore best
    if best_state is not None:
        model.load_state_dict(best_state)

    # ---- Evaluation ----
    metrics = evaluate(model, data, test_mask)
    print("\nTest metrics:", metrics)

    # ---- Visualization ----
    with torch.no_grad():
        pred = model(data.x, data.edge_index).cpu()
        plot_graph_with_scores(data.cpu(), data.y.cpu(), title="True diffusion centrality")
        plot_graph_with_scores(data.cpu(), pred, title="Predicted centrality")

    # ---- Explainability ----
    node_id = int(cfg["explain"]["node_id"])
    explanation = explain_node(model, data, node_id=node_id, device=device)
    print("\nExplanation object keys:", explanation.available_explanations)
    if explanation.edge_mask is not None:
        print("Edge mask shape:", explanation.edge_mask.shape)
    if explanation.node_mask is not None:
        print("Node feature mask shape:", explanation.node_mask.shape)


if __name__ == "__main__":
    main()