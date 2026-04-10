import torch
import numpy as np
from torch_geometric.explain import Explainer, GNNExplainer, PGExplainer
from torch_geometric.explain.config import ModelConfig


def explain_node(model, data, node_id: int, device="cpu"):
    model.eval()
    model = model.to(device)
    data = data.to(device)

    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=200),
        explanation_type="model",
        node_mask_type="attributes",
        edge_mask_type="object",
        model_config=ModelConfig(
            mode="regression",
            task_level="node",
            return_type="raw",
        ),
    )

    explanation = explainer(data.x, data.edge_index, index=node_id)
    return explanation


def explain_node_pgexplainer(model, data, node_id: int, device="cpu"):
    model.eval()
    model = model.to(device)
    data = data.to(device)

    algorithm = PGExplainer(epochs=30, lr=0.003)
    explainer = Explainer(
        model=model,
        algorithm=algorithm,
        explanation_type="model",
        node_mask_type="attributes",
        edge_mask_type="object",
        model_config=ModelConfig(mode="regression", task_level="node", return_type="raw"),
    )
    for epoch in range(algorithm.epochs):
        algorithm.train(epoch, model, data.x, data.edge_index, target=data.y)
    return explainer(data.x, data.edge_index, index=node_id)


def explain_gradient_saliency(model, data, node_id: int, device="cpu"):
    model.eval()
    x = data.x.clone().detach().to(device).requires_grad_(True)
    edge_index = data.edge_index.to(device)
    out = model(x, edge_index).view(-1)
    out[node_id].backward()
    node_mask = x.grad.detach().abs()
    edge_mask = torch.ones(edge_index.size(1), device=device)
    return {"node_mask": node_mask, "edge_mask": edge_mask}


def random_explanation(data, device="cpu"):
    node_mask = torch.rand_like(data.x.to(device))
    edge_mask = torch.rand(data.edge_index.size(1), device=device)
    return {"node_mask": node_mask, "edge_mask": edge_mask}


@torch.no_grad()
def deletion_insertion_fidelity(model, data, node_id: int, edge_mask: torch.Tensor, k_values):
    model.eval()
    baseline = model(data.x, data.edge_index).view(-1)[node_id].item()
    edge_mask = edge_mask.detach().cpu().numpy()
    order = np.argsort(-edge_mask)
    e = data.edge_index.size(1)

    delete_curve = []
    insert_curve = []
    for k in k_values:
        k = int(min(k, e))
        keep_idx = order[:k]
        del_idx = order[k:]

        edge_keep = data.edge_index[:, keep_idx].to(data.x.device)
        edge_del = data.edge_index[:, del_idx].to(data.x.device)

        pred_insert = model(data.x, edge_keep).view(-1)[node_id].item() if edge_keep.size(1) > 0 else 0.0
        pred_delete = model(data.x, edge_del).view(-1)[node_id].item() if edge_del.size(1) > 0 else 0.0

        insert_curve.append(float(pred_insert / (baseline + 1e-9)))
        delete_curve.append(float(pred_delete / (baseline + 1e-9)))

    auc_insert = float(np.trapezoid(insert_curve, x=np.array(k_values))) if len(k_values) > 1 else float(insert_curve[0])
    auc_delete = float(np.trapezoid(delete_curve, x=np.array(k_values))) if len(k_values) > 1 else float(delete_curve[0])
    return {
        "insert_curve": insert_curve,
        "delete_curve": delete_curve,
        "auc_insert": auc_insert,
        "auc_delete": auc_delete,
    }


def jaccard_at_k(mask_a: torch.Tensor, mask_b: torch.Tensor, k: int):
    a = torch.topk(mask_a.view(-1), k=min(k, mask_a.numel())).indices.cpu().numpy().tolist()
    b = torch.topk(mask_b.view(-1), k=min(k, mask_b.numel())).indices.cpu().numpy().tolist()
    sa, sb = set(a), set(b)
    return len(sa & sb) / max(len(sa | sb), 1)