import torch
from torch_geometric.explain import Explainer, GNNExplainer
from torch_geometric.explain.config import ModelConfig, ExplainerConfig


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