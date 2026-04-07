import numpy as np
import torch
from scipy.stats import spearmanr, kendalltau


@torch.no_grad()
def evaluate(model, data, mask):
    model.eval()
    pred = model(data.x, data.edge_index).view(-1).cpu().numpy()
    true = data.y.view(-1).cpu().numpy()

    pred_m = pred[mask.cpu().numpy()]
    true_m = true[mask.cpu().numpy()]

    mse = float(np.mean((pred_m - true_m) ** 2))
    mae = float(np.mean(np.abs(pred_m - true_m)))

    sp = spearmanr(true_m, pred_m).correlation
    kt = kendalltau(true_m, pred_m).correlation

    return {
        "mse": mse,
        "mae": mae,
        "spearman": float(sp) if sp is not None else None,
        "kendall": float(kt) if kt is not None else None,
    }