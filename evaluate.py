import numpy as np
import torch
from scipy.stats import spearmanr, kendalltau
from sklearn.metrics import r2_score


@torch.no_grad()
def evaluate(model, data, mask, topk_values=(10, 20, 50)):
    model.eval()
    pred = model(data.x, data.edge_index).view(-1).cpu().numpy()
    true = data.y.view(-1).cpu().numpy()

    pred_m = pred[mask.cpu().numpy()]
    true_m = true[mask.cpu().numpy()]

    mse = float(np.mean((pred_m - true_m) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(pred_m - true_m)))
    r2 = float(r2_score(true_m, pred_m)) if len(true_m) > 1 else 0.0

    sp = spearmanr(true_m, pred_m).correlation
    kt = kendalltau(true_m, pred_m).correlation

    ranking = compute_ranking_metrics(true_m, pred_m, topk_values=topk_values)

    out = {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "spearman": float(sp) if sp is not None else None,
        "kendall": float(kt) if kt is not None else None,
    }
    out.update(ranking)
    return out


def _topk_idx(a: np.ndarray, k: int):
    k = min(k, len(a))
    if k <= 0:
        return np.array([], dtype=np.int64)
    return np.argpartition(-a, kth=k - 1)[:k]


def _ndcg_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int):
    k = min(k, len(y_true))
    if k <= 0:
        return 0.0
    order = np.argsort(-y_score)[:k]
    gains = y_true[order]
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = float(np.sum(gains * discounts))

    ideal = np.sort(y_true)[::-1][:k]
    idcg = float(np.sum(ideal * discounts)) + 1e-12
    return dcg / idcg


def compute_ranking_metrics(y_true: np.ndarray, y_pred: np.ndarray, topk_values=(10, 20, 50)):
    out = {}
    for k in topk_values:
        true_top = set(_topk_idx(y_true, k).tolist())
        pred_top = set(_topk_idx(y_pred, k).tolist())
        inter = len(true_top & pred_top)

        precision = inter / max(len(pred_top), 1)
        recall = inter / max(len(true_top), 1)
        union = len(true_top | pred_top)
        jaccard = inter / max(union, 1)
        ndcg = _ndcg_at_k(y_true=y_true, y_score=y_pred, k=k)

        out[f"precision@{k}"] = float(precision)
        out[f"recall@{k}"] = float(recall)
        out[f"ndcg@{k}"] = float(ndcg)
        out[f"jaccard_top@{k}"] = float(jaccard)
    return out


def aggregate_seed_metrics(seed_metrics):
    keys = sorted(seed_metrics[0].keys())
    out = {}
    for k in keys:
        vals = [m[k] for m in seed_metrics if isinstance(m.get(k), (float, int))]
        if not vals:
            continue
        out[k] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "n": int(len(vals)),
        }
    return out