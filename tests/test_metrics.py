import numpy as np

from evaluate import compute_ranking_metrics


def test_ranking_metrics_keys_exist():
    y_true = np.array([0.9, 0.8, 0.1, 0.2, 0.3], dtype=np.float32)
    y_pred = np.array([0.85, 0.75, 0.2, 0.1, 0.4], dtype=np.float32)
    out = compute_ranking_metrics(y_true, y_pred, topk_values=(2, 3))
    assert "precision@2" in out
    assert "recall@3" in out
    assert "ndcg@2" in out
    assert "jaccard_top@3" in out
