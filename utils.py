import json
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
from scipy.stats import ttest_rel, wilcoxon


def ensure_dir(path):
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_json(path, obj):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def write_csv(path, rows: List[Dict]):
    import csv

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys = sorted(rows[0].keys())
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def compare_metrics_across_seeds(metric_a: Iterable[float], metric_b: Iterable[float]):
    a = np.asarray(list(metric_a), dtype=np.float64)
    b = np.asarray(list(metric_b), dtype=np.float64)
    if len(a) != len(b) or len(a) < 2:
        return {"t_test_p": None, "wilcoxon_p": None}

    t_p = float(ttest_rel(a, b).pvalue)
    try:
        w_p = float(wilcoxon(a, b).pvalue)
    except Exception:
        w_p = None
    return {"t_test_p": t_p, "wilcoxon_p": w_p}


def bootstrap_ci(values, alpha=0.05, n_boot=1000, seed=42):
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return {"low": None, "high": None}
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(n_boot):
        sample = rng.choice(values, size=len(values), replace=True)
        means.append(float(np.mean(sample)))
    low = float(np.quantile(means, alpha / 2.0))
    high = float(np.quantile(means, 1 - alpha / 2.0))
    return {"low": low, "high": high}
