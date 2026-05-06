"""
Utility functions for the Influencer Detection project.
Includes metrics calculation, visualization helpers, and common operations.
"""

import numpy as np
import pandas as pd
import os
import json
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def ensure_dir(path: str) -> str:
    """Create directory if it doesn't exist."""
    Path(path).mkdir(parents=True, exist_ok=True)
    return path


def save_dict_to_json(data: Dict, filepath: str):
    """Save dictionary to JSON file."""
    ensure_dir(os.path.dirname(filepath))
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)


def load_dict_from_json(filepath: str) -> Dict:
    """Load dictionary from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def save_predictions_to_csv(predictions: Dict, filepath: str):
    """Save predictions to CSV file."""
    ensure_dir(os.path.dirname(filepath))
    df = pd.DataFrame(predictions)
    df.to_csv(filepath, index=False)


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate MSE, MAE, RMSE for predictions.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        
    Returns:
        Dictionary with metric values
    """
    mse = np.mean((y_true - y_pred) ** 2)
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(mse)
    
    # Calculate R² if there's variance in y_true
    if np.var(y_true) > 0:
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
    else:
        r2 = 0.0
    
    return {
        'MSE': float(mse),
        'MAE': float(mae),
        'RMSE': float(rmse),
        'R2': float(r2)
    }


def calculate_ranking_metrics(y_true: np.ndarray, y_pred: np.ndarray, k: int = 10) -> Dict[str, float]:
    """
    Calculate Spearman correlation and ranking metrics (Precision@k, Recall@k).
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        k: Number of top items to consider
        
    Returns:
        Dictionary with ranking metrics
    """
    from scipy.stats import spearmanr
    
    # Spearman correlation
    spearman_corr, spearman_pval = spearmanr(y_true, y_pred)
    
    # Get top-k indices
    true_top_k = set(np.argsort(-y_true)[:k])
    pred_top_k = set(np.argsort(-y_pred)[:k])
    
    # Calculate overlap metrics
    overlap = len(true_top_k & pred_top_k)
    precision_at_k = overlap / k
    recall_at_k = overlap / k
    
    return {
        'Spearman': float(spearman_corr),
        f'Precision@{k}': float(precision_at_k),
        f'Recall@{k}': float(recall_at_k)
    }


def get_top_k_nodes(scores: np.ndarray, k: int = 10) -> List[Tuple[int, float]]:
    """
    Get top-k nodes with their scores.
    
    Args:
        scores: Node influence scores
        k: Number of top nodes
        
    Returns:
        List of (node_id, score) tuples sorted by score
    """
    top_indices = np.argsort(-scores)[:k]
    return [(int(idx), float(scores[idx])) for idx in top_indices]


def create_evaluation_table(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    k: int = 10,
    dataset_name: str = "Dataset"
) -> pd.DataFrame:
    """
    Create a comprehensive evaluation table.
    
    Args:
        y_true: Ground truth centrality scores
        y_pred: Model predictions
        k: Number of top nodes for ranking metrics
        dataset_name: Name of the dataset
        
    Returns:
        DataFrame with evaluation results
    """
    pred_metrics = calculate_metrics(y_true, y_pred)
    rank_metrics = calculate_ranking_metrics(y_true, y_pred, k=k)
    
    # Combine metrics
    all_metrics = {**pred_metrics, **rank_metrics}
    
    # Create table
    results = pd.DataFrame([all_metrics])
    results.insert(0, 'Dataset', dataset_name)
    
    return results


def normalize_scores(scores: np.ndarray, min_val: float = 0.0, max_val: float = 1.0) -> np.ndarray:
    """
    Normalize scores to a given range.
    
    Args:
        scores: Input scores
        min_val: Minimum value of target range
        max_val: Maximum value of target range
        
    Returns:
        Normalized scores
    """
    if np.max(scores) - np.min(scores) == 0:
        return np.full_like(scores, (min_val + max_val) / 2, dtype=np.float32)
    
    normalized = (scores - np.min(scores)) / (np.max(scores) - np.min(scores))
    return normalized * (max_val - min_val) + min_val


def get_device() -> torch.device:
    """Get the appropriate device (GPU if available, else CPU)."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def print_device_info():
    """Print information about the device being used."""
    device = get_device()
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
