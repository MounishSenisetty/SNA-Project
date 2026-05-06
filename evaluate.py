"""
Evaluation module for assessing model performance.
Includes prediction metrics and ranking metrics.
"""

import numpy as np
import pandas as pd
import torch
from typing import Dict, Tuple, List
from scipy.stats import spearmanr, kendalltau
from utils import normalize_scores, get_top_k_nodes


class InfluencerEvaluator:
    """
    Evaluator for influencer prediction models.
    """
    
    @staticmethod
    def prediction_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate prediction metrics (MSE, MAE, RMSE).
        
        Args:
            y_true: Ground truth values
            y_pred: Predicted values
            
        Returns:
            Dictionary with metric names and values
        """
        mse = np.mean((y_true - y_pred) ** 2)
        mae = np.mean(np.abs(y_true - y_pred))
        rmse = np.sqrt(mse)
        
        return {
            'MSE': float(mse),
            'MAE': float(mae),
            'RMSE': float(rmse)
        }
    
    @staticmethod
    def ranking_metrics(y_true: np.ndarray, y_pred: np.ndarray, k: int = 10) -> Dict[str, float]:
        """
        Calculate ranking metrics (Spearman correlation, Precision@k, Recall@k).
        
        Args:
            y_true: Ground truth centrality scores
            y_pred: Predicted scores
            k: Top-k for ranking metrics
            
        Returns:
            Dictionary with ranking metrics
        """
        # Spearman correlation
        spearman_corr, _ = spearmanr(y_true, y_pred)
        
        # Get top-k indices
        true_top_k = set(np.argsort(-y_true)[:k])
        pred_top_k = set(np.argsort(-y_pred)[:k])
        
        # Calculate overlap
        overlap = len(true_top_k & pred_top_k)
        precision_at_k = overlap / k
        recall_at_k = overlap / k  # Both equal when |top_k| is same
        
        # Kendall tau correlation
        kendall_corr, _ = kendalltau(y_true, y_pred)
        
        return {
            'Spearman': float(spearman_corr),
            'Kendall': float(kendall_corr),
            f'Precision@{k}': float(precision_at_k),
            f'Recall@{k}': float(recall_at_k)
        }
    
    @staticmethod
    def evaluate(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        k: int = 10
    ) -> Dict:
        """
        Comprehensive evaluation.
        
        Args:
            y_true: Ground truth scores
            y_pred: Predicted scores
            k: Top-k for ranking metrics
            
        Returns:
            Dictionary with all metrics
        """
        pred_metrics = InfluencerEvaluator.prediction_metrics(y_true, y_pred)
        rank_metrics = InfluencerEvaluator.ranking_metrics(y_true, y_pred, k=k)
        
        return {**pred_metrics, **rank_metrics}
    
    @staticmethod
    def create_evaluation_table(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        k: int = 10,
        dataset_name: str = "Dataset",
        model_name: str = "Model"
    ) -> pd.DataFrame:
        """
        Create a formatted evaluation table.
        
        Args:
            y_true: Ground truth scores
            y_pred: Predicted scores
            k: Top-k value
            dataset_name: Name of dataset
            model_name: Name of model
            
        Returns:
            DataFrame with evaluation results
        """
        metrics = InfluencerEvaluator.evaluate(y_true, y_pred, k=k)
        
        df = pd.DataFrame([metrics])
        df.insert(0, 'Model', model_name)
        df.insert(0, 'Dataset', dataset_name)
        
        return df


class TopKAnalyzer:
    """
    Analyzer for top-k influencer predictions vs ground truth.
    """
    
    @staticmethod
    def analyze(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        k: int = 10
    ) -> Dict:
        """
        Analyze top-k nodes.
        
        Args:
            y_true: Ground truth scores
            y_pred: Predicted scores
            k: Number of top nodes to analyze
            
        Returns:
            Dictionary with analysis results
        """
        # Get top-k nodes
        true_top_k = get_top_k_nodes(y_true, k)
        pred_top_k = get_top_k_nodes(y_pred, k)
        
        # Get indices for overlap calculation
        true_top_indices = set([idx for idx, _ in true_top_k])
        pred_top_indices = set([idx for idx, _ in pred_top_k])
        
        overlap = true_top_indices & pred_top_indices
        
        return {
            'true_top_k': true_top_k,
            'pred_top_k': pred_top_k,
            'overlap_count': len(overlap),
            'overlap_indices': list(overlap),
            'precision': len(overlap) / k,
            'recall': len(overlap) / k
        }
    
    @staticmethod
    def create_comparison_table(analysis: Dict) -> pd.DataFrame:
        """
        Create a comparison table for top-k nodes.
        
        Args:
            analysis: Analysis results from analyze()
            
        Returns:
            DataFrame comparing true vs predicted top-k
        """
        k = len(analysis['true_top_k'])
        
        data = {
            'Rank': list(range(1, k + 1)),
            'True_Node': [idx for idx, _ in analysis['true_top_k']],
            'True_Score': [score for _, score in analysis['true_top_k']],
            'Pred_Node': [idx for idx, _ in analysis['pred_top_k']],
            'Pred_Score': [score for _, score in analysis['pred_top_k']],
            'Match': ['✓' if pred_idx in analysis['overlap_indices'] else '✗'
                     for pred_idx, _ in analysis['pred_top_k']]
        }
        
        return pd.DataFrame(data)


def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    k: int = 10,
    dataset_name: str = "Dataset",
    model_name: str = "Model"
) -> Tuple[Dict, pd.DataFrame, Dict]:
    """
    Comprehensive model evaluation.
    
    Args:
        y_true: Ground truth scores
        y_pred: Predicted scores
        k: Top-k value
        dataset_name: Dataset name
        model_name: Model name
        
    Returns:
        Tuple of (metrics_dict, evaluation_table, top_k_analysis)
    """
    # Calculate metrics
    evaluator = InfluencerEvaluator()
    metrics = evaluator.evaluate(y_true, y_pred, k=k)
    eval_table = evaluator.create_evaluation_table(y_true, y_pred, k, dataset_name, model_name)
    
    # Top-k analysis
    analyzer = TopKAnalyzer()
    top_k_analysis = analyzer.analyze(y_true, y_pred, k=k)
    
    return metrics, eval_table, top_k_analysis
