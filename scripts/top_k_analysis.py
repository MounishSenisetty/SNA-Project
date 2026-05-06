"""
Top-K Influencer Analysis Script
Compare predicted vs actual top-k influential nodes
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import json

from dataset import InfluencerDataset
from evaluate import TopKAnalyzer
from utils import get_top_k_nodes


def analyze_top_k_influencers(
    dataset_name: str,
    centrality_method: str,
    predictions: np.ndarray,
    k: int = 10,
    save_path: str = None
):
    """
    Analyze top-k influencers and compare with ground truth.
    
    Args:
        dataset_name: Name of dataset
        centrality_method: Centrality method for ground truth
        predictions: Model predictions
        k: Number of top nodes
        save_path: Path to save analysis
        
    Returns:
        Analysis dataframe
    """
    # Load dataset
    dataset = InfluencerDataset(name=dataset_name)
    dataset.load_dataset()
    dataset.create_networkx_graph()
    y_true = dataset.compute_centrality(centrality_method)
    
    # Run analysis
    analyzer = TopKAnalyzer()
    analysis = analyzer.analyze(y_true, predictions, k=k)
    
    # Create comparison table
    comp_table = analyzer.create_comparison_table(analysis)
    
    print(f"\n{'='*80}")
    print(f\"TOP-{k} INFLUENCER ANALYSIS: {dataset_name}\")
    print(f\"Centrality Method: {centrality_method.upper()}\")
    print(f\"{'='*80}\")
    print(f\"\n{comp_table.to_string(index=False)}\")
    
    print(f\"\n{'='*80}\")
    print(f\"SUMMARY:\")
    print(f\"  • Overlap (Nodes in both top-{k}): {analysis['overlap_count']}/{k}\")
    print(f\"  • Precision@{k}: {analysis['precision']:.4f}\")
    print(f\"  • Recall@{k}: {analysis['recall']:.4f}\")
    print(f\"  • Match Rate: {(analysis['overlap_count']/k)*100:.1f}%\")
    print(f\"{'='*80}\\n\")
    
    # Save if path provided
    if save_path:
        comp_table.to_csv(save_path, index=False)
        print(f\"Analysis saved to: {save_path}\")
    
    return comp_table


def compare_multiple_methods(
    dataset_name: str,
    predictions: np.ndarray,
    k: int = 10
):
    \"\"\"
    Compare top-k analysis across all centrality methods.
    
    Args:
        dataset_name: Name of dataset
        predictions: Model predictions
        k: Number of top nodes
    \"\"\"
    methods = ['degree', 'pagerank', 'eigenvector']
    
    print(f\"\\nComparative Top-{k} Analysis for {dataset_name}\")
    print(f\"=\"*80)
    
    all_analyses = {}
    
    for method in methods:
        try:
            analysis = analyze_top_k_influencers(
                dataset_name,
                method,
                predictions,
                k=k
            )
            all_analyses[method] = analysis
        except Exception as e:
            print(f\"Error analyzing {method}: {e}\")
    
    return all_analyses


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Analyze top-K influencers'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='Cora',
        help='Dataset name (Cora, CiteSeer, Karate)'
    )
    parser.add_argument(
        '--predictions',
        type=str,
        required=True,
        help='Path to predictions CSV file'
    )
    parser.add_argument(
        '--k',
        type=int,
        default=10,
        help='Number of top nodes to analyze'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Path to save analysis results'
    )
    
    args = parser.parse_args()
    
    # Load predictions
    predictions_df = pd.read_csv(args.predictions)
    predictions = predictions_df['predicted'].values
    
    # Run analysis
    analyze_top_k_influencers(
        args.dataset,
        'pagerank',
        predictions,
        k=args.k,
        save_path=args.output
    )
