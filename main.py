"""
Main training and evaluation pipeline.
"""

import os
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from dataset import InfluencerDataset
from model import ModelFactory
from train import train_model
from evaluate import evaluate_model, TopKAnalyzer
from explain import ExplainabilityAnalyzer, create_explanation_subgraph
from visualize import create_result_summary_plots
from utils import set_seed, ensure_dir, save_dict_to_json, get_device, print_device_info


def run_experiment(
    dataset_name: str = 'Cora',
    centrality_method: str = 'pagerank',
    model_name: str = 'gcn',
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    hidden_dim: int = 64,
    learning_rate: float = 0.01,
    weight_decay: float = 5e-4,
    epochs: int = 200,
    batch_size: int = None,
    seed: int = 42,
    output_dir: str = './results',
    save_artifacts: bool = True
) -> Dict:
    """
    Run complete influencer detection experiment.
    
    Args:
        dataset_name: Dataset to use
        centrality_method: Centrality method for labels
        model_name: Model architecture
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        hidden_dim: Hidden layer dimension
        learning_rate: Learning rate
        weight_decay: Weight decay
        epochs: Number of training epochs
        batch_size: Batch size (not used for node-level prediction)
        seed: Random seed
        output_dir: Directory to save results
        save_artifacts: Whether to save plots and results
        
    Returns:
        Dictionary with results
    """
    # Set seed
    set_seed(seed)
    device = get_device()
    
    print(f"\n{'='*60}")
    print(f"INFLUENCER DETECTION EXPERIMENT")
    print(f"{'='*60}")
    
    # Print device info
    print_device_info()
    
    # Create output directory
    output_path = ensure_dir(os.path.join(output_dir, dataset_name, model_name, f'seed_{seed}'))
    
    # Load dataset
    print(f"\n[1/6] Loading dataset: {dataset_name}")
    dataset = InfluencerDataset(name=dataset_name, root='./data')
    data = dataset.load_dataset()
    centralities = dataset.generate_all_centralities()
    data.y = dataset.get_labels(centrality_method)
    dataset.info()
    
    print(f"Using {centrality_method} centrality as labels")
    
    # Create train/val/test split
    print(f"\n[2/6] Creating train/val/test split")
    num_nodes = data.num_nodes
    indices = np.arange(num_nodes)
    np.random.shuffle(indices)
    
    train_size = int(num_nodes * train_ratio)
    val_size = int(num_nodes * val_ratio)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    
    train_mask[train_indices] = True
    val_mask[val_indices] = True
    test_mask[test_indices] = True
    
    print(f"Train: {train_size} ({train_ratio*100:.1f}%)")
    print(f"Val:   {val_size} ({val_ratio*100:.1f}%)")
    print(f"Test:  {len(test_indices)} ({test_ratio*100:.1f}%)")
    
    # Create model
    print(f"\n[3/6] Creating model: {model_name.upper()}")
    model = ModelFactory.create_model(
        model_name,
        input_dim=data.num_features,
        hidden_dim=hidden_dim,
        output_dim=1,
        num_layers=2,
        dropout=0.5
    )
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Train model
    print(f"\n[4/6] Training model for {epochs} epochs")
    results = train_model(
        model=model,
        data=data,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        device=device,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        epochs=epochs,
        early_stopping_patience=20,
        checkpoint_dir=os.path.join(output_path, 'checkpoints'),
        verbose=True
    )
    
    print(f"Training completed at epoch {results['best_epoch']}")
    print(f"Best validation loss: {results['best_val_loss']:.6f}")
    print(f"Test loss: {results['test_loss']:.6f}")
    
    # Evaluate model
    print(f"\n[5/6] Evaluating model")
    y_pred = results['final_predictions']
    y_true_np = data.y.numpy()
    
    metrics, eval_table, top_k_analysis = evaluate_model(
        y_true_np,
        y_pred,
        k=10,
        dataset_name=dataset_name,
        model_name=model_name.upper()
    )
    
    print("\nEvaluation Metrics:")
    print(eval_table.to_string(index=False))
    
    print("\nTop-10 Analysis:")
    print(f"Overlap between predicted and actual top-10: {top_k_analysis['overlap_count']}/10")
    print(f"Precision@10: {top_k_analysis['precision']:.4f}")
    print(f"Recall@10: {top_k_analysis['recall']:.4f}")
    
    # Explainability analysis
    print(f"\n[6/6] Generating explanations")
    analyzer = ExplainabilityAnalyzer(model, device)
    
    # Explain top-3 nodes
    explanations = analyzer.explain_top_k_nodes(data, y_pred, k=3, k_neighbors=5)
    
    print(f"Generated explanations for top-3 influential nodes:")
    for i, exp in enumerate(explanations, 1):
        print(f"  Node {exp['target_node']}: prediction={exp['prediction']:.4f}, "
              f"neighbors={len(exp['important_neighbors'])}")
    
    # Save artifacts
    if save_artifacts:
        print(f"\nSaving artifacts to: {output_path}")
        
        # Save metrics
        metrics_path = os.path.join(output_path, 'metrics.json')
        save_dict_to_json(metrics, metrics_path)
        
        # Save evaluation table
        eval_table_path = os.path.join(output_path, 'evaluation_table.csv')
        eval_table.to_csv(eval_table_path, index=False)
        
        # Save top-k analysis table
        comparison_table = TopKAnalyzer.create_comparison_table(top_k_analysis)
        topk_path = os.path.join(output_path, 'top_k_comparison.csv')
        comparison_table.to_csv(topk_path, index=False)
        
        # Save predictions
        predictions_path = os.path.join(output_path, 'predictions.csv')
        pred_df = pd.DataFrame({
            'node_id': range(len(y_pred)),
            'actual': y_true_np,
            'predicted': y_pred
        })
        pred_df.to_csv(predictions_path, index=False)
        
        # Save explanations
        explanations_path = os.path.join(output_path, 'explanations.json')
        explanations_dict = [
            {
                'target_node': int(exp['target_node']),
                'prediction': float(exp['prediction']),
                'neighbors': [(int(node), float(score)) for node, score in exp['important_neighbors']],
                'edges': [(int(src), int(tgt), float(score)) for src, tgt, score in exp['important_edges']]
            }
            for exp in explanations
        ]
        save_dict_to_json(explanations_dict, explanations_path)
        
        # Create visualizations
        print("Creating visualizations...")
        visualize_dir = os.path.join(output_path, 'visualizations')
        saved_plots = create_result_summary_plots(
            results['history'],
            y_true_np,
            y_pred,
            metrics,
            top_k_analysis,
            output_dir=visualize_dir
        )
        
        # Save subgraph explanations
        for i, exp in enumerate(explanations):
            important_nodes = [node for node, _ in exp['important_neighbors']]
            plot_path = os.path.join(
                visualize_dir,
                f'explanation_node_{exp["target_node"]}.png'
            )
            from visualize import plot_explanation_subgraph
            plot_explanation_subgraph(
                data,
                exp['target_node'],
                important_nodes,
                exp,
                save_path=plot_path
            )
        
        print(f"✓ Artifacts saved to {output_path}")
    
    # Summary
    print(f"\n{'='*60}")
    print(f"EXPERIMENT COMPLETED SUCCESSFULLY")
    print(f"{'='*60}\n")
    
    results['metrics'] = metrics
    results['eval_table'] = eval_table
    results['top_k_analysis'] = top_k_analysis
    results['explanations'] = explanations
    results['y_pred'] = y_pred
    results['y_true'] = y_true_np
    results['output_path'] = output_path
    
    return results


if __name__ == '__main__':
    import sys
    from typing import Dict
    
    # Run default experiment
    results = run_experiment(
        dataset_name='Cora',
        centrality_method='pagerank',
        model_name='gcn',
        hidden_dim=64,
        learning_rate=0.01,
        weight_decay=5e-4,
        epochs=200,
        seed=42,
        output_dir='./results'
    )
