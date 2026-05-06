"""
Comprehensive example demonstrating all features of the project.
Run this to see the complete workflow.
"""

import os
import sys
import pandas as pd
from pathlib import Path

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import run_experiment
from dataset import InfluencerDataset
from evaluate import TopKAnalyzer, InfluencerEvaluator
from explain import ExplainabilityAnalyzer
import warnings
warnings.filterwarnings('ignore')


def print_section(title):
    """Print a formatted section header."""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def example_1_basic_training():
    """Example 1: Basic model training and evaluation."""
    print_section("EXAMPLE 1: BASIC MODEL TRAINING")
    
    print("\n[1] Training GCN on Karate Club dataset...")
    results = run_experiment(
        dataset_name='Karate',
        centrality_method='pagerank',
        model_name='gcn',
        epochs=100,
        seed=42,
        output_dir='./examples_output',
        save_artifacts=True
    )
    
    print("\n[2] Results Summary:")
    print(results['eval_table'].to_string(index=False))
    
    print(f"\n[3] Output saved to: {results['output_path']}")
    
    return results


def example_2_model_comparison():
    """Example 2: Compare different models."""
    print_section("EXAMPLE 2: MODEL COMPARISON")
    
    models = ['gcn', 'graphsage']
    all_results = {}
    
    for model_name in models:
        print(f"\n[Training] {model_name.upper()} on Karate Club...")
        results = run_experiment(
            dataset_name='Karate',
            model_name=model_name,
            epochs=100,
            seed=42,
            output_dir='./examples_output',
            save_artifacts=False
        )
        all_results[model_name] = results
    
    # Compare
    print("\n[Comparison] Model Performance:")
    comparison_data = []
    for model_name, results in all_results.items():
        metrics = results['metrics']
        comparison_data.append({
            'Model': model_name.upper(),
            'MSE': f"{metrics['MSE']:.6f}",
            'Spearman': f"{metrics['Spearman']:.4f}",
            'Precision@10': f"{metrics['Precision@10']:.2f}"
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    print(comparison_df.to_string(index=False))
    
    return all_results


def example_3_centrality_comparison():
    """Example 3: Compare different centrality methods."""
    print_section("EXAMPLE 3: CENTRALITY METHOD COMPARISON")
    
    centralities = ['degree', 'pagerank', 'eigenvector']
    all_results = {}
    
    for centrality in centralities:
        print(f"\n[Training] Using {centrality.upper()} as labels...")
        results = run_experiment(
            dataset_name='Karate',
            centrality_method=centrality,
            epochs=100,
            seed=42,
            output_dir='./examples_output',
            save_artifacts=False
        )
        all_results[centrality] = results
    
    # Compare
    print("\n[Comparison] Centrality Method Performance:")
    comparison_data = []
    for method, results in all_results.items():
        metrics = results['metrics']
        comparison_data.append({
            'Centrality': method.capitalize(),
            'MSE': f"{metrics['MSE']:.6f}",
            'Spearman': f"{metrics['Spearman']:.4f}",
            'Precision@10': f"{metrics['Precision@10']:.2f}"
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    print(comparison_df.to_string(index=False))
    
    return all_results


def example_4_detailed_analysis():
    """Example 4: Detailed analysis of predictions."""
    print_section("EXAMPLE 4: DETAILED ANALYSIS")
    
    results = run_experiment(
        dataset_name='Karate',
        epochs=100,
        seed=42,
        output_dir='./examples_output',
        save_artifacts=True
    )
    
    print("\n[1] Top-K Analysis:")
    top_k = results['top_k_analysis']
    print(f"   • Overlap between predicted and actual top-10: {top_k['overlap_count']}/10")
    print(f"   • Precision@10: {top_k['precision']:.2%}")
    print(f"   • Recall@10: {top_k['recall']:.2%}")
    
    print("\n[2] Top-10 Predicted Influential Nodes:")
    for i, (node_id, score) in enumerate(top_k['pred_top_k'][:10], 1):
        marker = "✓" if node_id in top_k['overlap_indices'] else "✗"
        print(f"   {i:2d}. Node {node_id:3d} (score: {score:.4f}) {marker}")
    
    print("\n[3] Detailed Predictions:")
    pred_df = pd.DataFrame({
        'Node': range(len(results['y_pred'])),
        'Actual': results['y_true'],
        'Predicted': results['y_pred'],
        'Error': abs(results['y_true'] - results['y_pred'])
    })
    
    print("   Top 5 predictions:")
    top_pred = pred_df.nlargest(5, 'Predicted')[['Node', 'Actual', 'Predicted', 'Error']]
    print(top_pred.to_string(index=False))
    
    print("\n   Worst 5 predictions (highest error):")
    worst_pred = pred_df.nlargest(5, 'Error')[['Node', 'Actual', 'Predicted', 'Error']]
    print(worst_pred.to_string(index=False))
    
    return results


def example_5_explainability():
    """Example 5: Generate and analyze explanations."""
    print_section("EXAMPLE 5: EXPLAINABILITY ANALYSIS")
    
    results = run_experiment(
        dataset_name='Karate',
        epochs=100,
        seed=42,
        output_dir='./examples_output',
        save_artifacts=True
    )
    
    explanations = results['explanations']
    
    print(f"\n[Explanations for Top-3 Influential Nodes]")
    
    for i, exp in enumerate(explanations, 1):
        print(f"\n[{i}] Node {exp['target_node']}")
        print(f"    • Predicted influence score: {exp['prediction']:.4f}")
        print(f"    • Number of neighbors: {exp['num_neighbors']}")
        
        print(f"    • Important neighbors:")
        for node, score in exp['important_neighbors'][:3]:
            print(f"      - Node {node:3d} (importance: {score:.4f})")
        
        print(f"    • Important edges:")
        for src, tgt, score in exp['important_edges'][:3]:
            print(f"      - Edge {src}-{tgt} (importance: {score:.4f})")
    
    return results


def example_6_multi_dataset():
    """Example 6: Train on multiple datasets."""
    print_section("EXAMPLE 6: MULTI-DATASET TRAINING")
    
    datasets = ['Karate', 'Cora']
    all_results = {}
    
    for dataset_name in datasets:
        print(f"\n[Training on {dataset_name}...]")
        results = run_experiment(
            dataset_name=dataset_name,
            epochs=100 if dataset_name == 'Karate' else 200,
            seed=42,
            output_dir='./examples_output',
            save_artifacts=True
        )
        all_results[dataset_name] = results
    
    print("\n[Comparison across datasets]")
    comparison_data = []
    for dataset_name, results in all_results.items():
        metrics = results['metrics']
        y_pred = results['y_pred']
        comparison_data.append({
            'Dataset': dataset_name,
            'Nodes': len(y_pred),
            'MSE': f"{metrics['MSE']:.6f}",
            'Spearman': f"{metrics['Spearman']:.4f}",
            'Precision@10': f"{metrics['Precision@10']:.2f}"
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    print(comparison_df.to_string(index=False))
    
    return all_results


def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("  COMPREHENSIVE PROJECT EXAMPLES")
    print("  Explainable Influencer Detection in Social Networks")
    print("="*70)
    
    # Create output directory
    Path('./examples_output').mkdir(parents=True, exist_ok=True)
    
    # Run examples
    print("\nSelect examples to run:")
    print("1. Basic Training")
    print("2. Model Comparison")
    print("3. Centrality Method Comparison")
    print("4. Detailed Analysis")
    print("5. Explainability")
    print("6. Multi-Dataset Training")
    print("7. Run All")
    print("0. Exit")
    
    choice = input("\nEnter choice (0-7): ").strip()
    
    examples = {
        '1': example_1_basic_training,
        '2': example_2_model_comparison,
        '3': example_3_centrality_comparison,
        '4': example_4_detailed_analysis,
        '5': example_5_explainability,
        '6': example_6_multi_dataset,
    }
    
    if choice == '7':
        # Run all
        for example_func in examples.values():
            try:
                example_func()
            except Exception as e:
                print(f"\nError in example: {e}")
                import traceback
                traceback.print_exc()
    elif choice in examples:
        try:
            examples[choice]()
        except Exception as e:
            print(f"\nError: {e}")
            import traceback
            traceback.print_exc()
    elif choice == '0':
        print("Exiting...")
    else:
        print("Invalid choice!")
    
    print("\n" + "="*70)
    print("  EXAMPLES COMPLETED")
    print(f"  Results saved to: ./examples_output/")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
