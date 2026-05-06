"""
Training script for Influencer Detection.
Run this to train a model and save results.
"""

import argparse
from main import run_experiment


def main():
    parser = argparse.ArgumentParser(
        description='Train influencer detection model'
    )
    
    # Dataset arguments
    parser.add_argument(
        '--dataset',
        type=str,
        default='Cora',
        choices=['Cora', 'CiteSeer', 'Karate'],
        help='Dataset name'
    )
    parser.add_argument(
        '--centrality',
        type=str,
        default='pagerank',
        choices=['degree', 'pagerank', 'eigenvector'],
        help='Centrality method for labels'
    )
    
    # Model arguments
    parser.add_argument(
        '--model',
        type=str,
        default='gcn',
        choices=['gcn', 'graphsage', 'gat'],
        help='Model architecture'
    )
    parser.add_argument(
        '--hidden-dim',
        type=int,
        default=64,
        help='Hidden layer dimension'
    )
    parser.add_argument(
        '--dropout',
        type=float,
        default=0.5,
        help='Dropout rate'
    )
    
    # Training arguments
    parser.add_argument(
        '--epochs',
        type=int,
        default=200,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=0.01,
        help='Learning rate'
    )
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=5e-4,
        help='Weight decay'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )
    
    # Data split arguments
    parser.add_argument(
        '--train-ratio',
        type=float,
        default=0.7,
        help='Training set ratio'
    )
    parser.add_argument(
        '--val-ratio',
        type=float,
        default=0.15,
        help='Validation set ratio'
    )
    
    # Output arguments
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./results',
        help='Output directory'
    )
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Do not save artifacts'
    )
    
    args = parser.parse_args()
    
    # Run experiment
    results = run_experiment(
        dataset_name=args.dataset,
        centrality_method=args.centrality,
        model_name=args.model,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=1.0 - args.train_ratio - args.val_ratio,
        hidden_dim=args.hidden_dim,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        seed=args.seed,
        output_dir=args.output_dir,
        save_artifacts=not args.no_save
    )


if __name__ == '__main__':
    main()
