# Example Configuration for Influencer Detection

## Quick Start Example
This configuration trains a GCN model on the Cora dataset to predict PageRank centrality.

```python
from main import run_experiment

results = run_experiment(
    dataset_name='Cora',
    centrality_method='pagerank',
    model_name='gcn',
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    hidden_dim=64,
    learning_rate=0.01,
    weight_decay=5e-4,
    epochs=200,
    seed=42,
    output_dir='./results',
    save_artifacts=True
)
```

## Command Line Training

### Train on Cora with GCN
```bash
python scripts/train.py \
    --dataset Cora \
    --centrality pagerank \
    --model gcn \
    --epochs 200 \
    --lr 0.01
```

### Train on CiteSeer with GraphSAGE
```bash
python scripts/train.py \
    --dataset CiteSeer \
    --centrality degree \
    --model graphsage \
    --hidden-dim 128 \
    --epochs 250 \
    --lr 0.005
```

### Train on Karate Club with GAT
```bash
python scripts/train.py \
    --dataset Karate \
    --centrality eigenvector \
    --model gat \
    --hidden-dim 32 \
    --epochs 150
```

## Parameter Tuning Guide

### Learning Rate
- **0.01**: Good default for most cases
- **0.001 - 0.005**: For larger datasets or when overfitting
- **0.05 - 0.1**: For faster convergence (may be unstable)

### Hidden Dimension
- **32**: Small graphs (Karate Club)
- **64**: Medium graphs (Cora, CiteSeer)
- **128+**: Large graphs or complex patterns

### Epochs
- **100**: Quick experimentation
- **200**: Recommended for most cases
- **300+**: For datasets that need more training

### Weight Decay
- **5e-4**: Default, works well
- **1e-3**: Stronger regularization
- **1e-5 - 1e-4**: Lighter regularization

### Train/Val/Test Split
- **0.7 / 0.15 / 0.15**: Standard (recommended)
- **0.8 / 0.1 / 0.1**: For larger datasets
- **0.6 / 0.2 / 0.2**: More validation data

## Expected Results

### Cora Dataset
- Model: GCN
- Centrality: PageRank
- Expected MSE: ~0.01-0.02
- Expected Spearman: ~0.80-0.85
- Expected Precision@10: ~0.70-0.80

### CiteSeer Dataset
- Model: GraphSAGE
- Centrality: Degree
- Expected MSE: ~0.015-0.025
- Expected Spearman: ~0.75-0.82
- Expected Precision@10: ~0.65-0.75

### Karate Club
- Model: GAT
- Centrality: Eigenvector
- Expected MSE: ~0.005-0.015
- Expected Spearman: ~0.85-0.95
- Expected Precision@10: ~0.80-1.00

## Output Structure

After training, the results are organized as:

```
results/
├── {dataset}/
│   ├── {model}/
│   │   └── seed_{seed}/
│   │       ├── visualizations/
│   │       │   ├── training_curves.png
│   │       │   ├── predictions_vs_actual.png
│   │       │   ├── top_k_comparison.png
│   │       │   ├── metrics_table.png
│   │       │   └── explanation_node_*.png
│   │       ├── checkpoints/
│   │       │   └── best_model.pt
│   │       ├── metrics.json
│   │       ├── evaluation_table.csv
│   │       ├── predictions.csv
│   │       ├── top_k_comparison.csv
│   │       └── explanations.json
```

## Common Configurations

### For Research (High Accuracy)
```bash
python scripts/train.py \
    --dataset Cora \
    --model gcn \
    --epochs 300 \
    --hidden-dim 128 \
    --weight-decay 1e-4 \
    --lr 0.005
```

### For Quick Experiment
```bash
python scripts/train.py \
    --dataset Karate \
    --model gcn \
    --epochs 100 \
    --hidden-dim 32
```

### For Comparison Study
```bash
for model in gcn graphsage gat; do
    python scripts/train.py \
        --dataset Cora \
        --model $model \
        --epochs 200 \
        --seed 42
done
```

### For Robustness Testing (Multiple Seeds)
```bash
for seed in 42 123 456 789 999; do
    python scripts/train.py \
        --dataset CiteSeer \
        --model gcn \
        --epochs 200 \
        --seed $seed
done
```

## Hyperparameter Ranges

### Recommended Ranges for Grid Search

```python
param_grid = {
    'hidden_dim': [32, 64, 128],
    'learning_rate': [0.001, 0.01, 0.05],
    'weight_decay': [1e-5, 5e-4, 1e-3],
    'dropout': [0.0, 0.3, 0.5]
}
```

## Troubleshooting

### High Loss
- Decrease learning rate
- Reduce hidden dimension
- Increase weight decay

### Overfitting
- Increase weight decay
- Increase dropout
- Use earlier stopping

### GPU Memory Issues
- Reduce hidden dimension
- Use gradient accumulation
- Try smaller batch sizes (if applicable)

### Slow Training
- Use GPU (CUDA)
- Reduce hidden dimension
- Use faster model (GCN vs GAT)

## Multi-GPU Training (Future)

For future multi-GPU support:
```python
# This will be implemented
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
model = nn.DataParallel(model)
```

## Next Steps

1. **Baseline Comparison**: Compare with non-learning baselines
2. **Cross-Dataset**: Train on one dataset, test on another
3. **Centrality Fusion**: Combine multiple centrality methods
4. **Temporal Analysis**: Apply to dynamic graphs
5. **Attention Visualization**: Visualize attention weights (GAT)

## Additional Resources

- See README.md for detailed documentation
- Check notebooks/ for example Jupyter notebooks
- Review scripts/ for analysis utilities
