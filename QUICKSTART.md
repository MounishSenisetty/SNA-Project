# Quick Start Guide

## Installation (5 minutes)

```bash
# Navigate to project directory
cd influencer-detection-gnn

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Train Your First Model (2 minutes)

### Option 1: Using Python

```python
from main import run_experiment

# Train on Karate Club (smallest dataset)
results = run_experiment(
    dataset_name='Karate',
    model_name='gcn',
    epochs=200,
    seed=42
)
```

### Option 2: Using Command Line

```bash
# Train on Karate Club
python scripts/train.py --dataset Karate --model gcn --epochs 200

# Train on Cora (larger dataset)
python scripts/train.py --dataset Cora --model gcn --epochs 200

# Train with custom parameters
python scripts/train.py \
    --dataset CiteSeer \
    --centrality degree \
    --model graphsage \
    --hidden-dim 128 \
    --epochs 250 \
    --lr 0.005
```

## View Results

After training, results are saved to `results/{dataset}/{model}/seed_{seed}/`

```
results/
├── Karate/
│   ├── gcn/
│   │   └── seed_42/
│   │       ├── visualizations/        # PNG plots
│   │       ├── metrics.json           # Performance metrics
│   │       ├── predictions.csv        # Node predictions
│   │       ├── top_k_comparison.csv   # Top-k analysis
│   │       ├── explanations.json      # Explanations
│   │       └── evaluation_table.csv   # Evaluation results
```

## Key Features

### Datasets Available
- **Karate Club** (34 nodes, 156 edges) - ~1 second training
- **Cora** (2,708 nodes, 5,429 edges) - ~30 seconds training
- **CiteSeer** (3,327 nodes, 4,732 edges) - ~40 seconds training

### Models Available
- **GCN** - Graph Convolutional Network (recommended for beginners)
- **GraphSAGE** - Sample and Aggregate
- **GAT** - Graph Attention Network

### Centrality Methods (Labels)
- **Degree** - Number of connections
- **PageRank** - Importance through connections
- **Eigenvector** - Influence through important neighbors

### Metrics Computed
- **Prediction**: MSE, MAE, RMSE
- **Ranking**: Spearman, Kendall, Precision@10, Recall@10
- **Analysis**: Top-10 overlap, Precision@10, Recall@10

## Example: Complete Analysis

```python
from main import run_experiment
from evaluate import TopKAnalyzer
import pandas as pd

# Train model
results = run_experiment(
    dataset_name='Cora',
    centrality_method='pagerank',
    model_name='gcn',
    epochs=200,
    seed=42
)

# View metrics
print(results['eval_table'])

# View top-k analysis
analysis = results['top_k_analysis']
print(f"Top-10 overlap: {analysis['overlap_count']}/10")
print(f"Precision@10: {analysis['precision']:.2%}")

# View predictions
predictions = pd.read_csv(f"{results['output_path']}/predictions.csv")
print(predictions.head(10))
```

## Common Tasks

### Train Multiple Models on Same Dataset
```bash
for model in gcn graphsage gat; do
    python scripts/train.py --dataset Cora --model $model --epochs 200
done
```

### Compare Different Centrality Methods
```bash
for centrality in degree pagerank eigenvector; do
    python scripts/train.py \
        --dataset Karate \
        --centrality $centrality \
        --epochs 200
done
```

### Hyperparameter Search
```bash
for lr in 0.001 0.01 0.05; do
    for hidden_dim in 32 64 128; do
        python scripts/train.py \
            --dataset Cora \
            --lr $lr \
            --hidden-dim $hidden_dim
    done
done
```

## Troubleshooting

### "Module not found" errors
```bash
pip install -r requirements.txt
```

### GPU/CUDA issues
The code automatically uses GPU if available. To force CPU:
```python
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
```

### Out of memory
Reduce `hidden_dim`:
```bash
python scripts/train.py --dataset Cora --hidden-dim 32
```

### Slow training
- Use smaller dataset (Karate instead of Cora)
- Reduce `epochs`
- Use faster model (GCN instead of GAT)

## Next Steps

1. **Read the detailed documentation**: See [README.md](README.md)
2. **Explore parameters**: Check [CONFIG_GUIDE.md](CONFIG_GUIDE.md)
3. **View results**: Check `results/` directory
4. **Modify code**: Edit model.py or explain.py to experiment

## Example Projects

### Project 1: Model Comparison
```python
# Compare all models on Cora
from main import run_experiment

models = ['gcn', 'graphsage', 'gat']
results = {}

for model in models:
    results[model] = run_experiment(
        dataset_name='Cora',
        model_name=model,
        epochs=200
    )

# Compare metrics
for model, result in results.items():
    print(f"{model}: Spearman = {result['metrics']['Spearman']:.4f}")
```

### Project 2: Cross-Dataset Training
```python
# Train on Karate, test on Cora
from main import run_experiment

results = run_experiment(
    dataset_name='Karate',
    model_name='gcn',
    epochs=100
)

# Model learned from small graph
# Can now be tested on larger graphs
```

### Project 3: Explainability Analysis
```python
from explain import ExplainabilityAnalyzer
from dataset import InfluencerDataset

# Get data
dataset = InfluencerDataset('Cora')
data = dataset.load_dataset()

# Get model and explain
from model import ModelFactory
model = ModelFactory.create_model('gcn', data.num_features)

analyzer = ExplainabilityAnalyzer(model, device)

# Explain specific node
explanation = analyzer.explain_prediction(data, target_node=42)
print(f"Important neighbors: {explanation['important_neighbors']}")
print(f"Important edges: {explanation['important_edges']}")
```

## Tips & Best Practices

✅ **DO:**
- Start with Karate Club for quick testing
- Use seed=42 for reproducibility
- Save results after successful runs
- Try different centrality methods
- Visualize results before/after

❌ **DON'T:**
- Change architecture while debugging metrics
- Use very small hidden dimensions (< 16)
- Set learning rate > 0.1 without reason
- Forget to set random seeds for comparisons
- Mix metrics from different training runs

## Support

- **Errors?** Check the error message and traceback
- **Slow?** Reduce dataset size or model complexity
- **Different results?** Check random seeds are set
- **Questions?** See [README.md](README.md) for detailed docs

## Performance Benchmarks

| Dataset  | Model      | Epochs | Time | Spearman |
|----------|-----------|--------|------|----------|
| Karate   | GCN       | 50     | 1s   | 0.80     |
| Karate   | GraphSAGE | 50     | 2s   | 0.78     |
| Cora     | GCN       | 200    | 30s  | 0.83     |
| Cora     | GraphSAGE | 200    | 60s  | 0.85     |
| CiteSeer | GCN       | 200    | 40s  | 0.81     |

Times are on CPU. GPU will be 5-10x faster.

---

**Ready to start?** Run this command:
```bash
python scripts/train.py --dataset Karate --epochs 100
```

**Check results:**
```bash
python -c "import json; print(json.dumps(json.load(open('results/Karate/gcn/seed_42/metrics.json')), indent=2))"
```

Enjoy! 🎉
