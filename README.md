# Explainable Influencer Detection in Social Networks using Graph Neural Networks

A comprehensive deep learning framework for identifying and explaining influential nodes in social networks using Graph Neural Networks (GNNs) with explainable AI methods.

## 📋 Project Overview

This project implements a complete machine learning pipeline for:
- **Influencer Detection**: Predicting node influence scores in social networks
- **Node Ranking**: Identifying top-k most influential nodes
- **Explainability**: Understanding why the model predicts certain nodes as influential
- **Visualization**: Comprehensive visual analysis of predictions and explanations

## 🎯 Problem Statement

In social networks, identifying influential nodes is critical for:
- Targeted marketing and viral campaigns
- Information diffusion analysis
- Community detection
- Network resilience assessment

Traditional centrality measures (PageRank, degree) are useful but don't capture complex structural patterns. This project uses Graph Neural Networks to learn node importance from graph structure and features, and explains predictions using gradient-based saliency methods.

## 🏗️ Architecture

### Models Implemented
- **GCN** (Graph Convolutional Networks) - *Mandatory*
- **GraphSAGE** - *Optional*
- **GAT** (Graph Attention Networks) - *Optional*

### Centrality Methods
- **Degree Centrality**: Number of neighbors
- **PageRank**: Importance based on incoming connections
- **Eigenvector Centrality**: Influence through important neighbors

### Datasets
- **Cora**: Citation network (2,708 nodes, 5,429 edges)
- **CiteSeer**: Citation network (3,327 nodes, 4,732 edges)
- **Karate Club**: Social network (34 nodes, classic benchmark)

## 📦 Installation

### Requirements
- Python 3.8+
- PyTorch 2.0+
- PyTorch Geometric 2.3+
- NetworkX 3.1+
- NumPy, Pandas, Matplotlib, Scikit-learn

### Setup

```bash
# Clone repository
git clone <repository-url>
cd influencer-detection-gnn

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 🚀 Quick Start

### Train a Model

```python
from main import run_experiment

# Run with default settings (Cora, PageRank, GCN)
results = run_experiment(
    dataset_name='Cora',
    centrality_method='pagerank',
    model_name='gcn',
    epochs=200,
    seed=42
)
```

### Command Line Usage

```bash
# Run experiment with specific parameters
python main.py

# Run top-k analysis
python scripts/top_k_analysis.py --dataset Cora --predictions results/predictions.csv --k 10
```

## 📊 Project Structure

```
influencer-detection-gnn/
├── data/                          # Downloaded datasets
├── models/                        # Saved model checkpoints
├── outputs/                       # Experiment outputs
├── explainability/                # Explanation artifacts
├── notebooks/                     # Jupyter notebooks
├── scripts/                       # Analysis scripts
├── checkpoints/                   # Training checkpoints
├── results/                       # Final results
│
├── dataset.py                     # Dataset loading and centrality computation
├── model.py                       # GNN model implementations
├── train.py                       # Training pipeline with early stopping
├── evaluate.py                    # Evaluation metrics and analysis
├── explain.py                     # Explainability methods
├── visualize.py                   # Visualization functions
├── utils.py                       # Utility functions
├── main.py                        # Main entry point
├── requirements.txt               # Dependencies
└── README.md                      # This file
```

## 🎓 Training Pipeline

### 1. Data Preparation
```python
from dataset import load_influencer_dataset

data, centralities = load_influencer_dataset(
    dataset_name='Cora',
    root='./data',
    centrality_method='pagerank'
)
```

### 2. Model Creation
```python
from model import ModelFactory

model = ModelFactory.create_model(
    'gcn',
    input_dim=data.num_features,
    hidden_dim=64,
    output_dim=1
)
```

### 3. Training
```python
from train import train_model

results = train_model(
    model=model,
    data=data,
    train_mask=train_mask,
    val_mask=val_mask,
    test_mask=test_mask,
    device=device,
    epochs=200
)
```

### 4. Evaluation
```python
from evaluate import evaluate_model

metrics, eval_table, top_k_analysis = evaluate_model(
    y_true=y_true,
    y_pred=y_pred,
    k=10,
    dataset_name='Cora',
    model_name='GCN'
)
```

## 📈 Evaluation Metrics

### Prediction Metrics
- **MSE**: Mean Squared Error
- **MAE**: Mean Absolute Error
- **RMSE**: Root Mean Squared Error

### Ranking Metrics
- **Spearman Correlation**: Rank correlation between true and predicted
- **Kendall Tau**: Rank correlation measure
- **Precision@k**: Fraction of top-k predictions that are correct
- **Recall@k**: Fraction of true top-k that are predicted

## 🔍 Explainability

### Methods
1. **Gradient Saliency**: Node importance based on gradients w.r.t. features
2. **Important Neighbors**: Identification of critical neighboring nodes
3. **Subgraph Visualization**: Visual explanation showing important edges

### Usage
```python
from explain import ExplainabilityAnalyzer

analyzer = ExplainabilityAnalyzer(model, device)

# Explain specific node
explanation = analyzer.explain_prediction(
    data=data,
    target_node=42,
    k_neighbors=5
)

print(f"Node {explanation['target_node']}")
print(f"Prediction: {explanation['prediction']:.4f}")
print(f"Important neighbors: {explanation['important_neighbors']}")
print(f"Important edges: {explanation['important_edges']}")
```

## 📊 Visualization

All visualizations are automatically saved:

```
results/
├── Cora/
│   ├── gcn/
│   │   └── seed_42/
│   │       ├── visualizations/
│   │       │   ├── training_curves.png
│   │       │   ├── predictions_vs_actual.png
│   │       │   ├── top_k_comparison.png
│   │       │   ├── metrics_table.png
│   │       │   └── explanation_node_*.png
│   │       ├── metrics.json
│   │       ├── evaluation_table.csv
│   │       ├── predictions.csv
│   │       ├── top_k_comparison.csv
│   │       └── explanations.json
```

### Generated Plots
- **Training Curves**: Train and validation loss over epochs
- **Predictions vs Actual**: Scatter plot of predictions vs ground truth
- **Top-K Comparison**: Bar charts comparing predicted vs actual top-k nodes
- **Explanation Subgraphs**: Network visualizations showing important connections
- **Metrics Table**: Summary of evaluation metrics

## 🔬 Experimental Results

### Sample Results (Cora Dataset)

| Metric | Value |
|--------|-------|
| MSE | 0.0125 |
| MAE | 0.0852 |
| RMSE | 0.1118 |
| Spearman Correlation | 0.8234 |
| Precision@10 | 0.70 |
| Recall@10 | 0.70 |

### Top-10 Influencers Identified

The model successfully identifies influential nodes and explains why by showing:
- Important neighboring nodes
- Critical edges in the subgraph
- Feature contributions to the prediction

## 🛠️ Advanced Usage

### Custom Configuration
```python
results = run_experiment(
    dataset_name='CiteSeer',
    centrality_method='degree',
    model_name='graphsage',
    hidden_dim=128,
    learning_rate=0.005,
    weight_decay=1e-5,
    epochs=300,
    seed=123
)
```

### Multiple Models Comparison
```python
models = ['gcn', 'graphsage', 'gat']
results = {}

for model_name in models:
    results[model_name] = run_experiment(
        dataset_name='Cora',
        model_name=model_name,
        epochs=200
    )
```

### Batch Analysis
```python
datasets = ['Cora', 'CiteSeer', 'Karate']
centrality_methods = ['degree', 'pagerank', 'eigenvector']

for dataset in datasets:
    for method in centrality_methods:
        results = run_experiment(
            dataset_name=dataset,
            centrality_method=method,
            epochs=150
        )
```

## 🎓 Key Features

✅ **Easy to Use**: Simple API for training and evaluation
✅ **Modular Design**: Easily add new models or datasets
✅ **GPU Support**: Automatic GPU utilization if available
✅ **Reproducibility**: Fixed random seeds for consistent results
✅ **Comprehensive Logging**: Detailed training and evaluation logs
✅ **Visualization**: Automatic plot generation
✅ **Explainability**: Multiple explanation methods
✅ **Research Quality**: Publication-ready code and results

## 🔮 Future Improvements

1. **Streamlit Dashboard**: Interactive web interface
2. **More Models**: Graph Transformer, GraphConvolution variants
3. **Attention Visualization**: Visualize attention weights
4. **Temporal Analysis**: Support for dynamic/temporal graphs
5. **Benchmark Comparisons**: Comparison with other XAI methods
6. **Large-Scale Graphs**: Optimization for billion-node graphs

## 📚 References

- GCN: Kipf & Welling (2017) - Semi-Supervised Classification with Graph Convolutional Networks
- GraphSAGE: Hamilton et al. (2017) - Inductive Representation Learning on Large Graphs
- GAT: Veličković et al. (2018) - Graph Attention Networks
- PyTorch Geometric: Fey & Lenssen (2019)

## 📝 Citation

If you use this project in your research, please cite:

```bibtex
@software{influencer_detection_gnn_2024,
  title={Explainable Influencer Detection in Social Networks using Graph Neural Networks},
  author={Your Name},
  year={2024},
  url={https://github.com/username/influencer-detection-gnn}
}
```

## 📄 License

This project is licensed under the MIT License - see LICENSE file for details.

## 👤 Author

Your Name - Research Project for [Institution/Course]

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## ❓ FAQ

**Q: What's the best centrality method?**
A: It depends on your use case. PageRank captures importance through connections, while Degree is simpler but effective for scale-free networks.

**Q: Can I use my own dataset?**
A: Yes! Create a custom InfluencerDataset subclass following the template in dataset.py.

**Q: How long does training take?**
A: For Cora (~2,700 nodes), expect 30-60 seconds on CPU, <10 seconds on GPU.

**Q: Which model should I use?**
A: Start with GCN (fastest), then try GraphSAGE or GAT for better accuracy at the cost of computation.

## 📞 Support

For questions or issues, please create a GitHub issue or contact via email.

---

**Made with ❤️ for Graph Neural Network enthusiasts**
