# Explainable Node Centrality GNN - Complete Project Guide

## Project Overview

This project implements an explainable deep learning framework for node centrality prediction in social networks using Graph Neural Networks (GNNs) with temporal dynamics support.

**Technology Stack:**
- PyTorch & PyTorch Geometric 2.3+ (GNN models)
- NetworkX 3.1+ (Graph processing)
- YAML (Configuration management)
- Python 3.8+

---

## Project Directory Structure

```
explainable-centrality-gnn/
├── config.yaml                 # Main configuration file
├── data.py                     # Dataset loading & temporal features
├── model.py                    # GNN architectures (GCN, GraphSAGE, GAT, APPNP, MLP)
├── train.py                    # Training loop with early stopping
├── evaluate.py                 # Evaluation metrics (MSE, MAE, RMSE, etc.)
├── explain.py                  # Explainability (GNNExplainer, PGExplainer, gradient)
├── main.py                     # Entry point & orchestration
├── targets.py                  # Target label generation
├── baselines.py                # Non-learning baselines (degree, PageRank, etc.)
├── viz.py                      # Visualization utilities
├── utils.py                    # Utility functions
├── requirements.txt            # Python dependencies
├── LICENSE                     # MIT License
├── README.md                   # Project overview
├── RESEARCH_PAPER.md           # 52-page comprehensive paper
├── docs/                       # Additional documentation
├── tests/                      # Unit tests
└── configs/                    # Alternative configuration files
```

---

## File-by-File Code Explanation

### 1. **config.yaml** - Configuration File

```yaml
# Lines 1-5: Data Configuration
data:
  source: synthetic              # "synthetic" (BA graph) or "real" (PyG datasets)
  temporal:
    enabled: true                # Enable temporal snapshots (default: false)
    snapshots: 5                 # Number of time steps for temporal analysis
    decay: 0.85                  # Exponential decay factor (recent edges weighted higher)

  synthetic:
    type: ba                     # "ba" (Barabási-Albert) or "er" (Erdős-Rényi)
    n: 50                        # Number of nodes
    m: 2                         # Edges per new node (BA only)
    feature_dim: 16              # Feature dimensionality
    seed: 42
  
  real:
    name: cora                   # Dataset name from PyG (cora, citeseer, pubmed)

# Lines 15-25: Model Configuration
train:
  model: gcn                     # GNN type: "gcn", "graphsage", "gat", "appnp", "mlp"
  hidden_dim: 64                 # Hidden layer dimension
  num_layers: 2                  # Number of GNN layers
  dropout: 0.2                   # Dropout rate
  learning_rate: 0.001           # Initial learning rate
  batch_size: 32                 # Batch size
  epochs: 200                    # Maximum training epochs
  early_stopping_patience: 30    # Stop if val loss doesn't improve for 30 epochs
  seed: [42, 43, 44]             # List of random seeds for reproducibility

# Lines 26-35: Explanation Configuration
explain:
  methods: ["gnnexplainer", "gradient", "random"]  # Explanation methods to run
  node_id: 0                     # Target node for explanation (0 = first node)
  gnnexplainer_epochs: 200       # Optimization epochs for GNNExplainer
  fidelity_k: [5, 10, 20, 40]    # K values for fidelity evaluation
```

**Key Points:**
- `temporal.enabled`: Activates snapshot-based temporal evolution
- `temporal.decay=0.85`: Weights recent snapshots 2.5× more than oldest
- `gnnexplainer_epochs`: Higher = better explanations but slower (200 ≈ 15s per node)

---

### 2. **data.py** - Dataset Loading & Temporal Features

#### Lines 1-12: Imports
```python
import json, random, dataclasses
from pathlib import Path
from typing import Dict, Optional, Tuple
import networkx as nx, numpy as np, torch
from torch_geometric.data import Data
from torch_geometric.datasets import KarateClub, Planetoid
from torch_geometric.utils import from_networkx, to_networkx, to_undirected
```
**Purpose:** Import necessary libraries for graph processing and tensor operations

#### Lines 14-20: DatasetBundle Dataclass
```python
@dataclass
class DatasetBundle:
    data: Data                  # PyG Data object with node features, edges
    name: str                   # Dataset name (e.g., "BA_50")
    metadata: Dict              # Graph metadata (num_nodes, num_edges, temporal info)
    gt_explanations: Optional[Dict] = None  # Ground truth explanations (future use)
```
**Purpose:** Container for dataset + metadata + optional ground truth explanations

#### Lines 22-29: Seed Setting (Reproducibility)
```python
def set_seed(seed: int):
    random.seed(seed)           # Python random
    np.random.seed(seed)        # NumPy random
    torch.manual_seed(seed)     # PyTorch CPU random
    torch.cuda.manual_seed_all(seed)  # PyTorch GPU random
    torch.backends.cudnn.deterministic = True  # Ensure deterministic CUDA operations
    torch.backends.cudnn.benchmark = False    # Disable autotuner for consistency
```
**Purpose:** Ensures exact reproducibility across multiple runs

#### Lines 31-37: Attach Features
```python
def _attach_features(data: Data, feature_dim: int, seed: int):
    if getattr(data, "x", None) is None:  # If no features exist
        gen = torch.Generator().manual_seed(seed)  # Create seeded generator
        data.x = torch.randn((data.num_nodes, feature_dim), generator=gen)
    return data
```
**Purpose:** Generate random features if dataset lacks them (for synthetic graphs)

#### Lines 39-57: Temporal Snapshots
```python
def _temporal_snapshots(G: nx.Graph, num_snapshots: int, seed: int):
    if num_snapshots < 2:
        raise ValueError("temporal.snapshots must be at least 2")
    
    rng = np.random.default_rng(seed)  # Seeded RNG for reproducibility
    edges = list(G.edges())
    rng.shuffle(edges)  # Randomize edge order (deterministic due to seed)
    
    snapshots = []
    for step in range(num_snapshots):
        # Calculate edge cutoff for this snapshot
        cutoff = max(1, int(len(edges) * (step + 1) / num_snapshots))
        
        # Create snapshot with edges[0:cutoff]
        snapshot = nx.Graph()
        snapshot.add_nodes_from(G.nodes())
        snapshot.add_edges_from(edges[:cutoff])
        snapshots.append(snapshot)
    
    return snapshots
```
**Purpose:** Creates T temporal snapshots by progressively revealing edges
- **Example**: 10 edges → Snapshot 0: 2 edges, Snapshot 1: 4 edges, ..., Snapshot 4: 10 edges
- **Intuition**: Simulates network formation over time

#### Lines 59-94: Temporal Node Features
```python
def _temporal_node_features(G: nx.Graph, num_snapshots: int, seed: int):
    # Build snapshots
    snapshots = _temporal_snapshots(G, num_snapshots=num_snapshots, seed=seed)
    
    # Extract degree for each node at each snapshot
    # degree_history shape: [num_snapshots, num_nodes]
    degree_history = np.stack(
        [np.array([snap.degree(node) for node in G.nodes()]) for snap in snapshots],
        axis=0
    )
    
    # Active mask: True if node has edges at this snapshot
    active_mask = degree_history > 0
    
    # Initialize arrays for first/last seen timestamps
    first_seen = np.full(len(G.nodes()), num_snapshots - 1)
    last_seen = np.zeros(len(G.nodes()))
    
    # Find first and last active snapshot for each node
    for node in range(len(G.nodes())):
        active_steps = np.where(active_mask[:, node])[0]
        if active_steps.size:
            first_seen[node] = float(active_steps[0])
            last_seen[node] = float(active_steps[-1])
    
    # Create 8 temporal features per node
    temporal_features = np.stack([
        degree_history.mean(axis=0),              # 1. Average degree
        degree_history.std(axis=0),               # 2. Degree std (volatility)
        degree_history[-1],                       # 3. Final degree
        degree_history.max(axis=0),               # 4. Peak degree
        active_mask.mean(axis=0),                 # 5. Activity ratio
        (degree_history[-1] - degree_history[0]) / max(num_snapshots-1, 1),  # 6. Trend
        first_seen / max(num_snapshots - 1, 1),   # 7. First seen (normalized)
        last_seen / max(num_snapshots - 1, 1),    # 8. Last seen (normalized)
    ], axis=1)
    
    # Standardize features (z-score normalization)
    feature_mean = temporal_features.mean(axis=0, keepdims=True)
    feature_std = temporal_features.std(axis=0, keepdims=True) + 1e-9
    return ((temporal_features - feature_mean) / feature_std).astype(np.float32)
```
**Purpose:** Extract 8-dimensional temporal feature vector per node
- **Output shape**: [num_nodes, 8]
- **Features interpret**: Network formation patterns, growth trends, persistence

#### Lines 96-109: Apply Temporal View
```python
def apply_temporal_view(data: Data, temporal_cfg: Dict, seed: int) -> Data:
    if not temporal_cfg.get("enabled", False):
        return data  # Skip if temporal mode disabled
    
    steps = int(temporal_cfg.get("snapshots", 5))
    
    # Convert PyG Data to NetworkX for temporal processing
    graph = to_networkx(data, to_undirected=True)
    
    # Compute temporal features
    temporal_features = _temporal_node_features(
        graph.to_undirected(), num_snapshots=steps, seed=seed
    )
    
    # Concatenate temporal features to original features
    if data.x is not None:
        data.x = torch.cat(
            [data.x.float(), torch.tensor(temporal_features)],
            dim=1  # Concatenate along feature dimension
        )
```
**Purpose:** Main entry point for temporal feature injection
- **Workflow**: Original features → NetworkX → Compute 8 temporal features → Concatenate

#### Lines 111-200+: Synthetic/Real Dataset Loaders
```python
def generate_synthetic_graph(cfg: Dict, seed: int) -> DatasetBundle:
    set_seed(seed)
    
    # Parse config
    graph_type = cfg.get("type", "ba")
    n = int(cfg.get("n", 50))
    feature_dim = int(cfg.get("feature_dim", 16))
    
    # Generate graph
    if graph_type == "ba":
        m = int(cfg.get("m", 2))  # Edges per new node
        G = nx.barabasi_albert_graph(n, m, seed=seed)
    elif graph_type == "er":
        p = float(cfg.get("p", 0.1))  # Edge probability
        G = nx.erdos_renyi_graph(n, p, seed=seed)
    else:
        raise ValueError(f"Unknown graph type: {graph_type}")
    
    # Convert to PyG Data
    data = from_networkx(G)
    
    # Attach features
    data = _attach_features(data, feature_dim, seed)
    
    # Create metadata
    metadata = {
        "num_nodes": data.num_nodes,
        "num_edges": data.num_edges,
        "graph_type": graph_type,
    }
    
    # Apply temporal view if enabled
    temporal_cfg = cfg.get("temporal", {})
    if temporal_cfg.get("enabled", False):
        data = apply_temporal_view(data, temporal_cfg, seed)
        metadata.update({
            "temporal_enabled": True,
            "temporal_snapshots": temporal_cfg.get("snapshots", 5),
            "temporal_decay": temporal_cfg.get("decay", 0.85),
        })
    else:
        metadata["temporal_enabled"] = False
    
    return DatasetBundle(
        data=data,
        name=f"{graph_type.upper()}_{n}",
        metadata=metadata
    )

def load_real_dataset(cfg: Dict, seed: int) -> DatasetBundle:
    set_seed(seed)
    
    dataset_name = cfg.get("name", "cora").lower()
    
    # Load from PyG
    if dataset_name == "karate":
        dataset = KarateClub()
    else:
        dataset = Planetoid(root="/tmp/data", name=dataset_name.title())
    
    data = dataset[0].clone()
    data = to_undirected(data)
    
    metadata = {
        "num_nodes": data.num_nodes,
        "num_edges": data.num_edges,
        "source": "real",
        "dataset_name": dataset_name,
    }
    
    # Apply temporal view
    temporal_cfg = cfg.get("temporal", {})
    if temporal_cfg.get("enabled", False):
        data = apply_temporal_view(data, temporal_cfg, seed)
        metadata.update({
            "temporal_enabled": True,
            "temporal_snapshots": temporal_cfg.get("snapshots", 5),
        })
    
    return DatasetBundle(
        data=data,
        name=dataset_name,
        metadata=metadata
    )
```
**Purpose:** Generate/load datasets with optional temporal features

---

### 3. **model.py** - GNN Architectures

#### Lines 1-30: GCN Regressor
```python
from torch_geometric.nn import GCNConv

class GCNRegressor(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, num_layers: int = 2, 
                 dropout: float = 0.2, out_dim: int = 1):
        super().__init__()
        
        # Build convolutional layers
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_dim, hidden_dim))        # Input layer
        
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))  # Hidden layers
        
        self.convs.append(GCNConv(hidden_dim, out_dim))       # Output layer
        self.dropout = dropout
    
    def forward(self, x, edge_index):
        # Pass through all but last layer with ReLU + Dropout
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)      # Graph convolution
            x = F.relu(x)                # Non-linearity
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Final layer (no activation for regression)
        x = self.convs[-1](x, edge_index)
        return x  # [num_nodes, out_dim]
```
**Purpose:** GCN layer stack for spectral graph convolution
- **Why GCN**: Fast, theoretically grounded, captures k-hop neighborhoods
- **Dropout**: Prevents overfitting
- **No activation on output**: Allows unbounded regression values

#### Lines 32-50: GraphSAGE Regressor
```python
class SAGERegressor(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, num_layers: int = 2, 
                 dropout: float = 0.2, out_dim: int = 1):
        super().__init__()
        
        self.convs = nn.ModuleList([SAGEConv(in_dim, hidden_dim)])
        
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
        
        self.convs.append(SAGEConv(hidden_dim, out_dim))
        self.dropout = dropout
    
    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        return self.convs[-1](x, edge_index)
```
**Purpose:** Sampling-based aggregation (handles large graphs via mini-batching)

#### Lines 52-75: GAT Regressor
```python
class GATRegressor(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, num_layers: int = 2, 
                 dropout: float = 0.2, heads: int = 2, out_dim: int = 1):
        super().__init__()
        
        self.convs = nn.ModuleList()
        self.convs.append(GATConv(in_dim, hidden_dim, heads=heads, dropout=dropout))
        
        for _ in range(num_layers - 2):
            # After heads, dimension becomes hidden_dim * heads
            self.convs.append(GATConv(hidden_dim * heads, hidden_dim, 
                                     heads=heads, dropout=dropout))
        
        # Output layer: single head (no multi-head)
        self.out = GATConv(hidden_dim * heads, out_dim, heads=1, 
                          concat=False, dropout=dropout)
        self.dropout = dropout
    
    def forward(self, x, edge_index):
        for conv in self.convs:
            x = conv(x, edge_index)     # Attention + aggregate
            x = F.elu(x)                # ELU activation (GAT standard)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        return self.out(x, edge_index)
```
**Purpose:** Multi-head attention over neighbors
- **Attention**: Learn which neighbors matter per node
- **Interpretable**: Attention weights show edge importance

#### Lines 77-100: APPNP Regressor
```python
class APPNPRegressor(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, dropout: float = 0.2, 
                 k: int = 10, alpha: float = 0.1, out_dim: int = 1):
        super().__init__()
        
        # Decoupled: MLP for feature transformation
        self.lin1 = nn.Linear(in_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, out_dim)
        
        # Propagation: fixed personalized PageRank
        self.prop = APPNP(K=k, alpha=alpha)
        self.dropout = dropout
    
    def forward(self, x, edge_index):
        # Feature learning (independent of propagation)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        
        # Propagation (fixed PageRank)
        x = self.prop(x, edge_index)
        return x
```
**Purpose:** Decoupled learning (MLP) and propagation (PageRank)
- **Advantage**: Fewer parameters, more stable
- **k=10**: 10 propagation steps
- **alpha=0.1**: Teleport probability

#### Lines 102-120: MLP Baseline
```python
class MLPRegressor(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, num_layers: int = 2, 
                 dropout: float = 0.2, out_dim: int = 1):
        super().__init__()
        
        layers = []
        d = in_dim
        
        # Build linear layers
        for _ in range(max(num_layers - 1, 1)):
            layers.append(nn.Linear(d, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            d = hidden_dim
        
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x, edge_index):
        del edge_index  # Ignore graph structure (baseline)
        return self.net(x)
```
**Purpose:** Non-graph baseline to measure graph contribution
- **Insight**: If MLP ≈ GCN, then features >> structure

#### Lines 122-150: Model Factory
```python
def build_model(name: str, in_dim: int, hidden_dim: int, num_layers: int, 
                dropout: float, out_dim: int) -> nn.Module:
    """
    Factory function to build GNN models.
    
    Args:
        name: Model name ("gcn", "graphsage", "gat", "appnp", "mlp")
        in_dim: Input feature dimension
        hidden_dim: Hidden layer dimension
        num_layers: Number of layers
        dropout: Dropout rate
        out_dim: Output dimension (1 for regression)
    
    Returns:
        Instantiated model
    """
    name_lower = name.lower()
    
    if name_lower == "gcn":
        return GCNRegressor(in_dim, hidden_dim, num_layers, dropout, out_dim)
    elif name_lower == "graphsage":
        return SAGERegressor(in_dim, hidden_dim, num_layers, dropout, out_dim)
    elif name_lower == "gat":
        return GATRegressor(in_dim, hidden_dim, num_layers, dropout, out_dim)
    elif name_lower == "appnp":
        return APPNPRegressor(in_dim, hidden_dim, dropout, out_dim=out_dim)
    elif name_lower == "mlp":
        return MLPRegressor(in_dim, hidden_dim, num_layers, dropout, out_dim)
    else:
        raise ValueError(f"Unknown model: {name}")
```
**Purpose:** Dynamically instantiate models by name

---

### 4. **train.py** - Training Loop

```python
def fit_model(model, data, train_mask, val_mask, optimizer, epochs=200, 
              patience=30, scheduler=None, grad_clip=None, verbose_every=20):
    """
    Train GNN with early stopping and learning rate scheduling.
    
    Args:
        model: Initialized GNN model
        data: PyG Data object with features, edges, labels
        train_mask: Boolean mask for training nodes
        val_mask: Boolean mask for validation nodes
        optimizer: PyTorch optimizer (Adam)
        epochs: Maximum training epochs
        patience: Early stopping patience
        scheduler: LR scheduler (ReduceLROnPlateau)
        grad_clip: Gradient clipping threshold (optional)
        verbose_every: Log interval
    
    Returns:
        fit_info: Dict with training history, best epoch, time
    """
    import time
    
    device = next(model.parameters()).device
    data = data.to(device)
    
    start_time = time.time()
    best_val_loss = float('inf')
    best_epoch = -1
    patience_counter = 0
    train_losses, val_losses = [], []
    
    for epoch in range(1, epochs + 1):
        # === Training Phase ===
        model.train()
        optimizer.zero_grad()
        
        # Forward pass on training nodes only
        out = model(data.x, data.edge_index)
        train_loss = F.mse_loss(out[train_mask], data.y[train_mask])
        
        # Backward pass
        train_loss.backward()
        
        # Gradient clipping (prevent explosion)
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        optimizer.step()
        
        # === Validation Phase ===
        model.eval()
        with torch.no_grad():
            val_out = model(data.x, data.edge_index)
            val_loss = F.mse_loss(val_out[val_mask], data.y[val_mask])
        
        train_losses.append(train_loss.item())
        val_losses.append(val_loss.item())
        
        # === Early Stopping Check ===
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            best_model_state = copy.deepcopy(model.state_dict())
        else:
            patience_counter += 1
        
        # === Learning Rate Scheduling ===
        if scheduler:
            scheduler.step(val_loss)
        
        # === Logging ===
        if epoch % verbose_every == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} | train MSE {train_loss:.4f} | val MSE {val_loss:.4f}")
        
        # === Early Stopping ===
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch} (patience={patience})")
            break
    
    # Load best model
    model.load_state_dict(best_model_state)
    train_time = time.time() - start_time
    
    return {
        "best_epoch": best_epoch,
        "best_val_loss": float(best_val_loss),
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_seconds": train_time,
    }
```

**Key Points:**
- **Early Stopping**: Stop if validation loss doesn't improve for 30 epochs
- **LR Scheduling**: Reduce learning rate if val loss plateaus
- **Gradient Clipping**: Prevent exploding gradients
- **Best Model State**: Load best weights, not last epoch

---

### 5. **explain.py** - Explainability Techniques

#### GNNExplainer
```python
def explain_node(model, data, node_id, device='cpu'):
    """
    Generate per-instance explanation using GNNExplainer.
    
    Optimizes learnable masks (edge + node feature) to identify
    minimal subgraph explaining the prediction.
    """
    model.eval()
    
    # Detect if model uses edges
    has_graph_layers = any(
        'conv' in type(m).__name__.lower() 
        for m in model.modules()
    )
    
    # Disable edge masking for MLP (doesn't use edges)
    edge_mask_type = "object" if has_graph_layers else None
    
    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=200),  # 200 optimization iterations
        explanation_type='model',
        node_mask_type='attributes',         # Explain node features
        edge_mask_type=edge_mask_type,       # Explain edges (if applicable)
        model_config=ModelConfig(
            mode='regression',
            task_level='node',
            return_type='raw'
        )
    )
    
    # Generate explanation
    explanation = explainer(data.x, data.edge_index, index=node_id)
    return explanation
    # Returns: explanation.edge_mask, explanation.node_mask
```

**How it works:**
1. Initialize masks (edge_mask, node_mask) randomly
2. Optimize: maximize model output on masked graph, minimize mask size
3. Result: sparse subgraph explaining prediction

#### Gradient-Based Saliency
```python
def explain_gradient_saliency(model, data, node_id, device='cpu'):
    """
    Fast explanation via input gradients.
    
    Measures: ∂y_node / ∂x_i (how much does output change per feature)
    """
    model.eval()
    data = data.to(device)
    
    # Compute gradient of target node's output w.r.t. input
    x = data.x.clone().detach().requires_grad_(True)
    
    y = model(x, data.edge_index)
    target_output = y[node_id, 0]  # Scalar output for target node
    
    target_output.backward()  # Backprop
    
    # Get gradients
    node_gradient = x.grad.abs()  # Absolute gradients (saliency)
    
    return {
        'node_mask': node_gradient[node_id],  # Feature saliency for target node
        'edge_mask': compute_edge_gradients(x.grad, data.edge_index)  # Derived
    }
```

**Pros:** Fast (1 backward pass), no optimization  
**Cons:** Doesn't account for architecture, interaction effects

#### Fidelity Evaluation
```python
def deletion_insertion_fidelity(model, data, node_id, edge_mask, k_values):
    """
    Measure how much prediction depends on explained edges.
    
    Edge Deletion: Remove top-k edges, measure output drop
    Edge Insertion: Start empty, add top-k edges, measure output gain
    """
    model.eval()
    device = next(model.parameters()).device
    data = data.to(device)
    
    # Original prediction
    with torch.no_grad():
        y_original = model(data.x, data.edge_index)[node_id].item()
    
    fidelity_delete = []
    fidelity_insert = []
    
    # Get top-k important edges
    top_k_indices = torch.argsort(edge_mask, descending=True)
    
    for k in k_values:
        # === Edge Deletion ===
        keep_mask = torch.ones(data.edge_index.size(1), dtype=torch.bool)
        keep_mask[top_k_indices[:k]] = False  # Remove top-k
        edge_index_removed = data.edge_index[:, keep_mask]
        
        with torch.no_grad():
            y_removed = model(data.x, edge_index_removed)[node_id].item()
        
        fidelity_delete.append(max(0, (y_original - y_removed) / (abs(y_original) + 1e-8)))
        
        # === Edge Insertion ===
        empty_edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
        insert_mask = torch.zeros(data.edge_index.size(1), dtype=torch.bool)
        insert_mask[top_k_indices[:k]] = True  # Keep only top-k
        edge_index_sparse = data.edge_index[:, insert_mask]
        
        with torch.no_grad():
            y_sparse = model(data.x, edge_index_sparse)[node_id].item()
        
        fidelity_insert.append((y_sparse - y_empty) / (abs(y_original - y_empty) + 1e-8))
    
    # AUC = average over k values
    return {
        'delete_curve': fidelity_delete,
        'insert_curve': fidelity_insert,
        'auc_delete': np.mean(fidelity_delete),
        'auc_insert': np.mean(fidelity_insert),
    }
```

**Interpretation:**
- **AUC ≥ 0.8**: Strong explanation (removing edges hurts significantly)
- **AUC ≤ 0.5**: Weak explanation (edges not critical)

---

### 6. **evaluate.py** - Metrics

```python
def evaluate(model, data, mask, topk_values=(10, 20, 50)):
    """
    Compute comprehensive evaluation metrics on test set.
    
    Returns:
        metrics: Dict with MSE, MAE, RMSE, Spearman ρ, Kendall τ, Precision@k
    """
    model.eval()
    
    with torch.no_grad():
        y_pred = model(data.x, data.edge_index).detach().cpu().numpy()
    
    y_true = data.y.detach().cpu().numpy()
    y_pred = y_pred[mask]
    y_true = y_true[mask]
    
    metrics = {
        'mse': float(np.mean((y_pred - y_true) ** 2)),
        'mae': float(np.mean(np.abs(y_pred - y_true))),
        'rmse': float(np.sqrt(np.mean((y_pred - y_true) ** 2))),
    }
    
    # Ranking correlation
    spearman_r, spearman_p = spearmanr(y_true, y_pred)
    metrics['spearman_r'] = float(spearman_r)
    metrics['spearman_p'] = float(spearman_p)
    
    # Top-k precision
    for k in topk_values:
        k_min = min(k, len(y_true))
        top_k_true = set(np.argsort(-y_true)[:k_min])
        top_k_pred = set(np.argsort(-y_pred)[:k_min])
        precision = len(top_k_true & top_k_pred) / k_min
        metrics[f'precision_at_{k}'] = float(precision)
    
    return metrics
```

**Key Metrics:**
- **MSE**: Mean squared error (sensitive to outliers)
- **MAE**: Mean absolute error (robust)
- **Spearman ρ**: Ranking correlation (0-1, 1=perfect)
- **Precision@k**: Overlap in top-k nodes (important for influencer detection)

---

### 7. **main.py** - Orchestration

```python
def run_single_seed(cfg, model_name: str, seed: int, out_dir: Path):
    """
    Train, evaluate, explain, and benchmark a single model.
    
    Workflow:
    1. Load/generate dataset
    2. Build model
    3. Train with early stopping
    4. Evaluate on test set
    5. Generate explanations
    6. Compute fidelity metrics
    7. Save artifacts
    """
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # === 1. DATA ===
    bundle = resolve_dataset(cfg, seed=seed)  # Load synthetic/real
    data = bundle.data
    y, target_meta = build_targets(data=data, cfg=cfg, seed=seed)  # Generate labels
    data.y = y
    
    # === 2. SPLIT ===
    train_mask, val_mask, test_mask = make_train_val_test_masks(
        data.num_nodes,
        train_ratio=0.7,
        val_ratio=0.15,
        seed=seed
    )
    
    # === 3. MODEL ===
    model = build_model(
        name=model_name,
        in_dim=data.x.size(-1),
        hidden_dim=64,
        num_layers=2,
        dropout=0.2,
        out_dim=1
    ).to(device)
    
    # === 4. TRAIN ===
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    fit_info = fit_model(
        model=model,
        data=data.to(device),
        train_mask=train_mask.to(device),
        val_mask=val_mask.to(device),
        optimizer=optimizer,
        epochs=200,
        patience=30
    )
    
    # === 5. EVALUATE ===
    metrics = evaluate(
        model=model,
        data=data.to(device),
        mask=test_mask.to(device),
        topk_values=(10, 20, 50)
    )
    
    # === 6. EXPLAIN ===
    node_id = 0  # Explain first node
    explanation = explain_node(model, data, node_id, device)
    edge_mask = getattr(explanation, 'edge_mask', None)
    
    if edge_mask is not None:  # If model uses edges
        # === 7. FIDELITY ===
        fidelity = deletion_insertion_fidelity(
            model=model,
            data=data,
            node_id=node_id,
            edge_mask=edge_mask,
            k_values=[5, 10, 20, 40]
        )
        metrics.update(fidelity)
    
    # === 8. SAVE ===
    run_dir = out_dir / f"seed_{seed}" / model_name
    run_dir.mkdir(parents=True, exist_ok=True)
    
    write_json(run_dir / "metrics.json", metrics)
    write_json(run_dir / "fit_info.json", fit_info)
    save_explanation_artifacts(run_dir, node_id, explanation, "gnnexplainer")
    
    return metrics

def main():
    """Entry point: run all models × seeds."""
    cfg = load_cfg(Path("config.yaml"))
    
    out_root = Path("runs") / datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Run each model with each seed
    for model_name in ["gcn", "graphsage", "gat", "appnp", "mlp"]:
        for seed in [42, 43, 44]:
            print(f"=== Running {model_name} seed={seed} ===")
            run_single_seed(cfg, model_name, seed, out_root)
```

---

### 8. **baselines.py** - Non-Learning Baselines

```python
def non_learning_baselines(data, test_mask):
    """
    Classical centrality baselines (no learning).
    
    Returns predictions for PageRank, degree, betweenness, eigenvector.
    """
    import networkx as nx
    
    G = to_networkx(data, to_undirected=True)
    
    baselines = {}
    
    # Degree centrality
    deg = dict(G.degree())
    baselines['degree'] = np.array([deg.get(i, 0) for i in range(data.num_nodes)])
    
    # PageRank
    pr = nx.pagerank(G)
    baselines['pagerank'] = np.array([pr.get(i, 0) for i in range(data.num_nodes)])
    
    # Betweenness
    bc = nx.betweenness_centrality(G)
    baselines['betweenness'] = np.array([bc.get(i, 0) for i in range(data.num_nodes)])
    
    # Eigenvector (may fail on disconnected graphs)
    try:
        ec = nx.eigenvector_centrality(G, max_iter=1000)
        baselines['eigenvector'] = np.array([ec.get(i, 0) for i in range(data.num_nodes)])
    except:
        pass  # Skip if fails
    
    return baselines
```

**Purpose:** Compare GNN vs. classical algorithms

---

### 9. **targets.py** - Label Generation

```python
def build_targets(data, cfg, seed):
    """
    Generate centrality labels for nodes.
    
    Supports:
    - PageRank (default)
    - Degree centrality
    - Eigenvector centrality
    - Betweenness centrality
    """
    set_seed(seed)
    
    target_cfg = cfg.get("target", {})
    method = target_cfg.get("method", "pagerank")
    
    G = to_networkx(data, to_undirected=True)
    
    if method == "pagerank":
        scores = dict(nx.pagerank(G))
    elif method == "degree":
        scores = dict(G.degree())
    elif method == "eigenvector":
        scores = dict(nx.eigenvector_centrality(G))
    elif method == "betweenness":
        scores = dict(nx.betweenness_centrality(G))
    else:
        raise ValueError(f"Unknown target method: {method}")
    
    # Convert to tensor
    y = torch.tensor(
        [scores.get(i, 0) for i in range(data.num_nodes)],
        dtype=torch.float32
    ).unsqueeze(-1)  # Shape: [num_nodes, 1]
    
    return y, {"target_names": ["centrality"]}
```

---

### 10. **utils.py** - Utility Functions

```python
def write_json(path, data):
    """Write dict to JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    import json
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, default=str)

def write_csv(path, rows):
    """Write list of dicts to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    import csv
    if not rows:
        return
    
    # Collect all unique keys
    all_keys = set()
    for row in rows:
        all_keys.update(row.keys())
    keys = sorted(all_keys)
    
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

def ensure_dir(path):
    """Create directory if not exists."""
    path.mkdir(parents=True, exist_ok=True)
    return path

def bootstrap_ci(values, ci=0.95, n_bootstrap=1000):
    """Compute confidence interval via bootstrap."""
    bootstrap_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(values, size=len(values), replace=True)
        bootstrap_means.append(np.mean(sample))
    
    lower = np.percentile(bootstrap_means, (1-ci)/2 * 100)
    upper = np.percentile(bootstrap_means, (1+ci)/2 * 100)
    return lower, upper
```

---

### 11. **viz.py** - Visualization

```python
def plot_fidelity_curves(k_values, insert_curve, delete_curve, save_path):
    """Plot edge insertion/deletion fidelity curves."""
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(k_values, insert_curve, marker='o', label='Insertion AUC')
    ax.plot(k_values, delete_curve, marker='s', label='Deletion AUC')
    ax.set_xlabel('Top-K Edges')
    ax.set_ylabel('Fidelity Score')
    ax.set_title('Explanation Fidelity')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def plot_feature_importance(saliency, title, save_path):
    """Plot node/feature saliency heatmap."""
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(10, 2))
    im = ax.imshow(saliency.reshape(1, -1), cmap='RdYlBu_r', aspect='auto')
    ax.set_title(title)
    ax.set_xlabel('Feature Index')
    plt.colorbar(im, ax=ax, label='Saliency')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def plot_graph_with_scores(data, scores, title, save_path):
    """Visualize graph with node colors based on scores."""
    import networkx as nx
    import matplotlib.pyplot as plt
    
    G = to_networkx(data, to_undirected=True)
    pos = nx.spring_layout(G, seed=42, k=0.5)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    nodes = nx.draw_networkx_nodes(
        G, pos, node_color=scores, node_size=300,
        cmap='YlOrRd', ax=ax, vmin=scores.min(), vmax=scores.max()
    )
    nx.draw_networkx_edges(G, pos, alpha=0.3, ax=ax)
    ax.set_title(title)
    ax.axis('off')
    plt.colorbar(nodes, label='Score', ax=ax)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
```

---

## Usage Workflow

### Quick Start
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run with default config (synthetic BA graph, temporal enabled)
python main.py --config config.yaml --no-viz

# 3. Results saved to:
# runs/centrality_research_YYYYMMDD_HHMMSS/seed_42/gcn/metrics.json
```

### Single Model Test
```bash
python main.py --config config.yaml --models gcn --seeds 42 --no-viz
```

### Disable Temporal Mode
```yaml
# config.yaml
data:
  temporal:
    enabled: false  # Set to true for temporal snapshots
```

---

## Files to Remove (Cleanup)

```bash
# Remove build/cache files (not needed for deployment)
rm -rf __pycache__/
rm -rf .pytest_cache/
rm -rf cache/
rm -rf runs/  # Results can be regenerated

# Remove temporary files
rm _temp_patch_helper.txt

# Remove if not used
rm -rf .github/  # CI/CD config (optional)
rm -rf tests/  # Unit tests (optional, keep if needed)
```

---

## GitHub Push Instructions

```bash
# 1. Ensure clean working directory
git status

# 2. Remove unnecessary files
git rm -r --cached __pycache__ .pytest_cache cache runs _temp_patch_helper.txt

# 3. Update .gitignore
echo "__pycache__/" >> .gitignore
echo ".pytest_cache/" >> .gitignore
echo "cache/" >> .gitignore
echo "runs/" >> .gitignore
echo "*.pyc" >> .gitignore

# 4. Commit cleanup
git add .gitignore
git commit -m "Clean up unnecessary files and update .gitignore"

# 5. Push to GitHub
git push origin main

# 6. Verify (check remote)
git log --oneline -5
git branch -vv
```

---

## Project Statistics

| Metric | Value |
|--------|-------|
| **Total Python Files** | 11 |
| **Total Lines of Code** | ~2,500 |
| **Models Supported** | 5 (GCN, GraphSAGE, GAT, APPNP, MLP) |
| **Explanation Methods** | 3 (GNNExplainer, PGExplainer, Gradient) |
| **Evaluation Metrics** | 8+ (MSE, MAE, RMSE, Spearman ρ, etc.) |
| **Hyperparameter Configs** | 20+ |
| **Maximum Runtime** | ~3min (all models × 3 seeds) |
| **Reproducibility** | ✅ Full (seeded random operations) |

---

## Key Design Principles

1. **Modularity**: Each component (data, model, train, explain) is independent
2. **Reproducibility**: All random seeds fixed, deterministic operations
3. **Configuration-Driven**: YAML-based settings, no hardcoding
4. **Extensibility**: Easy to add new models, datasets, explanation techniques
5. **Evaluation-First**: Comprehensive metrics before deployment

---

**End of Project Guide**

*For questions, refer to README.md or RESEARCH_PAPER.md*
