# Explainable Centrality GNN (Research Edition)

This repository is structured for a paper-ready study on node-level influence/centrality prediction and explanation.

## 1) Formal Objective And Scope

### Problem Statement

Given a graph $G=(V,E)$ with node features $X$, learn a function $f_\theta(v, G, X)$ that predicts node influence/centrality targets:

$$
\hat{y}_v = f_\theta(v, G, X)
$$

for both:

- Regression quality (numeric centrality/influence score)
- Ranking quality (top influencer identification)

### Supported Targets

- Diffusion influence labels: IC, LT, SIR
- Classical centralities: degree, PageRank, betweenness, closeness, eigenvector
- Optional multi-target setting: diffusion + classical centrality heads

### Hypotheses

- H1: Message-passing GNNs can approximate diffusion influence from local structure and node attributes.
- H2: Better prediction/ranking quality correlates with more faithful explanations.
- H3: Explanation masks align with planted motifs in controlled synthetic graphs.

### Research Questions

- RQ1: How accurate are GNNs vs non-learning and classical ML baselines?
- RQ2: How well do models identify top-$k$ influencers?
- RQ3: Are explanations faithful and stable?
- RQ4: Do models generalize across graph families and real benchmarks?

## 2) Dataset Suite

### Synthetic

- BA, ER, WS, SBM generators
- Directed/undirected and weighted toggles
- Configurable scales and parameters
- Motif-injection generator with ground-truth important edges/nodes

### Real

- Planetoid citation graphs: Cora, CiteSeer, PubMed
- Karate Club for small social benchmark

### Data Integrity

- Deterministic seeds and preprocessing
- Per-run graph and feature metadata saved to artifacts
- Hash-based label cache for expensive diffusion simulations

## 3) Label Engines

- Independent Cascade (IC): fixed or heterogeneous edge probabilities
- Linear Threshold (LT)
- SIR contagion
- Monte Carlo controls: mc_runs, max_steps, n_jobs, seed
- Cached labels keyed by graph structure + diffusion parameters

## 4) Models And Baselines

### Neural Baselines

- MLP (features only)
- GCN
- GraphSAGE
- GAT
- APPNP

### Non-learning Baselines

- Degree, PageRank, betweenness, closeness, eigenvector
- k-hop neighborhood size
- Expected one-step spread ($p \times$ degree)

### Classical ML

- Linear, Ridge, Lasso
- Random Forest

## 5) Training Protocol

- Node-level train/val/test splits
- Multi-seed evaluation (`seeds` in config)
- Early stopping and best checkpoint restoration
- LR scheduling and gradient clipping
- CPU/GPU support
- Runtime recorded per run

## 6) Metrics

### Regression

- MSE, MAE, RMSE, $R^2$
- Spearman and Kendall correlation

### Ranking

- Precision@$k$, Recall@$k$
- NDCG@$k$
- Top-$k$ Jaccard overlap

### Across Seeds

- Mean and std aggregation
- Bootstrap confidence intervals
- Statistical test helpers in utility module

## 7) Explainability

Methods implemented:

- GNNExplainer
- PGExplainer
- Gradient saliency
- Random baseline explanations

Quantitative evaluation implemented:

- Deletion and insertion fidelity curves
- AUC of fidelity curves
- Stability proxy via top-$k$ Jaccard overlap
- Ground-truth motif alignment artifacts supported via saved GT masks

## 8) Visualization Outputs

- True vs predicted influence heatmaps on graph
- Feature attribution bars
- Fidelity curves
- CSV/JSON metrics and per-node prediction tables

All plots are saved to disk for paper figures; no GUI is required.

## 9) Experiment Management

- YAML config-driven runner
- Sweeps across seeds and model list in one command
- Standard run folder layout with:
	- checkpoints
	- metrics
	- predictions
	- explanations
	- plots
	- metadata snapshots

## 10) Reproducibility

- Deterministic seed handling for Python/NumPy/PyTorch/CUDA
- Pinned dependencies in requirements
- CI scaffold and tests included
- License and citation file included

## Quick Start

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python main.py --config config.yaml --no-viz
```

## Reproduce Main Table/Figure

Main baseline experiment:

```powershell
python main.py --config config.yaml --no-viz
```

Outputs appear in `runs/<experiment_name>/` with per-seed and aggregate files.

## Suggested Paper Sections

- Abstract
- Introduction and RQs
- Related Work (centrality learning + GNN explainability)
- Methods (datasets, targets, models, explainers)
- Experimental Protocol
- Results and Statistical Analysis
- Explanation Fidelity and Stability
- Limitations and Ethics

## Limitations And Ethics

- Monte Carlo diffusion labels are computationally expensive.
- Synthetic motifs simplify real-world diffusion complexity.
- Post-hoc explanations are not causal guarantees.
- Influence prediction can be misused for manipulation; include safeguards in deployment discussion.