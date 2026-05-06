# Paper Draft Template

## Abstract
This paper studies node-level influence prediction and explainability for graph neural networks under diffusion-based and classical centrality targets.

## 1. Introduction
- Motivation: scalable influencer identification
- Problem setup and contributions
- Research questions RQ1-RQ4

## 2. Related Work
- Graph centrality and influence maximization
- Learning-based centrality approximation
- GNN explainability methods

## 3. Methodology
### 3.1 Formal Task
Node-level regression and ranking from graph structure and node features.

### 3.2 Datasets
- Synthetic (BA/ER/WS/SBM/motif)
- Real (Cora/CiteSeer/PubMed/Karate)

### 3.3 Label Generation
- IC/LT/SIR Monte Carlo labeling
- Classical centrality targets

### 3.4 Models
- Neural: MLP, GCN, GraphSAGE, GAT, APPNP
- Non-learning and classical ML baselines

### 3.5 Explainability
- GNNExplainer, PGExplainer, gradient, random baseline
- Fidelity and stability evaluation

## 4. Experimental Protocol
- Splits, seeds, hardware, runtime
- Hyperparameters and ablations
- Statistical testing

## 5. Results
### 5.1 Prediction Metrics
### 5.2 Ranking Metrics
### 5.3 Cross-family Generalization
### 5.4 Explainability Fidelity and Stability

## 6. Discussion
- Why methods succeed/fail
- Limitations of simulation labels

## 7. Ethics and Limitations
- Compute cost and environmental footprint
- Risks of influence targeting misuse
- Post-hoc explanation caveats

## 8. Conclusion
- Key findings
- Future work
