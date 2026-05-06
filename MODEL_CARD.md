# Model Card

## Model Summary
This repository trains node-level regression models for influence/centrality prediction on graph data.

## Intended Use
- Academic research on centrality approximation
- Benchmarking explainability methods for graph models

## Not Intended Use
- High-stakes decision systems without robust auditing
- Manipulative social targeting use-cases

## Training Data
- Synthetic graph families (BA/ER/WS/SBM/motif)
- Citation/social benchmark graphs

## Metrics
- MSE, MAE, RMSE, R2
- Spearman, Kendall
- Precision@k, Recall@k, NDCG@k, Jaccard@k

## Risks and Limitations
- Synthetic diffusion can diverge from real contagion processes
- Explanations are post-hoc and not causal

## Reproducibility
Use fixed seeds, config snapshots, and run artifact folders in `runs/`.
