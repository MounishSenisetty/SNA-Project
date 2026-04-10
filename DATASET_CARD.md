# Dataset Card

## Included Datasets
- Synthetic: BA, ER, WS, SBM, motif-injection
- Real: Cora, CiteSeer, PubMed, Karate Club

## Generation and Preprocessing
- Deterministic generation with fixed seeds
- Optional directed/weighted graphs for synthetic families
- Node features generated if absent

## Labels
- Diffusion labels generated via IC, LT, SIR Monte Carlo simulation
- Optional classical centrality targets

## Splits
- Node-level train/val/test ratio controlled in config

## Bias and Limitations
- Citation graphs may not reflect social influence dynamics
- Synthetic motifs can over-simplify explanation ground truth

## Provenance
Planetoid datasets are downloaded via PyG. Synthetic datasets are generated in-code.
