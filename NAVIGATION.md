# 📚 Project Navigation Guide

Welcome to the **Explainable Influencer Detection in Social Networks using Graph Neural Networks** project!

This guide helps you navigate the project structure and find what you need.

## 🚀 Start Here

**New to the project?** 
1. Read [QUICKSTART.md](QUICKSTART.md) (5 minutes)
2. Run: `python scripts/train.py --dataset Karate --epochs 100`
3. Check results in `results/Karate/gcn/seed_42/`

**Want details?**
1. Read [README.md](README.md) (comprehensive guide)
2. Read [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md) (what was built)
3. Check [CONFIG_GUIDE.md](CONFIG_GUIDE.md) (parameters)

## 📁 Project Structure

```
influencer-detection-gnn/
│
├── 📄 Documentation (START HERE)
│   ├── README.md                    📖 Full documentation
│   ├── QUICKSTART.md                ⚡ 5-minute quick start
│   ├── CONFIG_GUIDE.md              ⚙️ Configuration guide
│   ├── PROJECT_SUMMARY.md           📊 What was built
│   └── requirements.txt              📦 Dependencies
│
├── 🔧 Core Code (Main Implementation)
│   ├── main.py                      ▶️ Entry point (run this)
│   ├── dataset.py                   📊 Data loading & centrality
│   ├── model.py                     🧠 GNN models (GCN, GraphSAGE, GAT)
│   ├── train.py                     🏃 Training pipeline
│   ├── evaluate.py                  📈 Evaluation metrics
│   ├── explain.py                   💡 Explainability methods
│   ├── visualize.py                 📉 Visualization functions
│   └── utils.py                     🛠️ Utility functions
│
├── 📜 Scripts (CLI Tools)
│   ├── scripts/train.py             🏃 Command-line training
│   ├── scripts/top_k_analysis.py    🔍 Top-k analysis tool
│   ├── examples.py                  🎓 Interactive examples
│   └── test_smoke.py                ✅ Smoke test
│
├── 📂 Directories (Outputs)
│   ├── data/                        📥 Downloaded datasets
│   ├── results/                     📋 Final results
│   ├── models/                      💾 Saved models
│   ├── checkpoints/                 🔄 Training checkpoints
│   ├── outputs/                     📤 Experiment outputs
│   └── notebooks/                   📓 Jupyter notebooks
│
└── 📊 Test Results (Example Output)
    └── results_test/Karate/gcn/seed_42/
        ├── metrics.json
        ├── predictions.csv
        ├── evaluations/
        └── visualizations/*.png
```

## 🎯 Quick Navigation

### "I want to train a model"
→ Read [QUICKSTART.md](QUICKSTART.md)
→ Run `python scripts/train.py --dataset Karate`

### "I want to understand how it works"
→ Read [README.md](README.md)
→ Read code comments in `main.py`, `model.py`
→ Run `python examples.py`

### "I want to configure hyperparameters"
→ Read [CONFIG_GUIDE.md](CONFIG_GUIDE.md)
→ See `scripts/train.py --help`

### "I want to compare models"
→ Read [CONFIG_GUIDE.md](CONFIG_GUIDE.md) - Multi-model section
→ Run multiple training commands

### "I want to understand explanations"
→ Read [README.md](README.md) - Explainability section
→ Check `explain.py` code
→ Run `examples.py` → Example 5

### "I want detailed analysis"
→ Read [README.md](README.md) - Evaluation section
→ Run `examples.py` → Example 4
→ Check `results/` directory

### "I want to use my own dataset"
→ Read [README.md](README.md) - Advanced Usage
→ Modify `dataset.py`
→ Create InfluencerDataset subclass

## 📖 Documentation Map

| File | Purpose | Read Time | Audience |
|------|---------|-----------|----------|
| [QUICKSTART.md](QUICKSTART.md) | Get started quickly | 5 min | Everyone |
| [README.md](README.md) | Full documentation | 20 min | Developers |
| [CONFIG_GUIDE.md](CONFIG_GUIDE.md) | Hyperparameter guide | 10 min | Researchers |
| [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md) | Project overview | 15 min | Stakeholders |
| Code docstrings | Implementation details | Varies | Developers |

## 🔍 Finding Things

### By Task

**Want to...**
- Train a model → `main.py` + `train.py`
- Load data → `dataset.py`
- See metrics → `evaluate.py`
- Understand why → `explain.py`
- Make plots → `visualize.py`
- Use from CLI → `scripts/train.py`
- See examples → `examples.py`

### By Component

**What is...**
- `main.py` → Entry point, orchestrates everything
- `dataset.py` → Loads Cora/CiteSeer/Karate, computes centrality
- `model.py` → GCN, GraphSAGE, GAT implementations
- `train.py` → Training loop, early stopping, checkpoints
- `evaluate.py` → MSE, MAE, Spearman, Precision@k, etc.
- `explain.py` → Gradient saliency, important neighbors
- `visualize.py` → Loss curves, predictions, explanations
- `utils.py` → Helpers for metrics, files, reproducibility

### By File Type

**Code files (read to understand):**
- `*.py` - All Python modules with docstrings

**Documentation (read for guidance):**
- `README.md` - Complete reference
- `QUICKSTART.md` - Get started fast
- `CONFIG_GUIDE.md` - Tune parameters
- `PROJECT_SUMMARY.md` - Project overview

**Configuration (modify for experiments):**
- `requirements.txt` - Dependencies
- `scripts/train.py` - CLI parameters

**Output (generated after running):**
- `results/` - Final results
- `results_test/` - Test results
- `*.csv` - Predictions and metrics
- `*.png` - Visualizations
- `*.json` - Detailed results

## 🚦 Workflow Flowchart

```
START
  ↓
INSTALL DEPENDENCIES (requirements.txt)
  ↓
CHOOSE DATASET (Cora, CiteSeer, Karate)
  ↓
CHOOSE MODEL (GCN, GraphSAGE, GAT)
  ↓
RUN TRAINING (main.py or scripts/train.py)
  ↓
EVALUATE RESULTS (evaluate.py)
  ↓
GENERATE EXPLANATIONS (explain.py)
  ↓
CREATE VISUALIZATIONS (visualize.py)
  ↓
SAVE ARTIFACTS (results/)
  ↓
ANALYZE & PUBLISH
  ↓
END
```

## 🎓 Learning Path

**Level 1: Beginner** (1-2 hours)
1. Install project: `pip install -r requirements.txt`
2. Run quick start: `python scripts/train.py --dataset Karate`
3. Read QUICKSTART.md
4. Check results in `results/Karate/`

**Level 2: Intermediate** (2-4 hours)
1. Read README.md sections: Architecture, Training, Evaluation
2. Run multiple models: `for m in gcn graphsage gat; do python scripts/train.py --model $m; done`
3. Read CONFIG_GUIDE.md
4. Try different hyperparameters

**Level 3: Advanced** (4-8 hours)
1. Read entire README.md and code comments
2. Run `examples.py` for all 6 examples
3. Modify code: Add new model in `model.py`
4. Create custom dataset in `dataset.py`
5. Extend explainability in `explain.py`

**Level 4: Expert** (8+ hours)
1. Understand all implementation details
2. Contribute enhancements
3. Write research papers using results
4. Deploy to production

## 💡 Common Tasks

### Task 1: Train on Karate Club
```bash
python scripts/train.py --dataset Karate --epochs 100
# Results → results/Karate/gcn/seed_42/
```

### Task 2: Compare Models
```bash
python scripts/train.py --dataset Cora --model gcn --epochs 200
python scripts/train.py --dataset Cora --model graphsage --epochs 200
# Compare metrics in results/Cora/
```

### Task 3: Analyze Results
```python
import pandas as pd
results = pd.read_csv('results/Karate/gcn/seed_42/metrics.json')
print(results)
```

### Task 4: View Predictions
```bash
python -c "import pandas as pd; df=pd.read_csv('results/Karate/gcn/seed_42/predictions.csv'); print(df.head())"
```

### Task 5: See Visualizations
```bash
# Open any PNG file in results/*/visualizations/
# Example plots: training_curves.png, predictions_vs_actual.png, etc.
```

## 🔗 Related Files

**If you're reading...**
- `README.md` → Also read: QUICKSTART.md, CONFIG_GUIDE.md
- `QUICKSTART.md` → Also read: README.md for details
- `CONFIG_GUIDE.md` → Also read: README.md for context
- `main.py` → Also read: dataset.py, model.py, train.py
- `model.py` → Also read: train.py for usage
- `explain.py` → Also read: visualize.py for output

## 📊 Example Output Structure

After running the training:
```
results/
├── Karate/                  (dataset)
│   └── gcn/                 (model)
│       └── seed_42/         (random seed)
│           ├── metrics.json                     ← Metrics summary
│           ├── predictions.csv                  ← Node predictions
│           ├── evaluation_table.csv            ← Evaluation results
│           ├── top_k_comparison.csv            ← Top-10 analysis
│           ├── explanations.json               ← Explanations
│           ├── checkpoints/
│           │   └── model_checkpoint.pt
│           └── visualizations/
│               ├── training_curves.png
│               ├── predictions_vs_actual.png
│               ├── top_k_comparison.png
│               ├── metrics_table.png
│               ├── explanation_node_0.png
│               ├── explanation_node_1.png
│               └── explanation_node_2.png
```

## ✅ Verification Checklist

- [ ] Dependencies installed: `pip install -r requirements.txt`
- [ ] All modules compile: `python -m py_compile *.py`
- [ ] Help available: `python scripts/train.py --help`
- [ ] Test passes: `python test_smoke.py`
- [ ] Results generated: `results/` directory exists

## 🆘 Troubleshooting

**Issue: "Module not found"**
→ Run `pip install -r requirements.txt`

**Issue: "Slow training"**
→ Use smaller dataset or reduce epochs

**Issue: "Out of memory"**
→ Reduce hidden_dim or batch_size

**Issue: "Can't find results"**
→ Check `results/` directory

**Issue: "Different results"**
→ Set `seed` parameter consistently

## 📞 Support Resources

1. **For setup issues**: Check this guide + README.md
2. **For hyperparameter questions**: Read CONFIG_GUIDE.md
3. **For code understanding**: Read docstrings in each file
4. **For examples**: Run examples.py
5. **For errors**: Check error message + traceback

## 🎉 You're Ready!

You now know:
- ✅ Where to find everything
- ✅ How to run the project
- ✅ What each file does
- ✅ How to train models
- ✅ How to view results

**Next step:** Read [QUICKSTART.md](QUICKSTART.md) and run your first model!

---

**Happy learning! 🚀**

For detailed information, see:
- Full guide: [README.md](README.md)
- Quick start: [QUICKSTART.md](QUICKSTART.md)
- Configuration: [CONFIG_GUIDE.md](CONFIG_GUIDE.md)
- Project overview: [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)
