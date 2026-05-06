"""Smoke test script for the project"""
from main import run_experiment
import warnings
warnings.filterwarnings('ignore')

print("\n" + "="*60)
print("RUNNING SMOKE TEST ON KARATE CLUB DATASET")
print("="*60)

results = run_experiment(
    dataset_name='Karate',
    centrality_method='pagerank',
    model_name='gcn',
    epochs=50,
    seed=42,
    output_dir='./results_test',
    save_artifacts=True
)

print("\n✓ SMOKE TEST PASSED!")
print(f"Output directory: {results['output_path']}")
