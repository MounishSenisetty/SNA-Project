import argparse
import json
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Aggregate run metrics into paper tables")
    parser.add_argument("--runs", type=str, default="runs", help="Root runs directory")
    parser.add_argument("--out", type=str, default="runs/summary", help="Output directory")
    args = parser.parse_args()

    run_root = Path(args.runs)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for p in run_root.rglob("metrics.json"):
        with open(p, "r", encoding="utf-8") as f:
            rows.append(json.load(f))

    if not rows:
        print("No metrics.json files found.")
        return

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "all_metrics.csv", index=False)

    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    agg = df.groupby("model")[numeric_cols].agg(["mean", "std"])
    agg.to_csv(out_dir / "table_mean_std.csv")

    print(f"Wrote: {out_dir / 'all_metrics.csv'}")
    print(f"Wrote: {out_dir / 'table_mean_std.csv'}")


if __name__ == "__main__":
    main()
