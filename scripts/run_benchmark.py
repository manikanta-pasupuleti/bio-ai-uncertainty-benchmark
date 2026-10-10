#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.experiments.benchmark import run_benchmark


def _resolve_datasets(dataset_arg: str) -> list[str]:
    value = dataset_arg.lower().strip()
    if value in {"all", "both"}:
        return ["breast_cancer", "heart_disease"]
    if value in {"breast_cancer", "breast-cancer", "breastcancer"}:
        return ["breast_cancer"]
    if value in {"heart_disease", "heart-disease", "heartdisease"}:
        return ["heart_disease"]
    raise ValueError(f"Unsupported dataset choice: {dataset_arg!r}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the medical uncertainty benchmark.")
    parser.add_argument("--dataset", default="breast_cancer", help="Dataset name or 'both'.")
    parser.add_argument("--n_trials", type=int, default=2, help="Number of random trials to run.")
    parser.add_argument("--alpha", type=float, default=0.1, help="Nominal coverage level (1 - alpha).")
    args = parser.parse_args()

    for dataset_name in _resolve_datasets(args.dataset):
        result = run_benchmark(dataset_name=dataset_name, n_trials=args.n_trials, alpha=args.alpha)
        print(f"\nDataset: {dataset_name}\n{result.to_string(index=False)}")
        output_path = ROOT / "results" / "tables" / f"{dataset_name}_benchmark.csv"
        print(f"Saved benchmark results to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
