from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def plot_summary(results: pd.DataFrame, output_path: str | None = None) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    if not results.empty:
        axes[0].bar(results["dataset"].unique(), results.groupby("dataset")["accuracy"].mean())
        axes[0].set_title("Mean accuracy by dataset")
        axes[1].bar(results["dataset"].unique(), results.groupby("dataset")["conformal_coverage"].mean())
        axes[1].set_title("Mean conformal coverage by dataset")
    else:
        axes[0].text(0.5, 0.5, "No data", ha="center", va="center")
        axes[1].text(0.5, 0.5, "No data", ha="center", va="center")
    fig.tight_layout()
    out = Path(output_path) if output_path else Path(__file__).resolve().parents[2] / "results" / "figures" / "summary.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
