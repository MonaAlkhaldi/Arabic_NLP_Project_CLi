from __future__ import annotations
from pathlib import Path
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt


def ensure_vis_dir(project_root: Path) -> Path:
    vis_dir = project_root / "outputs" / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)
    return vis_dir


def summarize_lengths(lengths: pd.Series) -> dict:
    return {
        "mean": float(lengths.mean()),
        "median": float(lengths.median()),
        "std": float(lengths.std(ddof=1)) if len(lengths) > 1 else 0.0,
        "min": int(lengths.min()) if len(lengths) else 0,
        "max": int(lengths.max()) if len(lengths) else 0,
    }


def plot_class_distribution_pie(df: pd.DataFrame, label_col: str, out_path: Path) -> None:
    counts = df[label_col].value_counts(dropna=False)

    plt.figure(figsize=(6, 6))
    plt.pie(
        counts.values,
        labels=counts.index.astype(str),
        autopct="%1.1f%%",
        startangle=90
    )
    plt.title("Class Distribution (Pie)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_text_length_histogram(lengths: pd.Series, unit: str, out_path: Path) -> None:
    plt.figure(figsize=(6, 4))
    plt.hist(lengths.values, bins=30)
    plt.title(f"Text Length Histogram ({unit})")
    plt.xlabel(f"Length ({unit})")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_top_words_bar(text_series: pd.Series, out_path: Path, top_k: int = 20) -> None:
  
    tokens = []
    for x in text_series.astype("string").fillna(""):
        tokens.extend(str(x).split())

    if not tokens:
        # Avoid crashing if text is empty
        return

    counts = Counter(tokens).most_common(top_k)
    words = [w for w, _ in counts]
    freqs = [c for _, c in counts]

    plt.figure(figsize=(8, 4))
    plt.bar(words, freqs)
    plt.title(f"Top {top_k} Words")
    plt.xlabel("Word")
    plt.ylabel("Frequency")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
