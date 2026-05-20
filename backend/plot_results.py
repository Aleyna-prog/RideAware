"""
RideAware BA2 - Lernkurven & Visualisierungen

Liest die Ergebnisse aus self_training.py und erstellt:
  1. Lernkurven: Supervised vs. Semi-Supervised (F1 & Accuracy)
  2. Iterationsverlauf: Wie verbessert sich Self-Training pro Iteration?
  3. Balkendiagramm: Direkter Vergleich bei jeder Labelmenge

Usage:
    python plot_results.py                        # alle Plots
    python plot_results.py --type learning_curve  # nur Lernkurven
    python plot_results.py --type iterations      # nur Iterationsverlauf
    python plot_results.py --model nb             # für Naive Bayes Ergebnisse
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ─── Konfiguration ────────────────────────────────────────────────────────────

RESULTS_DIR = Path("results")
PLOTS_DIR   = Path("plots")
PLOTS_DIR.mkdir(exist_ok=True)

# Farben (konsistent mit Frontend)
COLOR_SUPERVISED   = "#4f8cff"   # blau
COLOR_SEMISUP      = "#ff5a5f"   # rot
COLOR_GRID         = "#e5e7eb"

STYLE = {
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    "axes.grid":         True,
    "grid.color":        COLOR_GRID,
    "grid.linewidth":    0.8,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "font.family":       "sans-serif",
    "font.size":         11,
}

# ─── Hilfsfunktionen ──────────────────────────────────────────────────────────

def save_fig(fig: plt.Figure, name: str) -> None:
    """Speichert eine matplotlib Figure als PNG-Datei im plots/ Ordner."""
    path = PLOTS_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Gespeichert: {path}")


def load_experiment(model: str) -> pd.DataFrame | None:
    """Lädt die Ergebnisse des Lernkurven-Experiments aus dem results/ Ordner."""
    path = RESULTS_DIR / f"label_experiment_{model}.csv"
    if not path.exists():
        print(f"  {path} nicht gefunden.")
        print(f"      Führe zuerst aus:  python self_training.py --experiment --model {model}")
        return None
    df = pd.read_csv(path)
    print(f"  Geladen: {path}  ({len(df)} Zeilen)")
    return df


def load_iterations(model: str) -> pd.DataFrame | None:
    """Lädt den Iterationsverlauf des Self-Training-Experiments aus dem results/ Ordner."""
    path = RESULTS_DIR / f"self_training_iter_{model}.csv"
    if not path.exists():
        print(f"  {path} nicht gefunden.")
        print(f"      Führe zuerst aus:  python self_training.py --labeled 100 --model {model}")
        return None
    df = pd.read_csv(path)
    print(f"  Geladen: {path}  ({len(df)} Zeilen)")
    return df


# ─── Plot 1: Lernkurven ───────────────────────────────────────────────────────

def plot_learning_curves(df: pd.DataFrame, model: str) -> None:
    """
    Lernkurven: F1-Score und Accuracy für Supervised vs. Semi-Supervised
    über verschiedene Labelmengen (50 / 100 / 200 / 300).
    """
    plt.rcParams.update(STYLE)

    sup  = df[df["method"] == "Supervised"].sort_values("n_labeled")
    semi = df[df["method"] == "Semi-Supervised"].sort_values("n_labeled")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        f"Lernkurven – Supervised vs. Semi-Supervised  ({model.upper()})",
        fontsize=14, fontweight="bold", y=1.02,
    )

    for ax, metric, ylabel in zip(
        axes,
        ["f1_macro", "accuracy"],
        ["Macro-F1", "Accuracy"],
    ):
        ax.plot(
            sup["n_labeled"], sup[metric],
            marker="o", linewidth=2.5, color=COLOR_SUPERVISED,
            label="Supervised", markersize=7,
        )
        if not semi.empty:
            ax.plot(
                semi["n_labeled"], semi[metric],
                marker="s", linewidth=2.5, color=COLOR_SEMISUP,
                label="Semi-Supervised (Self-Training)", markersize=7,
                linestyle="--",
            )

            # Schraffierung zwischen den Kurven
            x_vals = sup["n_labeled"].values
            y_sup  = sup[metric].values
            y_semi = np.interp(x_vals, semi["n_labeled"].values, semi[metric].values)
            ax.fill_between(
                x_vals, y_sup, y_semi,
                alpha=0.12, color=COLOR_SEMISUP,
                label="Verbesserung durch Self-Training",
            )

        ax.set_xlabel("Anzahl gelabelter Trainingsbeispiele", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(ylabel, fontsize=12, fontweight="bold")
        ax.xaxis.set_major_locator(mticker.FixedLocator(sup["n_labeled"].tolist()))
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=9)

    plt.tight_layout()
    save_fig(fig, f"learning_curves_{model}.png")
    plt.close(fig)


# ─── Plot 2: Iterationsverlauf ─────────────────────────────────────────────────

def plot_iterations(df: pd.DataFrame, model: str) -> None:
    """
    Zeigt wie sich F1 und Accuracy pro Self-Training Iteration entwickeln.
    """
    plt.rcParams.update(STYLE)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        f"Self-Training Iterationsverlauf  ({model.upper()})",
        fontsize=14, fontweight="bold", y=1.02,
    )

    for ax, metric, ylabel in zip(
        axes,
        ["f1_macro", "accuracy"],
        ["Macro-F1", "Accuracy"],
    ):
        ax.plot(
            df["iteration"], df[metric],
            marker="o", linewidth=2.5, color=COLOR_SEMISUP,
            markersize=8,
        )

        # Annotationen: Pseudo-Labels pro Iteration
        if "n_pseudo_added" in df.columns:
            for _, row in df.iterrows():
                if row["n_pseudo_added"] > 0:
                    ax.annotate(
                        f"+{int(row['n_pseudo_added'])}",
                        xy=(row["iteration"], row[metric]),
                        xytext=(0, 10), textcoords="offset points",
                        ha="center", fontsize=9, color="#6b7280",
                    )

        # Startlinie (Supervised Baseline = Iteration 1)
        baseline = df[df["iteration"] == 1][metric].values
        if len(baseline):
            ax.axhline(
                baseline[0], color=COLOR_SUPERVISED,
                linewidth=1.5, linestyle=":", alpha=0.7,
                label=f"Supervised Baseline ({baseline[0]:.3f})",
            )

        ax.set_xlabel("Iteration", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(ylabel, fontsize=12, fontweight="bold")
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=9)

    plt.tight_layout()
    save_fig(fig, f"iterations_{model}.png")
    plt.close(fig)


# ─── Plot 3: Balkendiagramm ───────────────────────────────────────────────────

def plot_bar_comparison(df: pd.DataFrame, model: str) -> None:
    """
    Balkendiagramm: Direkter Vergleich Supervised vs. Semi-Supervised
    für jede Labelmenge nebeneinander.
    """
    plt.rcParams.update(STYLE)

    label_sizes = sorted(df["n_labeled"].unique())
    sup  = df[df["method"] == "Supervised"].set_index("n_labeled")
    semi = df[df["method"] == "Semi-Supervised"].set_index("n_labeled") if \
           "Semi-Supervised" in df["method"].values else None

    x     = np.arange(len(label_sizes))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        f"Vergleich Supervised vs. Semi-Supervised  ({model.upper()})",
        fontsize=14, fontweight="bold", y=1.02,
    )

    for ax, metric, ylabel in zip(
        axes,
        ["f1_macro", "accuracy"],
        ["Macro-F1", "Accuracy"],
    ):
        bars_sup = ax.bar(
            x - width / 2,
            [sup.loc[n, metric] if n in sup.index else 0 for n in label_sizes],
            width, label="Supervised", color=COLOR_SUPERVISED, alpha=0.85,
        )

        if semi is not None:
            bars_semi = ax.bar(
                x + width / 2,
                [semi.loc[n, metric] if n in semi.index else 0 for n in label_sizes],
                width, label="Semi-Supervised", color=COLOR_SEMISUP, alpha=0.85,
            )
            # Werte über den Balken
            for bar in bars_semi:
                h = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2, h + 0.01,
                    f"{h:.2f}", ha="center", va="bottom", fontsize=9,
                )

        # Werte über den Balken (Supervised)
        for bar in bars_sup:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2, h + 0.01,
                f"{h:.2f}", ha="center", va="bottom", fontsize=9,
            )

        ax.set_xlabel("Anzahl gelabelter Trainingsbeispiele", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(ylabel, fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([str(n) for n in label_sizes])
        ax.set_ylim(0, 1.15)
        ax.legend(fontsize=9)

    plt.tight_layout()
    save_fig(fig, f"bar_comparison_{model}.png")
    plt.close(fig)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    """
    Erstellt alle Visualisierungen basierend auf den Experimentergebnissen.
    Unterstützt drei Plot-Typen:
      - learning_curve: Lernkurven Supervised vs. Semi-Supervised
      - iterations: F1/Accuracy-Verlauf pro Self-Training Iteration
      - bar: Balkendiagramm für den direkten Vergleich
    """
    parser = argparse.ArgumentParser(description="RideAware BA2 – Plots")
    parser.add_argument("--type",  type=str, default="all",
                        choices=["all", "learning_curve", "iterations", "bar"],
                        help="Welcher Plot (default: all)")
    parser.add_argument("--model", type=str, default="logreg",
                        choices=["logreg", "nb"],
                        help="Modell: logreg oder nb (default: logreg)")
    args = parser.parse_args()

    print(f"\n{'=' * 70}")
    print(f"RIDEAWARE BA2 – PLOTS  (model: {args.model})")
    print(f"{'=' * 70}\n")

    if args.type in ("all", "learning_curve", "bar"):
        print("Lade Experiment-Ergebnisse...")
        df_exp = load_experiment(args.model)
        if df_exp is not None:
            if args.type in ("all", "learning_curve"):
                print("\nErstelle Lernkurven...")
                plot_learning_curves(df_exp, args.model)
            if args.type in ("all", "bar"):
                print("\nErstelle Balkendiagramm...")
                plot_bar_comparison(df_exp, args.model)

    if args.type in ("all", "iterations"):
        print("\nLade Iterationsverlauf...")
        df_iter = load_iterations(args.model)
        if df_iter is not None:
            print("Erstelle Iterationsverlauf...")
            plot_iterations(df_iter, args.model)

    print(f"\nAlle Plots gespeichert in: {PLOTS_DIR}/")


if __name__ == "__main__":
    main()