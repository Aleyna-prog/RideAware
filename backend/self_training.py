"""
RideAware BA2 - Self-Training (Semi-Supervised Learning)

Usage:
    python self_training.py
    python self_training.py --labeled 100 --iterations 5 --threshold 0.85
    python self_training.py --experiment --model logreg
"""

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score, classification_report

LABELS = [
    "Gefahrenstelle",
    "Hindernis",
    "Markierung oder Schild",
    "Ampel",
    "Lückenschluss",
]

TRAIN_POOL_PATH = Path("data/train_pool.csv")
TEST_PATH       = Path("data/test.csv")
UNLABELED_PATH  = Path("data/unlabeled.csv")
RESULTS_DIR     = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


def print_header(title: str) -> None:
    """Gibt einen formatierten Abschnittstitel in der Konsole aus."""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def subsample_labeled(df: pd.DataFrame, n_labeled: int) -> pd.DataFrame:
    """
    Wählt n_labeled Beispiele stratifiziert aus dem DataFrame aus, sodass
    alle Klassen möglichst gleichmäßig vertreten sind. Falls n_labeled größer
    als der Datensatz ist, wird der vollständige Datensatz zurückgegeben.
    """
    if n_labeled >= len(df):
        return df.copy()
    result = (
        df.groupby("label", group_keys=False)
        .apply(lambda g: g.sample(
            min(len(g), max(1, n_labeled // len(LABELS))),
            random_state=42,
        ))
        .reset_index(drop=True)
    )
    print(f"  -> Subsampling auf {len(result)} gelabelte Beispiele")
    return result


def load_unlabeled(path: Path) -> list[str]:
    """Lädt ungelabelte Texte aus einer CSV-Datei für das Self-Training."""
    if not path.exists():
        print(f"  {path} nicht gefunden – kein ungelabeltes Material.")
        return []
    df = pd.read_csv(path)
    df["text"] = df["text"].astype(str)
    return df["text"].tolist()


def make_pipeline(model_type: str = "logreg") -> Pipeline:
    """
    Erstellt eine sklearn Pipeline mit TF-IDF Vektorisierung und dem
    gewählten Klassifikator. Unterstützt Logistic Regression ('logreg')
    und Naive Bayes ('nb').
    """
    clf = (
        LogisticRegression(max_iter=2000, class_weight="balanced")
        if model_type == "logreg"
        else MultinomialNB(alpha=1.0)
    )
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_df=0.95)),
        ("clf", clf),
    ])


def evaluate(model, X_test, y_test, label) -> dict:
    """Bewertet das Modell auf den Testdaten und gibt Accuracy und Macro-F1 zurück."""
    preds = model.predict(X_test)
    acc   = accuracy_score(y_test, preds)
    f1m   = f1_score(y_test, preds, average="macro", zero_division=0)
    print(f"\n  [{label}]  Accuracy: {acc:.3f}  |  Macro-F1: {f1m:.3f}")
    return {"label": label, "accuracy": acc, "f1_macro": f1m}


def self_training(
    X_labeled, y_labeled, X_unlabeled, X_test, y_test,
    model_type="logreg", n_iterations=5, threshold=0.85, verbose=True,
) -> list[dict]:
    """
    Implementiert den Self-Training Algorithmus (Semi-Supervised Learning).
    In jeder Iteration wird das Modell neu trainiert und auf den ungelabelten
    Daten angewendet. Beispiele, bei denen die Konfidenz den Schwellenwert
    überschreitet, werden als Pseudo-Labels dem Trainingsset hinzugefügt.
    Das gibt zurück wie sich Accuracy und F1 pro Iteration entwickeln.
    """
    results  = []
    X_train  = list(X_labeled)
    y_train  = list(y_labeled)
    X_pool   = list(X_unlabeled)

    print_header(f"SELF-TRAINING ({model_type.upper()}) | Start: {len(X_train)} gelabelte Beispiele")
    print(f"  Ungelabelte Beispiele: {len(X_pool)}")
    print(f"  Iterationen: {n_iterations}  |  Konfidenz-Schwelle: {threshold}")

    for iteration in range(n_iterations):
        print(f"\n── Iteration {iteration + 1}/{n_iterations} " + "─" * 40)

        model = make_pipeline(model_type)
        model.fit(X_train, y_train)

        metrics = evaluate(model, X_test, y_test, f"iter_{iteration + 1}")
        metrics.update({
            "iteration": iteration + 1,
            "n_labeled": len(X_train),
            "n_pseudo_added": 0,
        })

        if not X_pool:
            print("  Kein ungelabeltes Material mehr.")
            results.append(metrics)
            break

        proba         = model.predict_proba(X_pool)
        max_conf      = proba.max(axis=1)
        pseudo_labels = model.predict(X_pool)
        confident     = max_conf >= threshold
        n_added       = int(confident.sum())

        # Konfidenz-Statistik für die Arbeit (Kapitel 6.3)
        print(f"  Konfidenz-Statistik der Pseudo-Labels:")
        print(f"    Max:        {max_conf.max():.3f}")
        print(f"    Mittelwert: {max_conf.mean():.3f}")
        print(f"    Median:     {np.median(max_conf):.3f}")
        print(f"    Min:        {max_conf.min():.3f}")
        print(f"    >= 0.85:    {(max_conf >= 0.85).sum()} / {len(max_conf)}")
        print(f"    >= 0.75:    {(max_conf >= 0.75).sum()} / {len(max_conf)}")
        print(f"    >= 0.70:    {(max_conf >= 0.70).sum()} / {len(max_conf)}")

        # Konfidenz-Histogramm speichern
        plots_dir = Path("plots")
        plots_dir.mkdir(exist_ok=True)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(max_conf, bins=20, edgecolor="black", alpha=0.7)
        ax.axvline(0.85, color="red",    linestyle="--", label="Schwellenwert 0,85")
        ax.axvline(0.75, color="orange", linestyle="--", label="Alternativ 0,75")
        ax.set_xlabel("Maximale Konfidenz pro Beispiel")
        ax.set_ylabel("Anzahl Beispiele")
        ax.set_title(f"Konfidenzverteilung – Iteration {iteration + 1}")
        ax.legend()
        plt.tight_layout()
        hist_path = plots_dir / f"konfidenz_verteilung_iter{iteration + 1}.png"
        fig.savefig(hist_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Konfidenz-Verteilung gespeichert: {hist_path}")

        if verbose:
            print(f"  Konfidente Pseudo-Labels: {n_added} / {len(X_pool)}")

        if n_added == 0:
            print("  Keine konfidenziellen Pseudo-Labels – stoppe.")
            results.append(metrics)
            break

        X_pseudo = [X_pool[i] for i in range(len(X_pool)) if confident[i]]
        y_pseudo = [pseudo_labels[i] for i in range(len(X_pool)) if confident[i]]
        X_train.extend(X_pseudo)
        y_train.extend(y_pseudo)
        X_pool = [X_pool[i] for i in range(len(X_pool)) if not confident[i]]

        metrics["n_pseudo_added"] = n_added
        results.append(metrics)
        print(f"  Trainingsset jetzt: {len(X_train)} Beispiele (+ {n_added} Pseudo-Labels)")

    print_header("FINALES MODELL")
    final_model = make_pipeline(model_type)
    final_model.fit(X_train, y_train)
    preds = final_model.predict(X_test)
    print(f"\nTrainingsset (final): {len(X_train)} Beispiele")
    print(classification_report(y_test, preds, labels=LABELS, zero_division=0))

    return results


def run_label_size_experiment(
    df_pool, X_unlabeled, X_test, y_test,
    label_sizes, model_type="logreg", n_iterations=5, threshold=0.85,
) -> pd.DataFrame:
    """
    Führt das Lernkurven-Experiment durch. Für jede Labelmenge in label_sizes
    wird zuerst ein rein überwachtes Modell (Supervised Baseline) trainiert,
    danach Self-Training angewendet. Die Ergebnisse werden als DataFrame
    zurückgegeben und können mit plot_results.py visualisiert werden.
    """
    all_rows = []

    for n in label_sizes:
        print_header(f"LABELMENGE: {n}")

        # ── FIX: DataFrame direkt subsamplen statt load_labeled aufrufen ──
        df_labeled = subsample_labeled(df_pool, n_labeled=n)
        X_lab = df_labeled["text"].tolist()
        y_lab = df_labeled["label"].tolist()

        # Supervised Baseline
        print(f"\n  [Supervised] Training mit {len(X_lab)} Beispielen...")
        model_sup = make_pipeline(model_type)
        model_sup.fit(X_lab, y_lab)
        acc_sup = accuracy_score(y_test, model_sup.predict(X_test))
        f1_sup  = f1_score(y_test, model_sup.predict(X_test), average="macro", zero_division=0)
        print(f"  Supervised  → Acc: {acc_sup:.3f}  F1: {f1_sup:.3f}")

        all_rows.append({
            "n_labeled": n,
            "method":    "Supervised",
            "accuracy":  acc_sup,
            "f1_macro":  f1_sup,
        })

        # Semi-Supervised
        if X_unlabeled:
            results = self_training(
                X_labeled=X_lab, y_labeled=y_lab,
                X_unlabeled=X_unlabeled,
                X_test=X_test, y_test=y_test,
                model_type=model_type,
                n_iterations=n_iterations,
                threshold=threshold,
                verbose=False,
            )
            last = results[-1]
            print(f"  Semi-Sup.   → Acc: {last['accuracy']:.3f}  F1: {last['f1_macro']:.3f}")
            all_rows.append({
                "n_labeled": n,
                "method":    "Semi-Supervised",
                "accuracy":  last["accuracy"],
                "f1_macro":  last["f1_macro"],
            })
        else:
            print("  Keine ungelabelten Daten – Semi-Supervised übersprungen.")

    return pd.DataFrame(all_rows)


def main():
    """
    Einstiegspunkt für das Self-Training Skript. Unterstützt zwei Modi:
    - Standard: Self-Training mit der gesamten Trainingsdatenmenge (oder --labeled N)
    - Experiment (--experiment): Lernkurven-Experiment über verschiedene Labelmengen
    """
    parser = argparse.ArgumentParser(description="RideAware BA2 – Self-Training")
    parser.add_argument("--labeled",    type=int,   default=None)
    parser.add_argument("--iterations", type=int,   default=5)
    parser.add_argument("--threshold",  type=float, default=0.85)
    parser.add_argument("--model",      type=str,   default="logreg",
                        choices=["logreg", "nb"])
    parser.add_argument("--experiment", action="store_true")
    args = parser.parse_args()

    print_header("RIDEAWARE BA2 – SELF-TRAINING")

    if not TRAIN_POOL_PATH.exists():
        print(f"\n{TRAIN_POOL_PATH} nicht gefunden.")
        print("   Führe zuerst init_data_splits.py aus.")
        return

    if not TEST_PATH.exists():
        print(f"\n{TEST_PATH} nicht gefunden.")
        return

    df_pool = pd.read_csv(TRAIN_POOL_PATH)
    df_pool["text"]  = df_pool["text"].astype(str)
    df_pool["label"] = df_pool["label"].astype(str)

    df_test = pd.read_csv(TEST_PATH)
    df_test["text"]  = df_test["text"].astype(str)
    df_test["label"] = df_test["label"].astype(str)

    X_test      = df_test["text"].tolist()
    y_test      = df_test["label"].tolist()
    X_unlabeled = load_unlabeled(UNLABELED_PATH)

    print(f"\n  Trainingsdaten (Pool): {len(df_pool)} Beispiele")
    print(f"  Testdaten:             {len(df_test)} Beispiele")
    print(f"  Ungelabelte Daten:     {len(X_unlabeled)} Beispiele")

    if args.experiment:
        label_sizes = [50, 100, 150, 225]
        print_header(f"EXPERIMENT: Labelmengen {label_sizes}")

        results_df = run_label_size_experiment(
            df_pool=df_pool,
            X_unlabeled=X_unlabeled,
            X_test=X_test,
            y_test=y_test,
            label_sizes=label_sizes,
            model_type=args.model,
            n_iterations=args.iterations,
            threshold=args.threshold,
        )

        out_path = RESULTS_DIR / f"label_experiment_{args.model}.csv"
        results_df.to_csv(out_path, index=False)

        print_header("ERGEBNISSE")
        print(results_df.to_string(index=False))
        print(f"\nErgebnisse gespeichert: {out_path}")
        print("   Tipp: Führe plot_results.py aus um Lernkurven zu generieren.")

    else:
        df_labeled = subsample_labeled(df_pool, args.labeled) if args.labeled else df_pool.copy()
        X_labeled  = df_labeled["text"].tolist()
        y_labeled  = df_labeled["label"].tolist()

        results = self_training(
            X_labeled=X_labeled, y_labeled=y_labeled,
            X_unlabeled=X_unlabeled,
            X_test=X_test, y_test=y_test,
            model_type=args.model,
            n_iterations=args.iterations,
            threshold=args.threshold,
        )

        out_path = RESULTS_DIR / f"self_training_iter_{args.model}.csv"
        pd.DataFrame(results).to_csv(out_path, index=False)
        print(f"\nIterationsverlauf gespeichert: {out_path}")


if __name__ == "__main__":
    main()