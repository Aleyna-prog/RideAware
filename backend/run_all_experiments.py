"""
RideAware BA2 - Alle Experimente auf einmal ausführen

Führt alle Schritte automatisch durch:
1. Unlabeled Data erstellen
2. Labelmenge-Experiment LR
3. Labelmenge-Experiment NB
4. Modelle vergleichen
5. Grafiken erstellen

Usage:
    python run_all_experiments.py
"""

import subprocess
from pathlib import Path

def run(cmd: str, description: str) -> bool:
    print(f"\n{'=' * 70}")
    print(f"  {description}")
    print(f"   Befehl: {cmd}")
    print("=" * 70)

    result = subprocess.run(
        cmd, shell=True, text=True,
        capture_output=False
    )

    if result.returncode != 0:
        print(f"\nFehler bei: {cmd}")
        return False

    print(f"\nFertig: {description}")
    return True


def main():
    print("\n" + "=" * 70)
    print("RIDEAWARE BA2 – ALLE EXPERIMENTE")
    print("=" * 70)
    print("Dieser Befehl führt alle Schritte automatisch durch.")
    print("Das dauert ca. 2-5 Minuten.\n")

    # Prüfe ob alle nötigen Dateien vorhanden sind
    required = [
        Path("data/train_pool.csv"),
        Path("data/test.csv"),
        Path("data/train_split.csv"),
        Path("data/synthetic.csv"),
    ]
    for path in required:
        if not path.exists():
            print(f"Fehlende Datei: {path}")
            print("   Führe zuerst aus:")
            print("   python preprocess_reports.py")
            print("   python init_data_splits.py")
            print("   python create_train_split.py")
            return

    print("Alle benötigten Dateien vorhanden.\n")

    steps = [
        # (Befehl, Beschreibung)
        ("python create_unlabeled.py",
         "Schritt 1: Unlabeled Data erstellen"),

        ("python self_training.py --experiment --model logreg",
         "Schritt 2: Labelmenge-Experiment (Logistic Regression)"),

        ("python self_training.py --experiment --model nb",
         "Schritt 3: Labelmenge-Experiment (Naive Bayes)"),

        ("python train_and_compare.py",
         "Schritt 4: Modelle vergleichen (NB vs LR)"),

        ("python plot_results.py --model logreg",
         "Schritt 5a: Grafiken für Logistic Regression"),

        ("python plot_results.py --model nb",
         "Schritt 5b: Grafiken für Naive Bayes"),

        ("python plot_data.py",
         "Schritt 6: Datensatz visualisieren"),
    ]

    success_count = 0
    for cmd, description in steps:
        ok = run(cmd, description)
        if ok:
            success_count += 1
        else:
            print(f"\nSchritt fehlgeschlagen – fahre trotzdem fort...")

    # Abschlussbericht
    print("\n" + "=" * 70)
    print("ABSCHLUSSBERICHT")
    print("=" * 70)
    print(f"{success_count}/{len(steps)} Schritte erfolgreich\n")

    if Path("results/label_experiment_logreg.csv").exists():
        import pandas as pd
        print("ERGEBNISSE LOGISTIC REGRESSION:")
        df = pd.read_csv("results/label_experiment_logreg.csv")
        print(df.to_string(index=False))

    if Path("results/label_experiment_nb.csv").exists():
        import pandas as pd
        print("\nERGEBNISSE NAIVE BAYES:")
        df = pd.read_csv("results/label_experiment_nb.csv")
        print(df.to_string(index=False))

    if Path("results/model_comparison.csv").exists():
        import pandas as pd
        print("\nMODELLVERGLEICH (volles Training):")
        df = pd.read_csv("results/model_comparison.csv")
        print(df.to_string(index=False))

    print("\n" + "=" * 70)
    print("Alle Experimente abgeschlossen.")
    print("\nGrafiken gespeichert in: plots/")
    print("Ergebnisse gespeichert in: results/")
    print("\nNächster Schritt:")
    print("  BERT in Google Colab trainieren")
    print("  Dann Kapitel 5 und 6 der BA schreiben")
    print("=" * 70)


if __name__ == "__main__":
    main()