from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# Dateien (bei dir im backend/data Ordner)
ALL_PATH = Path("data/train.csv")
TRAIN_PATH = Path("data/train_split.csv")
TEST_PATH = Path("data/test.csv")

LABELS = ["Hindernis", "Infrastrukturproblem", "Gefahrenstelle", "Positives Feedback", "Spam"]

def plot_class_distribution(df: pd.DataFrame, title: str):
    counts = df["label"].value_counts().reindex(LABELS, fill_value=0)
    plt.figure()
    plt.bar(counts.index, counts.values)
    plt.title(title)
    plt.ylabel("Anzahl")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()

def plot_text_length(df: pd.DataFrame, title: str):
    lengths = df["text"].astype(str).apply(len)
    plt.figure()
    plt.hist(lengths, bins=10)
    plt.title(title)
    plt.xlabel("Textlänge (Zeichen)")
    plt.ylabel("Häufigkeit")
    plt.tight_layout()

def main():
    df_all = pd.read_csv(ALL_PATH)
    df_train = pd.read_csv(TRAIN_PATH) if TRAIN_PATH.exists() else None
    df_test = pd.read_csv(TEST_PATH) if TEST_PATH.exists() else None

    # 1) Gesamte Klassenverteilung
    plot_class_distribution(df_all, "Klassenverteilung – Gesamt (train.csv)")

    # 2) Train/Test Verteilung (falls vorhanden)
    if df_train is not None and df_test is not None:
        plot_class_distribution(df_train, "Klassenverteilung – Train (train_split.csv)")
        plot_class_distribution(df_test, "Klassenverteilung – Test (test.csv)")

    # 3) Textlängen
    plot_text_length(df_all, "Textlängen-Verteilung – Gesamt (train.csv)")

    plt.show()

if __name__ == "__main__":
    main()
