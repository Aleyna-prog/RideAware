from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

TRAIN_POOL_PATH = Path("data/train_pool.csv")
TRAIN_PATH      = Path("data/train_split.csv")
TEST_PATH       = Path("data/test.csv")

LABELS = ["Gefahrenstelle", "Hindernis", "Markierung oder Schild", "Ampel", "Lückenschluss"]

def plot_class_distribution(df: pd.DataFrame, title: str):
    counts = df["label"].value_counts().reindex(LABELS, fill_value=0)
    plt.figure(figsize=(8, 4))
    bars = plt.bar(counts.index, counts.values, color=["#ff5a5f","#ffb020","#2dd4bf","#4f8cff","#a78bfa"])
    plt.title(title, fontweight="bold")
    plt.ylabel("Anzahl")
    plt.xticks(rotation=20, ha="right")
    for bar in bars:
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                 str(int(bar.get_height())), ha="center", fontsize=9)
    plt.tight_layout()

def plot_text_length(df: pd.DataFrame, title: str):
    lengths = df["text"].astype(str).apply(len)
    plt.figure(figsize=(8, 4))
    plt.hist(lengths, bins=20, color="#4f8cff", alpha=0.8)
    plt.title(title, fontweight="bold")
    plt.xlabel("Textlänge (Zeichen)")
    plt.ylabel("Häufigkeit")
    plt.axvline(lengths.mean(), color="red", linestyle="--", label=f"Durchschnitt: {lengths.mean():.0f}")
    plt.legend()
    plt.tight_layout()

def main():
    if not TRAIN_POOL_PATH.exists():
        print("train_pool.csv nicht gefunden. Führe init_data_splits.py aus.")
        return

    df_pool  = pd.read_csv(TRAIN_POOL_PATH)
    df_train = pd.read_csv(TRAIN_PATH) if TRAIN_PATH.exists() else None
    df_test  = pd.read_csv(TEST_PATH)  if TEST_PATH.exists()  else None

    plot_class_distribution(df_pool, "Klassenverteilung – Trainingspool")
    if df_test is not None:
        plot_class_distribution(df_test, "Klassenverteilung – Testset")
    if df_train is not None:
        plot_class_distribution(df_train, "Klassenverteilung – Trainingsset (inkl. synthetisch)")
    plot_text_length(df_pool, "Textlängen – Trainingspool")

    plt.show()

if __name__ == "__main__":
    main()