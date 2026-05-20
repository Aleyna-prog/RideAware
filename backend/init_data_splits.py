"""
RideAware BA2 - Datensatz aufteilen

Liest RideAware_Dataset_clean.csv und erstellt:
  - data/train_pool.csv  (~75% fuer Training)
  - data/test.csv        (~25% fuer Evaluation, nur echte Daten!)

Usage:
    python init_data_splits.py
"""

from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

LABELS = ["Gefahrenstelle", "Hindernis", "Markierung oder Schild", "Ampel", "Lueckenschluss"]

IN_PATH    = Path("data/RideAware_Dataset_clean.csv")
TRAIN_POOL = Path("data/train_pool.csv")
TEST_PATH  = Path("data/test.csv")

TEST_SIZE    = 0.25
RANDOM_STATE = 42


def main():
    if not IN_PATH.exists():
        raise FileNotFoundError(
            f"Datei nicht gefunden: {IN_PATH}\n"
            "Fuehre zuerst aus: python preprocess_reports.py --file data/RideAware_Dataset.csv"
        )

    df = pd.read_csv(IN_PATH)
    df["text"]  = df["text"].astype(str)
    df["label"] = df["label"].astype(str)

    # Lückenschluss fix fuer CSV encoding
    LABELS_CHECK = ["Gefahrenstelle", "Hindernis", "Markierung oder Schild", "Ampel", "L\u00fcckenschluss"]
    bad = sorted(set(df["label"]) - set(LABELS_CHECK))
    if bad:
        raise ValueError(f"Unbekannte Labels: {bad}")

    print(f"Datensatz geladen: {len(df)} Meldungen")
    print("\nKlassenverteilung gesamt:")
    print(df["label"].value_counts())

    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df["label"],
    )

    TRAIN_POOL.parent.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(TRAIN_POOL, index=False, encoding="utf-8")
    test_df.to_csv(TEST_PATH,   index=False, encoding="utf-8")

    print(f"\nTrainingspool: {TRAIN_POOL} ({len(train_df)} Meldungen)")
    print(f"Testset:       {TEST_PATH}  ({len(test_df)} Meldungen)")
    print("\nTestset Klassenverteilung:")
    print(test_df["label"].value_counts())
    print("\nWichtig: Das Testset enthaelt NUR echte Meldungen!")
    print("Synthetische Daten kommen nur ins Trainingsset.")


if __name__ == "__main__":
    main()