from pathlib import Path
import pandas as pd

# Option B (FINAL):
# - train_pool.csv = wächst mit neuen Daten
# - test.csv       = FIX (wird hier NICHT erstellt oder geändert)
# - train_split.csv = nutzt 100 % des Pools fürs Training

POOL_PATH = Path("data/train_pool.csv")
OUT_TRAIN = Path("data/train_split.csv")

def main():
    if not POOL_PATH.exists():
        raise FileNotFoundError(
            f"{POOL_PATH} not found. "
            f"Make sure you have a train_pool.csv with training data."
        )

    df = pd.read_csv(POOL_PATH)
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(str)

    # Use the FULL training pool (no internal split)
    df.to_csv(OUT_TRAIN, index=False, encoding="utf-8")

    print(f"Wrote train_split: {OUT_TRAIN} ({len(df)} rows)")
    print("\nLabel distribution (train_split):")
    print(df["label"].value_counts())

if __name__ == "__main__":
    main()
