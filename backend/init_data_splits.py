# init_data_splits.py
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

LABELS = ["Hindernis", "Infrastrukturproblem", "Gefahrenstelle", "Positives Feedback", "Spam"]

IN_PATH = Path("data/train.csv")
TRAIN_POOL = Path("data/train_pool.csv")
TEST_PATH = Path("data/test.csv")

TEST_SIZE = 0.25
RANDOM_STATE = 42

def main():
    if not IN_PATH.exists():
        raise FileNotFoundError(f"Missing: {IN_PATH}")

    df = pd.read_csv(IN_PATH)
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(str)

    # Basic label check
    bad = sorted(set(df["label"]) - set(LABELS))
    if bad:
        raise ValueError(f"Unknown labels in {IN_PATH}: {bad}")

    # Create fixed test split once
    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df["label"],
    )

    # Write files
    TRAIN_POOL.parent.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(TRAIN_POOL, index=False, encoding="utf-8")
    test_df.to_csv(TEST_PATH, index=False, encoding="utf-8")

    print(f"Created fixed test set: {TEST_PATH} ({len(test_df)} rows)")
    print(f"Created training pool:  {TRAIN_POOL} ({len(train_df)} rows)")
    print("\nTest label distribution:")
    print(test_df["label"].value_counts())

if __name__ == "__main__":
    main()
