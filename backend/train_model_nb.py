from __future__ import annotations

from pathlib import Path
import json
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, f1_score

LABELS = ["Hindernis", "Infrastrukturproblem", "Gefahrenstelle", "Positives Feedback", "Spam"]

TRAIN_PATH = Path("data/train_split.csv")
TEST_PATH = Path("data/test.csv")

MODEL_DIR = Path("model")
MODEL_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODEL_DIR / "rideaware_model_nb.joblib"
META_PATH = MODEL_DIR / "meta_nb.json"

MODEL_NAME = "tfidf+naivebayes"
MODEL_VERSION = "1.0"


def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(str)

    # Label sanity check
    unknown = sorted(set(df["label"]) - set(LABELS))
    if unknown:
        raise ValueError(f"Unknown labels in {path}: {unknown}")

    return df


def main():
    if not TRAIN_PATH.exists():
        raise FileNotFoundError(f"Missing {TRAIN_PATH}. Run split_data.py first.")

    df_train = load_csv(TRAIN_PATH)
    print(f"Training dataset: {TRAIN_PATH} ({len(df_train)} rows)")

    # Pipeline with TF-IDF + Naive Bayes
    pipe = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_df=0.95)),
            ("clf", MultinomialNB(alpha=1.0)),  # alpha is smoothing parameter
        ]
    )

    X_train = df_train["text"].tolist()
    y_train = df_train["label"].tolist()

    print("\nTraining Naive Bayes model...")
    pipe.fit(X_train, y_train)

    # Optional evaluation on fixed external test set
    if TEST_PATH.exists():
        df_test = load_csv(TEST_PATH)
        print(f"Test dataset:      {TEST_PATH} ({len(df_test)} rows)")

        X_test = df_test["text"].tolist()
        y_test = df_test["label"].tolist()

        preds = pipe.predict(X_test)
        acc = accuracy_score(y_test, preds)
        f1m = f1_score(y_test, preds, average="macro")

        print("\n=== Evaluation on test.csv ===")
        print(f"Accuracy: {acc:.3f}")
        print(f"Macro-F1: {f1m:.3f}")
        print(classification_report(y_test, preds, labels=LABELS, zero_division=0))
    else:
        print("\nNOTE: data/test.csv not found. Model trained without external evaluation.")

    # Save model
    joblib.dump(pipe, MODEL_PATH)

    # Save metadata
    META_PATH.write_text(
        json.dumps({"model_name": MODEL_NAME, "model_version": MODEL_VERSION}, indent=2),
        encoding="utf-8",
    )

    print(f"\nSaved model to: {MODEL_PATH}")
    print(f"Saved meta  to: {META_PATH}")
    print(f"\n✅ Naive Bayes model trained successfully!")
    print(f"Model: {MODEL_NAME} v{MODEL_VERSION}")


if __name__ == "__main__":
    main()
