"""
RideAware - Combined Training & Evaluation Script

Trains both ML models (Logistic Regression + Naive Bayes) and compares them.
Perfect for BA2 documentation - one command shows everything!

Usage:
    python train_and_compare.py
"""

from __future__ import annotations

from pathlib import Path
import json
import pandas as pd
import joblib
import time

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix

LABELS = ["Hindernis", "Infrastrukturproblem", "Gefahrenstelle", "Positives Feedback", "Spam"]

TRAIN_PATH = Path("data/train_split.csv")
TEST_PATH = Path("data/test.csv")

MODEL_DIR = Path("model")
MODEL_DIR.mkdir(exist_ok=True)

MODEL_LR_PATH = MODEL_DIR / "rideaware_model.joblib"
META_LR_PATH = MODEL_DIR / "meta.json"
MODEL_NB_PATH = MODEL_DIR / "rideaware_model_nb.joblib"
META_NB_PATH = MODEL_DIR / "meta_nb.json"


def print_header(title: str):
    """Print formatted header"""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def load_csv(path: Path) -> pd.DataFrame:
    """Load and validate CSV"""
    df = pd.read_csv(path)
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(str)
    unknown = sorted(set(df["label"]) - set(LABELS))
    if unknown:
        raise ValueError(f"Unknown labels: {unknown}")
    return df


def train_model(name: str, pipeline: Pipeline, X_train, y_train):
    """Train model and return time"""
    print(f"\n  Training {name}...")
    start = time.time()
    pipeline.fit(X_train, y_train)
    elapsed = time.time() - start
    print(f"  ✓ Completed in {elapsed:.2f}s")
    return elapsed


def evaluate_model(name: str, model, X_test, y_test):
    """Evaluate model"""
    print_header(f"{name}")
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    f1m = f1_score(y_test, preds, average="macro")
    
    print(f"Accuracy:  {acc:.3f}")
    print(f"Macro-F1:  {f1m:.3f}\n")
    print(classification_report(y_test, preds, labels=LABELS, zero_division=0))
    
    cm = confusion_matrix(y_test, preds, labels=LABELS)
    cm_df = pd.DataFrame(cm, index=[l[:6] for l in LABELS], columns=[l[:6] for l in LABELS])
    print("\nConfusion Matrix:")
    print(cm_df)
    
    return acc, f1m


def save_model(model, model_path, meta_path, name, version):
    """Save model and metadata"""
    joblib.dump(model, model_path)
    meta_path.write_text(json.dumps({"model_name": name, "model_version": version}, indent=2))


def main():
    print_header("RIDEAWARE - TRAIN & COMPARE BOTH ML MODELS")
    
    # Load data
    print("\n Step 1/5: Loading data...")
    if not TRAIN_PATH.exists():
        print(f"\n❌ Error: {TRAIN_PATH} not found")
        print("Run: python init_data_splits.py && python split_data.py")
        return
    
    df_train = load_csv(TRAIN_PATH)
    print(f"  ✓ Training: {len(df_train)} samples")
    
    if TEST_PATH.exists():
        df_test = load_csv(TEST_PATH)
        print(f"  ✓ Test: {len(df_test)} samples")
        X_test = df_test["text"].tolist()
        y_test = df_test["label"].tolist()
    else:
        print(f"  ⚠️  No test data found")
        df_test = None
    
    X_train = df_train["text"].tolist()
    y_train = df_train["label"].tolist()
    
    # Create pipelines
    print("\n Step 2/5: Creating pipelines...")
    pipe_lr = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_df=0.95)),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
    ])
    pipe_nb = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_df=0.95)),
        ("clf", MultinomialNB(alpha=1.0)),
    ])
    print("  ✓ Logistic Regression pipeline ready")
    print("  ✓ Naive Bayes pipeline ready")
    
    # Train both
    print("\n Step 3/5: Training both models...")
    time_lr = train_model("Logistic Regression", pipe_lr, X_train, y_train)
    time_nb = train_model("Naive Bayes", pipe_nb, X_train, y_train)
    
    # Evaluate
    if df_test is not None:
        print("\n📊 Step 4/5: Evaluating on test set...")
        acc_lr, f1_lr = evaluate_model("TF-IDF + Logistic Regression", pipe_lr, X_test, y_test)
        acc_nb, f1_nb = evaluate_model("TF-IDF + Naive Bayes", pipe_nb, X_test, y_test)
        
        # Comparison
        print_header("COMPARISON")
        print(f"{'Model':<32} {'Acc':<8} {'F1':<8} {'Time':<8}")
        print("-" * 70)
        print(f"{'Logistic Regression':<32} {acc_lr:<8.3f} {f1_lr:<8.3f} {time_lr:<8.2f}s")
        print(f"{'Naive Bayes':<32} {acc_nb:<8.3f} {f1_nb:<8.3f} {time_nb:<8.2f}s")
        print("-" * 70)
        
        if acc_lr > acc_nb:
            print(f"🏆 Best Accuracy: Logistic Regression (+{acc_lr-acc_nb:.3f})")
        elif acc_nb > acc_lr:
            print(f"🏆 Best Accuracy: Naive Bayes (+{acc_nb-acc_lr:.3f})")
        else:
            print("🤝 Tied on Accuracy")
        
        if time_nb < time_lr:
            print(f"⚡ Fastest: Naive Bayes ({time_lr-time_nb:.2f}s faster)")
        else:
            print(f"⚡ Fastest: Logistic Regression ({time_nb-time_lr:.2f}s faster)")
    else:
        print("\n  Step 4/5: Skipping evaluation (no test data)")
    
    # Save
    print("\n Step 5/5: Saving models...")
    save_model(pipe_lr, MODEL_LR_PATH, META_LR_PATH, "tfidf+logreg", "1.0")
    save_model(pipe_nb, MODEL_NB_PATH, META_NB_PATH, "tfidf+naivebayes", "1.0")
    print(f"  ✓ {MODEL_LR_PATH}")
    print(f"  ✓ {MODEL_NB_PATH}")
    
    print("\n" + "=" * 70)
    print("✅ DONE! Both models trained, evaluated, and saved.")
    print("=" * 70)


if __name__ == "__main__":
    main()