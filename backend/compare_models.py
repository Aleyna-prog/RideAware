from __future__ import annotations

from pathlib import Path
import pandas as pd
import joblib

from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix

LABELS = ["Gefahrenstelle", "Hindernis", "Markierung oder Schild", "Ampel", "Lückenschluss"]

TEST_PATH    = Path("data/test.csv")
MODEL_LOGREG = Path("model/rideaware_model.joblib")
MODEL_NB     = Path("model/rideaware_model_nb.joblib")


def load_model(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")
    return joblib.load(path)


def evaluate_model(model, X_test, y_test, model_name: str):
    preds = model.predict(X_test)
    acc   = accuracy_score(y_test, preds)
    f1m   = f1_score(y_test, preds, average="macro", zero_division=0)

    print("\n" + "=" * 70)
    print(f"MODEL: {model_name}")
    print("=" * 70)
    print(f"Accuracy:  {acc:.3f}")
    print(f"Macro-F1:  {f1m:.3f}\n")
    print("Classification Report:")
    print(classification_report(y_test, preds, labels=LABELS, zero_division=0))
    print("Confusion Matrix (rows=true, cols=pred):")
    cm    = confusion_matrix(y_test, preds, labels=LABELS)
    cm_df = pd.DataFrame(cm, index=[f"T:{l}" for l in LABELS], columns=[f"P:{l}" for l in LABELS])
    print(cm_df)
    return acc, f1m


def main():
    if not TEST_PATH.exists():
        raise FileNotFoundError(f"Missing {TEST_PATH}. Run init_data_splits.py first.")

    df_test = pd.read_csv(TEST_PATH)
    df_test["text"]  = df_test["text"].astype(str)
    df_test["label"] = df_test["label"].astype(str)

    X_test = df_test["text"].tolist()
    y_test = df_test["label"].tolist()

    print("=" * 70)
    print("RIDEAWARE BA2 - MODEL COMPARISON")
    print("=" * 70)
    print(f"Test dataset: {TEST_PATH} ({len(df_test)} samples)")

    try:
        model_logreg = load_model(MODEL_LOGREG)
        model_nb     = load_model(MODEL_NB)
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\nPlease train both models first:")
        print("  python train_model.py")
        print("  python train_model_nb.py")
        return

    acc_lr, f1_lr = evaluate_model(model_logreg, X_test, y_test, "TF-IDF + Logistic Regression")
    acc_nb, f1_nb = evaluate_model(model_nb,     X_test, y_test, "TF-IDF + Naive Bayes")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Model':<30} {'Accuracy':<12} {'Macro-F1':<12}")
    print("-" * 70)
    print(f"{'TF-IDF + Logistic Regression':<30} {acc_lr:<12.3f} {f1_lr:<12.3f}")
    print(f"{'TF-IDF + Naive Bayes':<30} {acc_nb:<12.3f} {f1_nb:<12.3f}")
    print("-" * 70)

    if f1_lr > f1_nb:
        print(f"🏆 Winner (Macro-F1): Logistic Regression (+{f1_lr - f1_nb:.3f})")
    elif f1_nb > f1_lr:
        print(f"🏆 Winner (Macro-F1): Naive Bayes (+{f1_nb - f1_lr:.3f})")
    else:
        print("🤝 Tie in Macro-F1!")


if __name__ == "__main__":
    main()