from __future__ import annotations

from pathlib import Path
import pandas as pd

from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix

from ml_classifier import classify_text_ml

DATA_PATH = Path("data/test.csv")
LABELS = ["Hindernis", "Infrastrukturproblem", "Gefahrenstelle", "Positives Feedback", "Spam"]


def classify_text_baseline(text: str) -> tuple[str, float]:
    t = (text or "").lower()

    spam_keywords = [
        "http", "www", ".com", ".net", "buy", "free", "promo", "sale",
        "discount", "offer", "click", "subscribe", "abonnier", "follow",
        "coupon", "deal", "limited", "win money", "get rich"
    ]
    if any(k in t for k in spam_keywords):
        return "Spam", 0.90

    hindernis_keywords = [
        "glas", "scherben", "stein", "felsen", "ast", "baum",
        "hindernis", "blockiert", "müll", "container",
        "obstacle", "debris", "branch", "rock", "blocked"
    ]
    if any(k in t for k in hindernis_keywords):
        return "Hindernis", 0.80

    gefahren_keywords = [
        "gefährlich", "gefahr", "unübersichtlich", "kreuzung",
        "zu schnell", "raser", "unfall", "beinahe", "beinaheunfall",
        "sicht schlecht", "keine sicht", "straße gesperrt",
        "dangerous", "near miss", "almost", "accident", "crash",
        "intersection", "poor visibility", "closed road"
    ]
    if any(k in t for k in gefahren_keywords):
        return "Gefahrenstelle", 0.78

    infrastruktur_keywords = [
        "radweg", "straße", "strasse", "weg", "baustelle",
        "markierung", "beschilderung", "schlagloch", "riss",
        "infrastruktur", "lane", "road", "surface",
        "construction", "markings", "sign"
    ]
    if any(k in t for k in infrastruktur_keywords):
        return "Infrastrukturproblem", 0.75

    positive_keywords = [
        "danke", "super", "gut", "toll", "perfekt", "endlich",
        "great", "nice", "good", "thanks", "love", "awesome"
    ]
    if any(k in t for k in positive_keywords):
        return "Positives Feedback", 0.70

    return "Infrastrukturproblem", 0.55


def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing {DATA_PATH}. Run split_data.py first.")
    df = pd.read_csv(DATA_PATH)
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(str)
    return df


def evaluate(name: str, predict_label_fn, df: pd.DataFrame) -> None:
    y_true = df["label"].tolist()
    y_pred = [predict_label_fn(t) for t in df["text"].tolist()]

    acc = accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, average="macro")

    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)
    print(f"Dataset:   {DATA_PATH} ({len(df)} rows)")
    print(f"Accuracy:  {acc:.3f}")
    print(f"Macro-F1:  {f1m:.3f}\n")

    print("Classification report:")
    print(classification_report(y_true, y_pred, labels=LABELS, zero_division=0))

    print("Confusion matrix (rows=true, cols=pred):")
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    cm_df = pd.DataFrame(cm, index=[f"T:{l}" for l in LABELS], columns=[f"P:{l}" for l in LABELS])
    print(cm_df)


def main():
    df = load_data()

    baseline_pred = lambda text: classify_text_baseline(text)[0]
    ml_pred = lambda text: classify_text_ml(text)[0]

    evaluate("Baseline (rule-based)", baseline_pred, df)
    evaluate("ML (TF-IDF + LogisticRegression)", ml_pred, df)

    print("\nNote:")
    print(f"- Evaluation is performed on: {DATA_PATH}")
    print("- Train your model using train_model.py (trained on train_split.csv).")
    print("- This script compares baseline vs ML on the same test split.")


if __name__ == "__main__":
    main()
