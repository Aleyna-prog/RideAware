from __future__ import annotations

from pathlib import Path
import pandas as pd

from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix

from ml_classifier import classify_text_ml

DATA_PATH = Path("data/test.csv")
LABELS = ["Gefahrenstelle", "Hindernis", "Markierung oder Schild", "Ampel", "Lückenschluss"]


def classify_text_baseline(text: str) -> tuple[str, float]:
    t = (text or "").lower()

    gefahren_keywords = [
        "gefährlich", "gefahr", "unübersichtlich", "kreuzung",
        "zu schnell", "unfall", "beinahe", "keine sicht",
        "dangerous", "near miss", "almost", "accident",
        "überholt", "abgedrängt", "close pass",
    ]
    if any(k in t for k in gefahren_keywords):
        return "Gefahrenstelle", 0.82

    hindernis_keywords = [
        "glas", "scherben", "stein", "ast", "baum",
        "hindernis", "blockiert", "müll", "pfütze",
        "obstacle", "debris", "branch", "blocked", "pothole", "schlagloch",
    ]
    if any(k in t for k in hindernis_keywords):
        return "Hindernis", 0.80

    markierung_keywords = [
        "markierung", "schild", "beschilderung", "sign", "marking",
        "fehlt", "nicht sichtbar", "verdeckt", "wegweiser", "tafel",
    ]
    if any(k in t for k in markierung_keywords):
        return "Markierung oder Schild", 0.78

    ampel_keywords = [
        "ampel", "ampelschaltung", "grünphase", "traffic light",
        "radampel", "defekt", "zeigt nichts", "zu kurz", "signal",
    ]
    if any(k in t for k in ampel_keywords):
        return "Ampel", 0.78

    luecke_keywords = [
        "endet", "lücke", "fehlende verbindung", "kein radweg",
        "kein anschluss", "radweg endet", "lückenschluss",
    ]
    if any(k in t for k in luecke_keywords):
        return "Lückenschluss", 0.75

    return "Gefahrenstelle", 0.50


def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing {DATA_PATH}. Run init_data_splits.py first.")
    df = pd.read_csv(DATA_PATH)
    df["text"]  = df["text"].astype(str)
    df["label"] = df["label"].astype(str)
    return df


def evaluate(name: str, predict_label_fn, df: pd.DataFrame) -> None:
    y_true = df["label"].tolist()
    y_pred = [predict_label_fn(t) for t in df["text"].tolist()]

    acc = accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, average="macro", zero_division=0)

    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)
    print(f"Accuracy:  {acc:.3f}")
    print(f"Macro-F1:  {f1m:.3f}\n")
    print(classification_report(y_true, y_pred, labels=LABELS, zero_division=0))
    cm    = confusion_matrix(y_true, y_pred, labels=LABELS)
    cm_df = pd.DataFrame(cm, index=[f"T:{l}" for l in LABELS], columns=[f"P:{l}" for l in LABELS])
    print(cm_df)


def main():
    df = load_data()
    baseline_pred = lambda text: classify_text_baseline(text)[0]
    ml_pred       = lambda text: classify_text_ml(text)[0]
    evaluate("Baseline (rule-based)", baseline_pred, df)
    evaluate("ML (TF-IDF + LogisticRegression)", ml_pred, df)


if __name__ == "__main__":
    main()