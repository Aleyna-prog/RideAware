"""
RideAware - Baseline vs. ML Evaluation

Vergleicht den regelbasierten Klassifikator mit dem trainierten ML-Modell
auf dem Testdatensatz und gibt Accuracy, Macro-F1 und die Konfusionsmatrix aus.

Aufruf: python main.py
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd

from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix

from ml_classifier import classify_text_ml

DATA_PATH = Path("data/test.csv")
LABELS = ["Gefahrenstelle", "Hindernis", "Markierung oder Schild", "Ampel", "Lückenschluss"]


def classify_text_baseline(text: str) -> tuple[str, float]:
    """
    Regelbasierter Klassifikator basierend auf Schlüsselwortlisten.
    Der Text wird in Kleinbuchstaben umgewandelt und dann gegen vordefinierte
    Keyword-Listen der einzelnen Kategorien geprüft. Die erste Kategorie,
    deren Keywords im Text vorkommen, wird zurückgegeben. Als Fallback wird
    'Gefahrenstelle' mit niedriger Konfidenz zurückgegeben.
    """
    t = (text or "").lower()

    gefahren_keywords = [
        "gefährlich", "gefahr", "unübersichtlich", "kreuzung",
        "zu schnell", "raser", "unfall", "beinahe", "beinaheunfall",
        "sicht schlecht", "keine sicht", "dangerous", "near miss",
        "almost", "accident", "crash", "intersection", "poor visibility",
        "auto kam", "überholt", "abgedrängt", "touchiert", "close pass",
    ]
    if any(k in t for k in gefahren_keywords):
        return "Gefahrenstelle", 0.82

    hindernis_keywords = [
        "glas", "scherben", "stein", "felsen", "ast", "baum",
        "hindernis", "blockiert", "müll", "container", "pfütze",
        "obstacle", "debris", "branch", "rock", "blocked", "pothole",
        "schlagloch", "öl", "nägel", "matsch", "laub",
    ]
    if any(k in t for k in hindernis_keywords):
        return "Hindernis", 0.80

    markierung_keywords = [
        "markierung", "schild", "beschilderung", "sign", "marking",
        "fehlt", "fehlendes schild", "nicht sichtbar", "verdeckt",
        "abbiege", "wegweiser", "tafel",
    ]
    if any(k in t for k in markierung_keywords):
        return "Markierung oder Schild", 0.78

    ampel_keywords = [
        "ampel", "ampelschaltung", "grünphase", "traffic light",
        "lichtanlage", "radampel", "defekt", "zeigt nichts",
        "zu kurz", "signal",
    ]
    if any(k in t for k in ampel_keywords):
        return "Ampel", 0.78

    lueckenschluss_keywords = [
        "endet", "lücke", "fehlende verbindung", "kein radweg",
        "kein anschluss", "unterbrechung", "baustelle", "gesperrt",
        "no bike lane", "missing lane", "radweg endet", "ohne umleitung",
        "lückenschluss",
    ]
    if any(k in t for k in lueckenschluss_keywords):
        return "Lückenschluss", 0.75

    return "Gefahrenstelle", 0.50


def load_data() -> pd.DataFrame:
    """Lädt den Testdatensatz aus data/test.csv und gibt ihn als DataFrame zurück."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing {DATA_PATH}. Run split_data.py first.")
    df = pd.read_csv(DATA_PATH)
    df["text"]  = df["text"].astype(str)
    df["label"] = df["label"].astype(str)
    return df


def evaluate(name: str, predict_label_fn, df: pd.DataFrame) -> None:
    """
    Wertet einen Klassifikator auf dem übergebenen DataFrame aus.
    Gibt Accuracy, Macro-F1, den Classification Report und die
    Konfusionsmatrix in der Konsole aus.
    """
    y_true = df["label"].tolist()
    y_pred = [predict_label_fn(t) for t in df["text"].tolist()]

    acc = accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, average="macro", zero_division=0)

    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)
    print(f"Dataset:   {DATA_PATH} ({len(df)} rows)")
    print(f"Accuracy:  {acc:.3f}")
    print(f"Macro-F1:  {f1m:.3f}\n")

    print("Classification report:")
    print(classification_report(y_true, y_pred, labels=LABELS, zero_division=0))

    print("Confusion matrix (rows=true, cols=pred):")
    cm    = confusion_matrix(y_true, y_pred, labels=LABELS)
    cm_df = pd.DataFrame(cm, index=[f"T:{l}" for l in LABELS], columns=[f"P:{l}" for l in LABELS])
    print(cm_df)


def main():
    """Lädt die Testdaten und vergleicht den Baseline-Klassifikator mit dem ML-Modell."""
    df = load_data()

    baseline_pred = lambda text: classify_text_baseline(text)[0]
    ml_pred       = lambda text: classify_text_ml(text)[0]

    evaluate("Baseline (rule-based)", baseline_pred, df)
    evaluate("ML (TF-IDF + LogisticRegression)", ml_pred, df)

    print("\nNote:")
    print(f"- Evaluation is performed on: {DATA_PATH}")
    print("- Train your model using train_model.py (trained on train_split.csv).")
    print("- This script compares baseline vs ML on the same test split.")


if __name__ == "__main__":
    main()