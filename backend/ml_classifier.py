"""
RideAware BA2 - ML Classifier

Lädt das trainierte Modell und klassifiziert neue Texte.
Preprocessing wird automatisch angewendet bevor das Modell aufgerufen wird.
"""

from __future__ import annotations

import re
from pathlib import Path
import joblib

LABELS = ["Gefahrenstelle", "Hindernis", "Markierung oder Schild", "Ampel", "Lückenschluss"]

MODEL_PATH = Path("model/rideaware_model.joblib")


# ─── Preprocessing ────────────────────────────────────────────────────────────

def preprocess(text: str, max_sentences: int = 3) -> str:
    """
    Bereinigt einen Meldungstext bevor er klassifiziert wird.
    - Entfernt 'Name schrieb am Datum:' Header
    - Entfernt Foto/Bild Hinweise in Klammern
    - Entfernt URLs
    - Kürzt auf max. 3 Sätze
    """
    if not text or not isinstance(text, str):
        return ""

    text = re.sub(
        r'^.*?schrieb am\s+\w+,?\s+\d{1,2}\.\s+\w+\s+\d{4}\s*:\s*',
        '', text, flags=re.IGNORECASE
    ).strip()

    text = re.sub(
        r'\([^)]*(?:foto|bild|siehe|photo|pic|image|anhang)[^)]*\)',
        '', text, flags=re.IGNORECASE
    ).strip()

    text = re.sub(r'https?://\S+|www\.\S+', '', text, flags=re.IGNORECASE).strip()
    text = re.sub(r'\s+', ' ', text).strip()

    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if len(sentences) > max_sentences:
        text = ' '.join(sentences[:max_sentences])

    return text.strip()


# ─── Klassifizierung ──────────────────────────────────────────────────────────

def classify_text_ml(text: str) -> tuple[str, float]:
    """
    Klassifiziert einen Text mit dem trainierten ML Modell.
    Preprocessing wird automatisch angewendet.

    Returns:
        (label, confidence) Tupel
    """
    cleaned = preprocess(text)
    if not cleaned:
        return "Gefahrenstelle", 0.0

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Kein Modell gefunden: {MODEL_PATH}\n"
            "Führe zuerst aus: python train_model.py"
        )

    model = joblib.load(MODEL_PATH)
    pred  = model.predict([cleaned])[0]
    proba = model.predict_proba([cleaned])[0]
    conf  = float(proba.max())

    return pred, conf


# ─── Test ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_texts = [
        "Kasimir S. schrieb am Donnerstag, 25. März 2021: Schlagloch auf dem Radweg.",
        "Gefährliche Kreuzung ohne Sicht (siehe Foto).",
        "Ampel für Radfahrer defekt seit 2 Wochen.",
        "Radwegmarkierung fehlt komplett.",
        "Radweg endet plötzlich ohne Anschluss.",
    ]

    print("Preprocessing + Klassifizierung Test:\n")
    for text in test_texts:
        cleaned = preprocess(text)
        print(f"  Original:  {text[:60]}...")
        print(f"  Bereinigt: {cleaned[:60]}")
        try:
            label, conf = classify_text_ml(text)
            print(f"  Ergebnis:  {label} ({conf:.2f})")
        except FileNotFoundError:
            print(f"  Ergebnis:  (Modell noch nicht trainiert)")
        print()