"""
RideAware BA2 - Preprocessing

Bereinigt RideAware_Dataset.csv und erstellt RideAware_Dataset_clean.csv.

Usage:
    python preprocess_reports.py
"""

from __future__ import annotations

import re
from pathlib import Path
import pandas as pd

# ─── Konfiguration ────────────────────────────────────────────────────────────

INPUT_PATH  = Path("data/RideAware_Dataset.csv")
OUTPUT_PATH = Path("data/RideAware_Dataset_clean.csv")

LABELS = [
    "Gefahrenstelle",
    "Hindernis",
    "Markierung oder Schild",
    "Ampel",
    "Lückenschluss",
]

# ─── Preprocessing ────────────────────────────────────────────────────────────

def preprocess(text: str, max_sentences: int = 3) -> str:
    if not text or not isinstance(text, str):
        return ""

    # 1. Header entfernen: "Name schrieb am Wochentag, DD. Monat YYYY:"
    text = re.sub(
        r'^.*?schrieb am\s+\w+,?\s+\d{1,2}\.\s+\w+\s+\d{4}\s*:\s*',
        '', text, flags=re.IGNORECASE
    ).strip()

    # 2. Foto/Bild Hinweise in Klammern entfernen
    text = re.sub(
        r'\([^)]*(?:foto|fotos|bild|siehe|photo|pic|image|anhang)[^)]*\)',
        '', text, flags=re.IGNORECASE
    ).strip()

    # 3. URLs entfernen
    text = re.sub(r'https?://\S+|www\.\S+', '', text, flags=re.IGNORECASE).strip()

    # 4. Mehrfache Leerzeichen normalisieren
    text = re.sub(r'\s+', ' ', text).strip()

    # 5. Auf max. 3 Sätze kürzen
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if len(sentences) > max_sentences:
        text = ' '.join(sentences[:max_sentences])

    return text.strip()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("RIDEAWARE BA2 – PREPROCESSING")
    print("=" * 70)

    if not INPUT_PATH.exists():
        print(f"\n❌ Datei nicht gefunden: {INPUT_PATH}")
        print("   Stelle sicher dass data/RideAware_Dataset.csv vorhanden ist.")
        return

    df = pd.read_csv(INPUT_PATH)
    df["text"]  = df["text"].astype(str)
    df["label"] = df["label"].astype(str)

    print(f"\n✓ Datei geladen: {INPUT_PATH} ({len(df)} Meldungen)")
    print(f"\nKlassenverteilung:")
    print(df["label"].value_counts().to_string())

    orig_len = df["text"].str.len().mean()

    print(f"\nPreprocessing läuft...")
    df["text"] = df["text"].apply(preprocess)

    clean_len = df["text"].str.len().mean()

    # Leere Texte entfernen
    empty = df["text"].str.strip() == ""
    if empty.sum() > 0:
        print(f"  ⚠️  {empty.sum()} leere Meldungen entfernt.")
        df = df[~empty].copy()

    print(f"\n  Ø Textlänge vorher:  {orig_len:.0f} Zeichen")
    print(f"  Ø Textlänge nachher: {clean_len:.0f} Zeichen")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df[["text", "label"]].to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

    print(f"\n{'=' * 70}")
    print(f"✅ Fertig! Gespeichert: {OUTPUT_PATH}")
    print(f"\nNächster Schritt:")
    print(f"  python init_data_splits.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
quit