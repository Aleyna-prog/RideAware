"""
RideAware BA2 - Unlabeled Data erstellen

Erstellt unlabeled.csv aus synthetic.csv (nur text Spalte, ohne Labels).
Diese Datei wird für das Semi-Supervised Self-Training benötigt.

Usage:
    python create_unlabeled.py
"""

from pathlib import Path
import pandas as pd

SYNTHETIC_PATH = Path("data/synthetic.csv")
OUTPUT_PATH    = Path("data/unlabeled.csv")

def main():
    print("=" * 70)
    print("RIDEAWARE BA2 – UNLABELED DATA ERSTELLEN")
    print("=" * 70)

    if not SYNTHETIC_PATH.exists():
        print(f"\n❌ {SYNTHETIC_PATH} nicht gefunden.")
        print("   Stelle sicher dass data/synthetic.csv vorhanden ist.")
        return

    df = pd.read_csv(SYNTHETIC_PATH)
    df["text"] = df["text"].astype(str)

    # Nur text Spalte – keine Labels!
    df[["text"]].to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

    print(f"\n✓ {len(df)} ungelabelte Texte aus synthetic.csv extrahiert")
    print(f"✅ Gespeichert: {OUTPUT_PATH}")
    print(f"\nNächster Schritt:")
    print(f"  python self_training.py --experiment --model logreg")
    print("=" * 70)

if __name__ == "__main__":
    main()