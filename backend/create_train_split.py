"""
RideAware BA2 - Trainingsset erstellen

Kombiniert train_pool.csv (echte Daten) + synthetic.csv (synthetische Daten)
zu train_split.csv (finales Trainingsset).

Usage:
    python create_train_split.py
"""

from pathlib import Path
import pandas as pd

TRAIN_POOL_PATH = Path("data/train_pool.csv")
SYNTHETIC_PATH  = Path("data/synthetic.csv")
OUTPUT_PATH     = Path("data/train_split.csv")

LABELS = [
    "Gefahrenstelle",
    "Hindernis",
    "Markierung oder Schild",
    "Ampel",
    "Lückenschluss",
]

def main():
    print("=" * 70)
    print("RIDEAWARE BA2 – TRAININGSSET ERSTELLEN")
    print("=" * 70)

    # ── Trainingspool laden ───────────────────────────────────────────────────
    if not TRAIN_POOL_PATH.exists():
        print(f"\n❌ {TRAIN_POOL_PATH} nicht gefunden.")
        print("   Führe zuerst aus: python init_data_splits.py")
        return

    df_pool = pd.read_csv(TRAIN_POOL_PATH)
    df_pool["text"]  = df_pool["text"].astype(str)
    df_pool["label"] = df_pool["label"].astype(str)
    print(f"\n✓ Trainingspool geladen: {len(df_pool)} echte Meldungen")

    # ── Synthetische Daten laden ──────────────────────────────────────────────
    if not SYNTHETIC_PATH.exists():
        print(f"\n⚠️  {SYNTHETIC_PATH} nicht gefunden.")
        print("   Trainingsset wird nur aus echten Daten erstellt.")
        df_combined = df_pool[["text", "label"]].copy()
    else:
        df_syn = pd.read_csv(SYNTHETIC_PATH)
        df_syn["text"]  = df_syn["text"].astype(str)
        df_syn["label"] = df_syn["label"].astype(str)
        print(f"✓ Synthetische Daten geladen: {len(df_syn)} Meldungen")

        # Kombinieren
        df_combined = pd.concat(
            [df_pool[["text", "label"]], df_syn[["text", "label"]]],
            ignore_index=True
        )
        # Mischen
        df_combined = df_combined.sample(frac=1, random_state=42).reset_index(drop=True)

    # ── Speichern ─────────────────────────────────────────────────────────────
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_combined.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

    print(f"\nKlassenverteilung (train_split):")
    print(df_combined["label"].value_counts().to_string())

    print(f"\n{'=' * 70}")
    print(f"✅ Fertig! Gespeichert: {OUTPUT_PATH} ({len(df_combined)} Meldungen)")
    print(f"   Davon echt:        {len(df_pool)}")
    print(f"   Davon synthetisch: {len(df_combined) - len(df_pool)}")
    print(f"\nNächster Schritt:")
    print(f"  python train_model.py")
    print("=" * 70)

if __name__ == "__main__":
    main()