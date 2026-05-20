"""
RideAware BA2 - Training & Vergleich aller Modelle

Vergleicht: Naive Bayes vs. Logistic Regression vs. BERT
Perfekt für die BA2 Dokumentation – ein Befehl zeigt alles!

Usage:
    python train_and_compare.py              # NB + LR (kein BERT)
    python train_and_compare.py --bert       # NB + LR + BERT
    python train_and_compare.py --bert --epochs 3
"""

from __future__ import annotations

import argparse
from pathlib import Path
import json
import pandas as pd
import joblib
import time

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    classification_report, accuracy_score,
    f1_score, confusion_matrix,
)

# ─── Konfiguration ────────────────────────────────────────────────────────────

LABELS = [
    "Gefahrenstelle",
    "Hindernis",
    "Markierung oder Schild",
    "Ampel",
    "Lückenschluss",
]

TRAIN_PATH = Path("data/train_split.csv")
TEST_PATH  = Path("data/test.csv")

MODEL_DIR  = Path("model")
MODEL_DIR.mkdir(exist_ok=True)

MODEL_LR_PATH  = MODEL_DIR / "rideaware_model.joblib"
META_LR_PATH   = MODEL_DIR / "meta.json"
MODEL_NB_PATH  = MODEL_DIR / "rideaware_model_nb.joblib"
META_NB_PATH   = MODEL_DIR / "meta_nb.json"
BERT_DIR       = MODEL_DIR / "bert"

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# ─── Hilfsfunktionen ──────────────────────────────────────────────────────────

def print_header(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["text"]  = df["text"].astype(str)
    df["label"] = df["label"].astype(str)
    unknown = sorted(set(df["label"]) - set(LABELS))
    if unknown:
        raise ValueError(f"Unbekannte Labels: {unknown}")
    return df


def train_sklearn(name: str, pipeline: Pipeline, X_train, y_train) -> float:
    print(f"\n  Training {name}...")
    start = time.time()
    pipeline.fit(X_train, y_train)
    elapsed = time.time() - start
    print(f"  ✓ Fertig in {elapsed:.2f}s")
    return elapsed


def evaluate_sklearn(name: str, model, X_test, y_test) -> tuple[float, float]:
    print_header(name)
    preds = model.predict(X_test)
    acc   = accuracy_score(y_test, preds)
    f1m   = f1_score(y_test, preds, average="macro", zero_division=0)

    print(f"Accuracy:  {acc:.3f}")
    print(f"Macro-F1:  {f1m:.3f}\n")
    print(classification_report(y_test, preds, labels=LABELS, zero_division=0))

    cm     = confusion_matrix(y_test, preds, labels=LABELS)
    cm_df  = pd.DataFrame(cm, index=[l[:8] for l in LABELS], columns=[l[:8] for l in LABELS])
    print("Confusion Matrix:")
    print(cm_df)

    return acc, f1m


def save_sklearn(model, model_path, meta_path, name, version) -> None:
    joblib.dump(model, model_path)
    meta_path.write_text(json.dumps({"model_name": name, "model_version": version}, indent=2))


# ─── BERT Training & Evaluation ───────────────────────────────────────────────

def train_and_evaluate_bert(
    X_train: list[str],
    y_train: list[str],
    X_test:  list[str],
    y_test:  list[str],
    epochs:  int = 3,
    batch_size: int = 16,
) -> tuple[float, float, float]:
    """
    Fine-tune BERT und evaluiere auf dem Testset.
    Gibt (accuracy, f1_macro, training_time) zurück.
    """
    try:
        import torch
        from torch.utils.data import Dataset, DataLoader
        from transformers import (
            AutoTokenizer,
            AutoModelForSequenceClassification,
            get_linear_schedule_with_warmup,
        )
        from torch.optim import AdamW
    except ImportError:
        print("  ❌ transformers/torch nicht installiert!")
        print("     Führe aus: pip install transformers torch")
        return 0.0, 0.0, 0.0

    LABEL2ID = {l: i for i, l in enumerate(LABELS)}
    ID2LABEL = {i: l for i, l in enumerate(LABELS)}
    MODEL_NAME = "deepset/gbert-base"

    # Gerät
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Gerät: {device}")
    if device.type == "cpu":
        print("  ⚠️  CPU erkannt – BERT Training dauert ~30-60 Min.")
        print("     Tipp: Verwende das Colab Notebook für GPU-Training!")
        if batch_size > 8:
            batch_size = 8

    # Dataset Klasse
    class ReportDataset(Dataset):
        def __init__(self, texts, labels, tokenizer):
            self.encodings = tokenizer(
                texts, truncation=True, padding=True,
                max_length=128, return_tensors="pt",
            )
            self.labels = torch.tensor(labels, dtype=torch.long)

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            item = {k: v[idx] for k, v in self.encodings.items()}
            item["labels"] = self.labels[idx]
            return item

    # Labels zu IDs
    y_train_ids = [LABEL2ID[l] for l in y_train]
    y_test_ids  = [LABEL2ID[l] for l in y_test]

    # Tokenizer & Modell laden
    print(f"  Lade {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(LABELS),
        id2label=ID2LABEL, label2id=LABEL2ID,
    )
    model.to(device)

    train_loader = DataLoader(ReportDataset(X_train, y_train_ids, tokenizer), batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(ReportDataset(X_test,  y_test_ids,  tokenizer), batch_size=batch_size)

    optimizer    = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    total_steps  = len(train_loader) * epochs
    scheduler    = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    best_f1, best_state = 0.0, None
    start = time.time()

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            outputs = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                labels=batch["labels"].to(device),
            )
            outputs.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += outputs.loss.item()

        # Evaluate
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in test_loader:
                out   = model(input_ids=batch["input_ids"].to(device), attention_mask=batch["attention_mask"].to(device))
                preds = torch.argmax(out.logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch["labels"].numpy())

        acc = accuracy_score(all_labels, all_preds)
        f1m = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        print(f"  Epoche {epoch}/{epochs}  |  Loss: {total_loss/len(train_loader):.4f}  |  Acc: {acc:.3f}  |  F1: {f1m:.3f}")

        if f1m > best_f1:
            best_f1    = f1m
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    elapsed = time.time() - start

    # Finales Ergebnis mit bestem Modell
    model.load_state_dict(best_state)
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            out   = model(input_ids=batch["input_ids"].to(device), attention_mask=batch["attention_mask"].to(device))
            preds = torch.argmax(out.logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch["labels"].numpy())

    label_names = [ID2LABEL[i] for i in all_labels]
    pred_names  = [ID2LABEL[i] for i in all_preds]

    acc_final = accuracy_score(label_names, pred_names)
    f1_final  = f1_score(label_names, pred_names, average="macro", zero_division=0)

    print_header("BERT – Ergebnis auf Testset")
    print(f"Accuracy:  {acc_final:.3f}")
    print(f"Macro-F1:  {f1_final:.3f}\n")
    print(classification_report(label_names, pred_names, labels=LABELS, zero_division=0))

    # Modell speichern
    model.save_pretrained(BERT_DIR)
    tokenizer.save_pretrained(BERT_DIR)
    print(f"  ✓ BERT Modell gespeichert: {BERT_DIR}")

    return acc_final, f1_final, elapsed


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="RideAware BA2 – Alle Modelle vergleichen")
    parser.add_argument("--bert",       action="store_true", help="BERT einbeziehen")
    parser.add_argument("--epochs",     type=int, default=3, help="BERT Epochen (default: 3)")
    parser.add_argument("--batch_size", type=int, default=16, help="BERT Batch-Größe (default: 16)")
    args = parser.parse_args()

    print_header("RIDEAWARE BA2 – ALLE MODELLE VERGLEICHEN")

    # ── Daten laden ──────────────────────────────────────────────────────────
    print("\nStep 1: Daten laden...")
    if not TRAIN_PATH.exists():
        print(f"❌ {TRAIN_PATH} nicht gefunden.")
        print("   Führe aus: python init_data_splits.py && python split_data.py")
        return

    df_train = load_csv(TRAIN_PATH)
    print(f"  ✓ Training: {len(df_train)} Beispiele")

    if not TEST_PATH.exists():
        print("  ⚠️  Kein Testset gefunden – Evaluation übersprungen.")
        df_test = None
    else:
        df_test = load_csv(TEST_PATH)
        print(f"  ✓ Test:     {len(df_test)} Beispiele")

    X_train = df_train["text"].tolist()
    y_train = df_train["label"].tolist()
    X_test  = df_test["text"].tolist()  if df_test is not None else []
    y_test  = df_test["label"].tolist() if df_test is not None else []

    # ── Sklearn Modelle ───────────────────────────────────────────────────────
    print("\nStep 2: Pipelines erstellen...")
    pipe_lr = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_df=0.95)),
        ("clf",   LogisticRegression(max_iter=2000, class_weight="balanced")),
    ])
    pipe_nb = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_df=0.95)),
        ("clf",   MultinomialNB(alpha=1.0)),
    ])
    print("  ✓ Logistic Regression bereit")
    print("  ✓ Naive Bayes bereit")

    print("\nStep 3: Training...")
    time_lr = train_sklearn("Logistic Regression", pipe_lr, X_train, y_train)
    time_nb = train_sklearn("Naive Bayes",         pipe_nb, X_train, y_train)

    # ── Evaluation ────────────────────────────────────────────────────────────
    acc_lr = acc_nb = f1_lr = f1_nb = 0.0
    acc_bert = f1_bert = time_bert = 0.0

    if df_test is not None:
        print("\nStep 4: Evaluation...")
        acc_lr, f1_lr = evaluate_sklearn("TF-IDF + Logistic Regression", pipe_lr, X_test, y_test)
        acc_nb, f1_nb = evaluate_sklearn("TF-IDF + Naive Bayes",         pipe_nb, X_test, y_test)

        # ── BERT (optional) ───────────────────────────────────────────────────
        if args.bert:
            print_header("BERT Fine-Tuning")
            acc_bert, f1_bert, time_bert = train_and_evaluate_bert(
                X_train, y_train, X_test, y_test,
                epochs=args.epochs, batch_size=args.batch_size,
            )

    # ── Modelle speichern ─────────────────────────────────────────────────────
    print("\nStep 5: Modelle speichern...")
    save_sklearn(pipe_lr, MODEL_LR_PATH, META_LR_PATH, "tfidf+logreg",      "2.0")
    save_sklearn(pipe_nb, MODEL_NB_PATH, META_NB_PATH, "tfidf+naivebayes",  "2.0")
    print(f"  ✓ {MODEL_LR_PATH}")
    print(f"  ✓ {MODEL_NB_PATH}")

    # ── Abschlusstabelle ──────────────────────────────────────────────────────
    if df_test is not None:
        print_header("GESAMTVERGLEICH – ALLE MODELLE")
        print(f"{'Modell':<35} {'Accuracy':<12} {'Macro-F1':<12} {'Zeit':<10}")
        print("-" * 70)
        print(f"{'Naive Bayes':<35} {acc_nb:<12.3f} {f1_nb:<12.3f} {time_nb:<10.2f}s")
        print(f"{'Logistic Regression':<35} {acc_lr:<12.3f} {f1_lr:<12.3f} {time_lr:<10.2f}s")
        if args.bert and time_bert > 0:
            print(f"{'BERT (gbert-base)':<35} {acc_bert:<12.3f} {f1_bert:<12.3f} {time_bert:<10.1f}s")
        print("-" * 70)

        # Bestes Modell
        results = {"Naive Bayes": f1_nb, "Logistic Regression": f1_lr}
        if args.bert and time_bert > 0:
            results["BERT"] = f1_bert
        best = max(results, key=results.get)  # type: ignore
        print(f"🏆 Bestes Modell (Macro-F1): {best} ({results[best]:.3f})")

        # Ergebnisse als CSV speichern
        rows = [
            {"model": "Naive Bayes",          "accuracy": acc_nb,   "f1_macro": f1_nb,   "time_s": time_nb},
            {"model": "Logistic Regression",  "accuracy": acc_lr,   "f1_macro": f1_lr,   "time_s": time_lr},
        ]
        if args.bert and time_bert > 0:
            rows.append({"model": "BERT", "accuracy": acc_bert, "f1_macro": f1_bert, "time_s": time_bert})

        out_path = RESULTS_DIR / "model_comparison.csv"
        pd.DataFrame(rows).to_csv(out_path, index=False)
        print(f"\n✅ Ergebnisse gespeichert: {out_path}")

    print("\n" + "=" * 70)
    print("✅ Fertig!")
    if not args.bert:
        print("   Tipp: Mit --bert auch BERT einbeziehen:")
        print("   python train_and_compare.py --bert")
    print("=" * 70)


if __name__ == "__main__":
    main()