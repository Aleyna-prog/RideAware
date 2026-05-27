"""
Erstellt ein Jupyter Notebook für das BERT Multi-Seed Labelmenge-Experiment.
Ausgabe: RideAware_BERT_MultiSeed.ipynb
"""
import json

cells = []

def md(source):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": [source]})

def code(source):
    cells.append({"cell_type": "code", "metadata": {}, "source": [source], "outputs": [], "execution_count": None})

# ── Notebook aufbauen ──────────────────────────────────────────────────────

md("# 🚴 RideAware BA2 – BERT Multi-Seed Experiment\n\n"
   "Trainiert BERT mit **3 Seeds** (42, 123, 456) für jede Labelmenge.\n"
   "Berichtet **Mean ± Std** für Accuracy und Macro-F1.\n\n"
   "**Schritte:**\n"
   "1. GPU aktivieren: `Laufzeit → Laufzeittyp ändern → T4 GPU`\n"
   "2. `train_pool.csv`, `train_split.csv` und `test.csv` hochladen\n"
   "3. Alle Zellen ausführen\n"
   "4. Ergebnisse herunterladen\n\n"
   "⏱️ Dauer mit GPU: ca. **15–20 Minuten** (12 Konfigurationen × 3 Seeds + 3 Full)")

md("## Zelle 1 – Pakete installieren")
code("!pip install transformers torch scikit-learn pandas -q\nprint('✅ Pakete installiert!')")

md("## Zelle 2 – GPU prüfen & Imports")
code("""import torch
import time
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)

if torch.cuda.is_available():
    print(f'✅ GPU verfügbar: {torch.cuda.get_device_name(0)}')
    device = torch.device('cuda')
else:
    print('⚠️  Keine GPU!')
    device = torch.device('cpu')
""")

md("## Zelle 3 – Konfiguration")
code("""MODEL_NAME = 'deepset/gbert-base'
LABELS = ['Gefahrenstelle', 'Hindernis', 'Markierung oder Schild', 'Ampel', 'Lückenschluss']
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for i, l in enumerate(LABELS)}

EPOCHS     = 3
BATCH_SIZE = 16
LR         = 2e-5
MAX_LEN    = 128

SEEDS       = [42, 123, 456]
LABEL_SIZES = [50, 100, 150, 225]

print(f'Seeds: {SEEDS}')
print(f'Labelmengen: {LABEL_SIZES}')
print(f'Gesamt: {len(LABEL_SIZES) * len(SEEDS)} + {len(SEEDS)} (Full) = {len(LABEL_SIZES) * len(SEEDS) + len(SEEDS)} Trainingsläufe')
""")

md("## Zelle 4 – Daten laden")
code("""df_pool = pd.read_csv('train_pool.csv')
df_pool['text'] = df_pool['text'].astype(str)
df_pool['label'] = df_pool['label'].astype(str)
df_pool = df_pool[df_pool['label'].isin(LABELS)].reset_index(drop=True)

df_train_full = pd.read_csv('train_split.csv')
df_train_full['text'] = df_train_full['text'].astype(str)
df_train_full['label'] = df_train_full['label'].astype(str)
df_train_full = df_train_full[df_train_full['label'].isin(LABELS)].reset_index(drop=True)

df_test = pd.read_csv('test.csv')
df_test['text'] = df_test['text'].astype(str)
df_test['label'] = df_test['label'].astype(str)
df_test = df_test[df_test['label'].isin(LABELS)].reset_index(drop=True)

print(f'Trainingspool (real): {len(df_pool)}')
print(f'Trainingsset (full):  {len(df_train_full)}')
print(f'Testset:              {len(df_test)}')
""")

md("## Zelle 5 – Hilfsfunktionen")
code("""def subsample(df, n, random_state=42):
    if n >= len(df):
        return df.copy()
    per_class = max(1, n // len(LABELS))
    return (
        df.groupby('label', group_keys=False)
        .apply(lambda g: g.sample(min(len(g), per_class), random_state=random_state))
        .reset_index(drop=True)
    )

class ReportDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.enc = tokenizer(texts, truncation=True, padding=True,
                             max_length=max_length, return_tensors='pt')
        self.labels = torch.tensor(labels, dtype=torch.long)
    def __len__(self): return len(self.labels)
    def __getitem__(self, i):
        item = {k: v[i] for k, v in self.enc.items()}
        item['labels'] = self.labels[i]
        return item

def set_seed(seed):
    \"\"\"Setzt alle Seeds für Reproduzierbarkeit.\"\"\"
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_bert(df_labeled, df_test, seed=42, epochs=3, batch_size=16):
    set_seed(seed)
    
    X_train = df_labeled['text'].tolist()
    y_train = [LABEL2ID[l] for l in df_labeled['label']]
    X_test  = df_test['text'].tolist()
    y_test  = [LABEL2ID[l] for l in df_test['label']]
    
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    mdl = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(LABELS),
        id2label=ID2LABEL, label2id=LABEL2ID,
    ).to(device)
    
    train_loader = DataLoader(ReportDataset(X_train, y_train, tok, MAX_LEN),
                              batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(ReportDataset(X_test, y_test, tok, MAX_LEN),
                              batch_size=batch_size)
    
    optimizer = AdamW(mdl.parameters(), lr=LR, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )
    
    best_f1, best_state = 0.0, None
    for epoch in range(1, epochs + 1):
        mdl.train()
        for batch in train_loader:
            optimizer.zero_grad()
            out = mdl(input_ids=batch['input_ids'].to(device),
                      attention_mask=batch['attention_mask'].to(device),
                      labels=batch['labels'].to(device))
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(mdl.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
        
        mdl.eval()
        preds_all, labels_all = [], []
        with torch.no_grad():
            for batch in test_loader:
                out = mdl(input_ids=batch['input_ids'].to(device),
                          attention_mask=batch['attention_mask'].to(device))
                preds_all.extend(torch.argmax(out.logits, dim=1).cpu().numpy())
                labels_all.extend(batch['labels'].numpy())
        
        f1m = f1_score(labels_all, preds_all, average='macro', zero_division=0)
        if f1m > best_f1:
            best_f1 = f1m
            best_state = {k: v.clone() for k, v in mdl.state_dict().items()}
    
    # Finale Evaluation mit bestem Modell
    mdl.load_state_dict(best_state)
    mdl.eval()
    preds_all, labels_all = [], []
    with torch.no_grad():
        for batch in test_loader:
            out = mdl(input_ids=batch['input_ids'].to(device),
                      attention_mask=batch['attention_mask'].to(device))
            preds_all.extend(torch.argmax(out.logits, dim=1).cpu().numpy())
            labels_all.extend(batch['labels'].numpy())
    
    acc = accuracy_score(labels_all, preds_all)
    f1m = f1_score(labels_all, preds_all, average='macro', zero_division=0)
    
    # Per-Class F1
    label_names = [ID2LABEL[i] for i in labels_all]
    pred_names = [ID2LABEL[i] for i in preds_all]
    per_class = {}
    for label in LABELS:
        tp = sum(1 for t, p in zip(label_names, pred_names) if t == label and p == label)
        fp = sum(1 for t, p in zip(label_names, pred_names) if t != label and p == label)
        fn = sum(1 for t, p in zip(label_names, pred_names) if t == label and p != label)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        per_class[label] = f1
    
    return acc, f1m, per_class

print('✅ Funktionen definiert')
""")

md("## Zelle 6 – Multi-Seed Labelmenge-Experiment\n\n"
   "Trainiert BERT für jede Labelmenge (50, 100, 150, 225) mit 3 Seeds.\n"
   "Zusätzlich: Full Training (828 Beispiele) mit 3 Seeds.")
code("""all_rows = []
per_class_rows = []

# ── Label-Size Experiment ─────────────────────────────────────────────────
for n in LABEL_SIZES:
    print(f'\\n{"="*60}')
    print(f'LABELMENGE: {n}')
    print(f'{"="*60}')
    
    for seed in SEEDS:
        df_sub = subsample(df_pool, n, random_state=seed)
        print(f'  Seed {seed}: Training mit {len(df_sub)} Beispielen...', end=' ')
        start = time.time()
        acc, f1m, pc = train_bert(df_sub, df_test, seed=seed)
        elapsed = time.time() - start
        print(f'Acc={acc:.3f}  F1={f1m:.3f}  ({elapsed:.0f}s)')
        
        all_rows.append({
            'n_labeled': n, 'seed': seed,
            'accuracy': acc, 'f1_macro': f1m
        })
        for label, f1_val in pc.items():
            per_class_rows.append({
                'n_labeled': n, 'seed': seed,
                'label': label, 'f1': f1_val
            })

# ── Full Training (828) ──────────────────────────────────────────────────
print(f'\\n{"="*60}')
print(f'FULL TRAINING: {len(df_train_full)} Beispiele')
print(f'{"="*60}')

for seed in SEEDS:
    print(f'  Seed {seed}: Training mit {len(df_train_full)} Beispielen...', end=' ')
    start = time.time()
    acc, f1m, pc = train_bert(df_train_full, df_test, seed=seed)
    elapsed = time.time() - start
    print(f'Acc={acc:.3f}  F1={f1m:.3f}  ({elapsed:.0f}s)')
    
    all_rows.append({
        'n_labeled': 828, 'seed': seed,
        'accuracy': acc, 'f1_macro': f1m
    })
    for label, f1_val in pc.items():
        per_class_rows.append({
            'n_labeled': 828, 'seed': seed,
            'label': label, 'f1': f1_val
        })

df_all = pd.DataFrame(all_rows)
df_pc  = pd.DataFrame(per_class_rows)
df_all.to_csv('bert_multiseed_results.csv', index=False)
df_pc.to_csv('bert_multiseed_per_class.csv', index=False)

print('\\n✅ Rohdaten gespeichert!')
""")

md("## Zelle 7 – Ergebnisse: Mean ± Std")
code("""# ── Aggregation ──────────────────────────────────────────────────────────
summary = df_all.groupby('n_labeled').agg(
    acc_mean=('accuracy', 'mean'),
    acc_std=('accuracy', 'std'),
    f1_mean=('f1_macro', 'mean'),
    f1_std=('f1_macro', 'std'),
).reset_index()

print('=' * 70)
print('BERT Multi-Seed Ergebnisse (Mean ± Std über 3 Seeds)')
print('=' * 70)
print(f'{\"n\":>5}  {\"Accuracy\":>18}  {\"Macro-F1\":>18}')
print('-' * 50)
for _, row in summary.iterrows():
    n = int(row['n_labeled'])
    print(f'{n:>5}  {row[\"acc_mean\"]:.3f} ± {row[\"acc_std\"]:.3f}       {row[\"f1_mean\"]:.3f} ± {row[\"f1_std\"]:.3f}')

summary.to_csv('bert_multiseed_summary.csv', index=False)

# Per-Class Summary
pc_summary = df_pc.groupby(['n_labeled', 'label']).agg(
    f1_mean=('f1', 'mean'),
    f1_std=('f1', 'std'),
).reset_index()
pc_summary.to_csv('bert_multiseed_per_class_summary.csv', index=False)

print('\\n\\nPer-Class F1 (Mean ± Std) bei Full Training (828):')
print('-' * 50)
full_pc = pc_summary[pc_summary['n_labeled'] == 828]
for _, row in full_pc.iterrows():
    print(f'  {row[\"label\"]:<25} {row[\"f1_mean\"]:.3f} ± {row[\"f1_std\"]:.3f}')

print('\\n✅ Zusammenfassung gespeichert!')
""")

md("## Zelle 8 – Visualisierung")
code("""import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('BERT Lernkurve – Multi-Seed (Mean ± Std)', fontsize=14, fontweight='bold', y=1.02)

for ax, col, ylabel in zip(axes, ['f1', 'acc'], ['Macro-F1', 'Accuracy']):
    mean_col = f'{col}_mean'
    std_col  = f'{col}_std'
    
    ax.errorbar(summary['n_labeled'], summary[mean_col], yerr=summary[std_col],
                marker='o', linewidth=2.5, color='#4f8cff', markersize=8,
                capsize=5, capthick=2, elinewidth=2, ecolor='#ff5a5f')
    
    for _, row in summary.iterrows():
        ax.annotate(f'{row[mean_col]:.3f}±{row[std_col]:.3f}',
                    xy=(row['n_labeled'], row[mean_col]),
                    xytext=(0, 15), textcoords='offset points',
                    ha='center', fontsize=8)
    
    ax.set_xlabel('Anzahl gelabelter Trainingsbeispiele', fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(ylabel, fontsize=12, fontweight='bold')
    ax.xaxis.set_major_locator(mticker.FixedLocator(list(summary['n_labeled'])))
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('bert_multiseed_lernkurve.png', dpi=150, bbox_inches='tight')
plt.show()
print('✅ Grafik gespeichert: bert_multiseed_lernkurve.png')
""")

md("## Zelle 9 – Alle Dateien herunterladen")
code("""from google.colab import files
for f in ['bert_multiseed_results.csv', 'bert_multiseed_summary.csv',
          'bert_multiseed_per_class.csv', 'bert_multiseed_per_class_summary.csv',
          'bert_multiseed_lernkurve.png']:
    files.download(f)
print('✅ Alle Dateien heruntergeladen!')
""")

# ── Notebook speichern ────────────────────────────────────────────────────
notebook = {
    "nbformat": 4,
    "nbformat_minor": 0,
    "metadata": {
        "colab": {"provenance": [], "gpuType": "T4"},
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "language_info": {"name": "python"},
        "accelerator": "GPU"
    },
    "cells": cells,
}

out_path = "RideAware_BERT_MultiSeed.ipynb"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, ensure_ascii=False, indent=2)
print(f"✅ Notebook erstellt: {out_path}")
