# RideAware – Bachelorarbeit BA2

**Supervised vs. Semi-Supervised Textklassifikation für Fahrradinfrastrukturmeldungen**  
Aleyna Kurtcuoglu | Hochschule Campus Wien | Mai 2026

---

## Projektübersicht

RideAware ist eine Web-Applikation zur automatischen Klassifikation von Fahrradinfrastrukturmeldungen. Nutzer können Meldungen auf einer Karte einreichen, die dann automatisch einer von fünf Kategorien zugeordnet werden:

- **Gefahrenstelle** – gefährliche Verkehrssituationen
- **Hindernis** – physische Hindernisse auf dem Radweg
- **Markierung oder Schild** – fehlende oder falsche Beschilderung
- **Ampel** – Probleme mit Radfahrerampeln
- **Lückenschluss** – fehlende Radwegverbindungen

Das Projekt besteht aus einem **Python-Backend** (FastAPI + ML-Modelle) und einem **React-Frontend**.

---

## Voraussetzungen

- Python 3.12
- Node.js (für das Frontend)
- pip (Python-Paketmanager)

---

## 1. Backend – Setup

```bash
cd backend
python -m venv .venv

# Windows:
.venv\Scripts\activate

# Mac/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

---

## 2. ML-Pipeline – Schritt für Schritt ausführen

Die folgenden Skripte bauen aufeinander auf und müssen **in dieser Reihenfolge** ausgeführt werden.

### Schritt 1 – Daten vorverarbeiten

```bash
python preprocess_reports.py
```

Liest `data/RideAware_Dataset.csv`, bereinigt die Texte (Header, Foto-Hinweise und URLs entfernen, auf maximal 3 Sätze kürzen) und speichert das Ergebnis als `data/RideAware_Dataset_clean.csv`.

### Schritt 2 – Datensatz aufteilen

```bash
python init_data_splits.py
```

Teilt den bereinigten Datensatz stratifiziert auf:
- `data/train_pool.csv` – 75 % Trainingsdaten (225 reale Meldungen)
- `data/test.csv` – 25 % Testdaten (75 reale Meldungen, nur für Evaluation)

### Schritt 3 – Modelle trainieren

**TF-IDF + Logistic Regression:**
```bash
python train_model.py
```
Speichert das Modell als `model/rideaware_model.joblib`.

**TF-IDF + Naive Bayes:**
```bash
python train_model_nb.py
```
Speichert das Modell als `model/rideaware_model_nb.joblib`.

### Schritt 4 – Modelle vergleichen (Experiment 1)

```bash
python compare_models.py
```

Vergleicht Logistic Regression und Naive Bayes auf dem Testset. Gibt Accuracy, Macro-F1, Classification Report und Konfusionsmatrix aus.

### Schritt 5 – Evaluation (Baseline vs. ML)

```bash
python evaluate.py
```

Vergleicht den regelbasierten Baseline-Klassifikator mit dem trainierten ML-Modell.

---

## 3. Semi-Supervised Experimente (Experiment 2 & 3)

### Self-Training mit dem vollständigen Trainingsdatensatz

```bash
python self_training.py
```

### Self-Training mit reduzierter Labelmenge (z. B. 100 Beispiele)

```bash
python self_training.py --labeled 100 --iterations 5 --threshold 0.85
```

### Lernkurven-Experiment über alle Labelmengen (50, 100, 150, 225)

```bash
# Logistic Regression als Basismodell
python self_training.py --experiment --model logreg

# Naive Bayes als Basismodell
python self_training.py --experiment --model nb
```

Die Ergebnisse werden als CSV-Dateien im Ordner `results/` gespeichert.

---

## 4. Visualisierungen erstellen

```bash
# Alle Plots auf einmal
python plot_results.py

# Einzelne Plot-Typen
python plot_results.py --type learning_curve   # Lernkurven Supervised vs. Semi-Supervised
python plot_results.py --type iterations       # F1/Accuracy-Verlauf pro Iteration
python plot_results.py --type bar              # Balkendiagramm Direktvergleich

# Für Naive Bayes Ergebnisse
python plot_results.py --model nb
```

Alle Grafiken werden im Ordner `plots/` gespeichert.

---

## 5. Backend-Server starten (Web-Applikation)

```bash
cd backend
uvicorn main:app --reload
```

Der Server läuft auf `http://127.0.0.1:8000`.  
Die API-Dokumentation ist erreichbar unter `http://127.0.0.1:8000/docs`.

---

## 6. Frontend – Setup und Start

```bash
cd frontend
npm install
npm run dev
```

Das Frontend ist erreichbar unter `http://localhost:5173`.

---

## Dateistruktur

```
RideAware/
├── backend/
│   ├── data/
│   │   ├── RideAware_Dataset.csv         # Rohdaten (Original)
│   │   ├── RideAware_Dataset_clean.csv   # nach Preprocessing
│   │   ├── train_pool.csv                # Trainingspool (225 reale Meldungen)
│   │   ├── train_split.csv               # Training inkl. synthetisch (828 Beispiele)
│   │   ├── test.csv                      # Testset (75 reale Meldungen, fix)
│   │   ├── unlabeled.csv                 # synthetische Daten für Self-Training
│   │   └── synthetic.csv                 # synthetische Trainingsdaten
│   ├── model/
│   │   ├── rideaware_model.joblib         # trainiertes LR-Modell
│   │   └── rideaware_model_nb.joblib      # trainiertes NB-Modell
│   ├── plots/                             # generierte Grafiken (nach plot_results.py)
│   ├── results/                           # Experiment-Ergebnisse (CSV-Dateien)
│   ├── preprocess_reports.py              # Schritt 1: Preprocessing
│   ├── init_data_splits.py                # Schritt 2: Train/Test-Split
│   ├── train_model.py                     # Schritt 3a: LR trainieren
│   ├── train_model_nb.py                  # Schritt 3b: NB trainieren
│   ├── compare_models.py                  # Experiment 1: Modellvergleich
│   ├── evaluate.py                        # Baseline vs. ML-Evaluation
│   ├── self_training.py                   # Experiment 2 & 3: Self-Training
│   ├── plot_results.py                    # Visualisierungen
│   └── ml_classifier.py                   # Klassifikation für die Web-App
└── frontend/
    └── src/
        └── App.tsx                        # React-Hauptkomponente
```

---

## Hinweis zu BERT

Das BERT-Modell (`deepset/gbert-base`) wurde in Google Colab auf einer NVIDIA T4 GPU trainiert und ist nicht lokal im Repository enthalten. Die verwendeten Hyperparameter waren: 3 Epochen, Batch-Größe 16, Lernrate 2×10⁻⁵, maximale Sequenzlänge 128 Tokens. Die Ergebnisse sind in der Bachelorarbeit dokumentiert (Kapitel 5.8).
