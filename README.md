# RideAware

Dein Projekt verwendet jetzt **2 ML-Verfahren** zur Textklassifikation:

1. **TF-IDF + Logistic Regression**
2. **TF-IDF + Naive Bayes** 

---

##  Workflow

# SetUp

# Backend
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python init_data_splits.py
python split_data.py
python train_and_compare.py

# Frontend
cd frontend
npm install

# Daily Usage

# Terminal 1 - Backend
cd backend
.venv\Scripts\activate
uvicorn main:app --reload

# Terminal 2 - Frontend
cd frontend
npm run dev

# Modell wechseln
Das System unterstützt zwei ML-Modelle:
bash# Logistic Regression (Standard)
uvicorn main:app --reload

# Naive Bayes
MODEL_TYPE=naivebayes uvicorn main:app --reload

# Windows PowerShell:
$env:MODEL_TYPE="naivebayes"
uvicorn main:app --reload