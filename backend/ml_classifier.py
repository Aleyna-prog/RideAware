from __future__ import annotations

from pathlib import Path
import json
import joblib

LABELS = ["Hindernis", "Infrastrukturproblem", "Gefahrenstelle", "Positives Feedback", "Spam"]

MODEL_DIR = Path("model")
MODEL_PATH = MODEL_DIR / "rideaware_model.joblib"
META_PATH = MODEL_DIR / "meta.json"

_model = None
_meta = None


def load_model():
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
        _model = joblib.load(MODEL_PATH)
    return _model


def load_meta() -> dict:
    global _meta
    if _meta is None:
        if META_PATH.exists():
            _meta = json.loads(META_PATH.read_text(encoding="utf-8"))
        else:
            _meta = {"model_name": "unknown", "model_version": "unknown"}
    return _meta


def clamp01(x: float) -> float:
    try:
        x = float(x)
    except Exception:
        return 0.0
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def normalize_label(label: str) -> str:
    label = str(label)
    if label in LABELS:
        return label
    # safe fallback if the model outputs something unexpected
    return "Infrastrukturproblem"


def classify_text_ml(text: str) -> tuple[str, float]:
    model = load_model()

    pred = model.predict([text])[0]
    label = normalize_label(pred)

    conf = 0.5
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba([text])[0]
        try:
            conf = float(max(probs))
        except Exception:
            conf = 0.5

    return label, clamp01(conf)


def get_model_info() -> tuple[str, str]:
    meta = load_meta()
    return str(meta.get("model_name", "unknown")), str(meta.get("model_version", "unknown"))
