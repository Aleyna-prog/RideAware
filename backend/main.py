from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Tuple

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy import Boolean, Column, DateTime, Float, Integer, String, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from ml_classifier import classify_text_ml, get_model_info

LABELS = ["Hindernis", "Infrastrukturproblem", "Gefahrenstelle", "Positives Feedback", "Spam"]

# -------------------------
# Database (SQLite)
# -------------------------
engine = create_engine("sqlite:///./rideaware.db", connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()


class ReportDB(Base):
    __tablename__ = "reports"

    id = Column(Integer, primary_key=True, index=True)
    text = Column(String, nullable=False)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    timestamp = Column(DateTime, nullable=False)

    category = Column(String, nullable=False)
    confidence = Column(Float, nullable=False)

    # SRS S1 (SHOULD): store model info + corrected flag
    model_name = Column(String, nullable=False, default="unknown")
    model_version = Column(String, nullable=False, default="unknown")
    is_corrected = Column(Boolean, nullable=False, default=False)

    source = Column(String, nullable=False, default="real")


Base.metadata.create_all(bind=engine)

# -------------------------
# Baseline classifier (keyword rules) = Benchmark/Fallback
# -------------------------
def classify_text_baseline(text: str) -> Tuple[str, float]:
    t = (text or "").lower()

    spam_keywords = ["http", "www", ".com", ".net", "buy", "free", "promo", "sale", "discount"]
    if any(k in t for k in spam_keywords):
        return "Spam", 0.90

    if any(k in t for k in ["loch", "schlagloch", "glas", "hindernis", "stein", "scherben", "ast", "debris"]):
        return "Hindernis", 0.80
    if any(k in t for k in ["radweg", "infrastruktur", "baustelle", "markierung", "schlecht", "sign", "markings"]):
        return "Infrastrukturproblem", 0.75
    if any(k in t for k in ["gefährlich", "kreuzung", "zu schnell", "beinahe", "unfall", "near miss", "close pass"]):
        return "Gefahrenstelle", 0.78
    if any(k in t for k in ["danke", "super", "gut", "toll", "great", "love", "nice", "thanks"]):
        return "Positives Feedback", 0.70

    return "Infrastrukturproblem", 0.55


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
    return label if label in LABELS else "Infrastrukturproblem"


# -------------------------
# Main classifier (SRS: call ML model) with safe fallback
# -------------------------
def classify_text(text: str) -> Tuple[str, float, str, str]:
    """
    returns: (category, confidence, model_name, model_version)
    """
    try:
        category, confidence = classify_text_ml(text)
        category = normalize_label(category)
        confidence = clamp01(confidence)
        model_name, model_version = get_model_info()
        return category, confidence, model_name, model_version
    except Exception:
        # fallback keeps system usable even without trained model
        category, confidence = classify_text_baseline(text)
        return normalize_label(category), clamp01(confidence), "baseline", "1.0"


# -------------------------
# API Schemas
# -------------------------
class ReportCreate(BaseModel):
    text: str = Field(min_length=5, max_length=150)
    latitude: float
    longitude: float
    source: str = "real"


class ReportOut(BaseModel):
    id: int
    text: str
    latitude: float
    longitude: float
    timestamp: str
    category: str
    confidence: float
    source: str
    model_name: str
    model_version: str
    is_corrected: bool


app = FastAPI(title="RideAware API")

# CORS (Frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def validate_text(text: str) -> None:
    stripped = (text or "").strip()
    if len(stripped) < 5:
        raise HTTPException(status_code=400, detail="Text must be at least 5 characters.")
    has_alnum = any(ch.isalnum() for ch in stripped)
    if not has_alnum:
        raise HTTPException(
            status_code=400,
            detail="Text must contain letters or numbers (not only emojis/symbols).",
        )


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/reports", response_model=ReportOut)
def create_report(payload: ReportCreate):
    validate_text(payload.text)

    category, confidence, model_name, model_version = classify_text(payload.text)
    now = datetime.now(timezone.utc)

    db = SessionLocal()
    try:
        r = ReportDB(
            text=payload.text.strip(),
            latitude=payload.latitude,
            longitude=payload.longitude,
            timestamp=now,
            category=category,
            confidence=float(confidence),
            model_name=model_name,
            model_version=model_version,
            source=payload.source,
            is_corrected=False,
        )
        db.add(r)
        db.commit()
        db.refresh(r)
        return ReportOut(
            id=r.id,
            text=r.text,
            latitude=r.latitude,
            longitude=r.longitude,
            timestamp=r.timestamp.isoformat(),
            category=r.category,
            confidence=r.confidence,
            source=r.source,
            model_name=r.model_name,
            model_version=r.model_version,
            is_corrected=r.is_corrected,
        )
    finally:
        db.close()


@app.get("/reports", response_model=List[ReportOut])
def list_reports(include_spam: bool = False):
    db = SessionLocal()
    try:
        q = db.query(ReportDB)
        if not include_spam:
            q = q.filter(ReportDB.category != "Spam")
        q = q.order_by(ReportDB.id.desc())
        items = q.all()
        return [
            ReportOut(
                id=r.id,
                text=r.text,
                latitude=r.latitude,
                longitude=r.longitude,
                timestamp=r.timestamp.isoformat(),
                category=r.category,
                confidence=r.confidence,
                source=r.source,
                model_name=r.model_name,
                model_version=r.model_version,
                is_corrected=r.is_corrected,
            )
            for r in items
        ]
    finally:
        db.close()
