"""
RideAware BA2 - Dummy Data Generator

Generiert und lädt Beispieldaten für Tests.
"""

import requests
import random
from typing import List, Tuple

API_BASE = "http://127.0.0.1:8000"

VIENNA_LAT_RANGE = (48.15, 48.25)
VIENNA_LNG_RANGE = (16.30, 16.45)

DUMMY_TEXTS = {
    "Gefahrenstelle": [
        "Gefährliche Kreuzung ohne Sicht",
        "Autos fahren hier viel zu schnell",
        "Beinahe-Unfall wegen schlechter Sicht",
        "Rechtsabbieger schneiden oft den Radweg",
        "Tür wurde plötzlich aufgemacht",
        "Kreuzung sehr unübersichtlich bei Nacht",
        "Auto hat mich fast abgedrängt",
        "Keine Sichtlinie beim Abbiegen",
    ],
    "Hindernis": [
        "Glasscherben auf dem Radweg",
        "Großer Ast blockiert die Fahrbahn",
        "Müllcontainer steht mitten auf dem Radweg",
        "Schlagloch auf der Radspur",
        "Pfütze blockiert den gesamten Radweg",
        "Bauschutt liegt auf dem Radweg",
        "Umgestürzter Baum versperrt den Weg",
        "Nägel auf dem Radweg entdeckt",
    ],
    "Markierung oder Schild": [
        "Radwegmarkierung fehlt komplett",
        "Schild fehlt bei der Abzweigung",
        "Fahrverbot ohne Ausnahme für Radfahrer",
        "Beschilderung sehr unübersichtlich",
        "Bodenmarkierung nach Baustelle fehlt",
        "Zusatztafel ausgenommen Fahrrad fehlt",
        "Schild von Baum verdeckt",
        "Wegweiser zeigt falsche Richtung",
    ],
    "Ampel": [
        "Ampel für Radfahrer defekt",
        "Grünphase für Radfahrer zu kurz",
        "Radfahrerampel fehlt komplett",
        "Ampel schaltet zu früh auf Rot",
        "Sensor der Bedarfsampel defekt",
        "Ampelschaltung für Radfahrer ungünstig",
        "Keine eigene Radampel vorhanden",
        "Ampel zeigt dauerhaft Rot für Radler",
    ],
    "Lückenschluss": [
        "Radweg endet plötzlich ohne Anschluss",
        "Fehlende Radwegverbindung",
        "Kein Radweg auf stark befahrener Straße",
        "Radweg endet mitten in der Kreuzung",
        "Fehlende Verbindung zum Donaukanal",
        "Unterbrechung der Hauptradroute",
        "Keine sichere Route vorhanden",
        "Radweg nach Baustelle nicht weitergeführt",
    ],
}


def generate_random_coordinate() -> Tuple[float, float]:
    lat = random.uniform(*VIENNA_LAT_RANGE)
    lng = random.uniform(*VIENNA_LNG_RANGE)
    return lat, lng


def generate_dummy_reports(count_per_category: int = 5) -> List[dict]:
    reports = []
    for category, texts in DUMMY_TEXTS.items():
        for _ in range(count_per_category):
            text = random.choice(texts)
            lat, lng = generate_random_coordinate()
            reports.append({"text": text, "latitude": lat, "longitude": lng, "source": "dummy"})
    random.shuffle(reports)
    return reports


def main():
    print("=" * 70)
    print("RideAware BA2 - Dummy Data Generator")
    print("=" * 70)

    try:
        response = requests.get(f"{API_BASE}/health", timeout=5)
        if not response.ok:
            print(f"❌ Backend nicht erreichbar. Starte zuerst: uvicorn main:app --reload")
            return
    except Exception:
        print("❌ Backend nicht erreichbar. Starte zuerst: uvicorn main:app --reload")
        return

    reports = generate_dummy_reports(count_per_category=5)
    print(f"✓ {len(reports)} Dummy-Meldungen generiert")

    success = 0
    for i, report in enumerate(reports, 1):
        try:
            res = requests.post(f"{API_BASE}/reports", json=report, timeout=10)
            if res.ok:
                success += 1
        except Exception:
            pass

    print(f"✓ {success}/{len(reports)} Meldungen hochgeladen")


if __name__ == "__main__":
    main()