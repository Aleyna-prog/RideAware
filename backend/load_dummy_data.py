"""
SRS S3: Dummy Data Integration

This script generates realistic dummy data and uploads it to the backend.
Can be used for testing, training data augmentation, or demonstrations.
"""

import requests
import random
from typing import List, Tuple

API_BASE = "http://127.0.0.1:8000"

# Vienna coordinates (approximate boundaries)
VIENNA_LAT_RANGE = (48.15, 48.25)
VIENNA_LNG_RANGE = (16.30, 16.45)

# Example texts per category (German & English mix)
DUMMY_TEXTS = {
    "Hindernis": [
        "Glasscherben auf dem Radweg",
        "Großer Ast blockiert die Fahrbahn",
        "Müllcontainer steht mitten auf dem Radweg",
        "Steinbrocken nach Bauarbeiten liegen rum",
        "Umgestürzter Baum versperrt den Weg",
        "Scherben bei der Kreuzung Mariahilfer Straße",
        "Hindernis: Container auf Radweg seit 3 Tagen",
        "Achtung Glasflaschen auf dem Weg zur Donauinsel",
        "Broken glass near the bike lane",
        "Large branch blocking the path",
        "Construction debris on cycle path",
        "Pothole in the middle of bike lane",
    ],
    "Infrastrukturproblem": [
        "Radweg ist sehr schlecht markiert",
        "Schlagloch auf der Ringstraße beim Burgtheater",
        "Radwegende ist nicht klar ersichtlich",
        "Keine Beschilderung bei der Abzweigung",
        "Radweg total holprig und uneben",
        "Ampelschaltung für Radfahrer zu kurz",
        "Baustelle ohne Umleitung für Radfahrer",
        "Radstreifen zu schmal, wird von Autos befahren",
        "Bike lane markings are fading",
        "Missing sign for cyclists",
        "Road surface is damaged and uneven",
        "No dedicated bike lane here",
    ],
    "Gefahrenstelle": [
        "Gefährliche Kreuzung ohne Sicht",
        "Autos fahren hier viel zu schnell",
        "Beinahe-Unfall wegen schlechter Sicht",
        "Kreuzung sehr unübersichtlich bei Nacht",
        "Close pass - Auto kam mir sehr nahe",
        "Rechtsabbieger schneiden oft den Radweg",
        "Tür wurde plötzlich aufgemacht - knapp!",
        "Sehr gefährliche Stelle beim Gürtel",
        "Near miss with a car turning right",
        "Dangerous crossing, poor visibility",
        "Car door opened right in front of me",
        "Almost hit at the intersection",
    ],
    "Positives Feedback": [
        "Neuer Radweg ist super!",
        "Endlich wurde die Kreuzung verbessert, danke!",
        "Toll, dass hier jetzt Markierungen sind",
        "Gute Ampelschaltung für Radfahrer hier",
        "Perfekter Radweg, mehr davon bitte!",
        "Danke für die neue Beschilderung",
        "Super breiter und sicherer Radweg hier",
        "Love the new bike lane",
        "Great improvement at this crossing",
        "Thanks for fixing the road surface",
        "Nice smooth surface on the bike lane now",
        "Good job on the new markings",
    ],
    "Spam": [
        "Buy cheap bikes now at www.fakebikes.com",
        "Click here for free cycling gear: http://spam.net",
        "Limited offer: Get 50% discount on everything!",
        "Subscribe to my channel for cycling tips - free money",
        "WIN a FREE bike now!!! click here",
        "Best bike deals!!! http://super-deals.com",
        "Follow me for promo codes!!!",
        "Exclusive discount, click www.sale.net",
    ],
}


def generate_random_coordinate() -> Tuple[float, float]:
    """Generate random Vienna coordinate"""
    lat = random.uniform(*VIENNA_LAT_RANGE)
    lng = random.uniform(*VIENNA_LNG_RANGE)
    return lat, lng


def generate_dummy_reports(count_per_category: int = 8) -> List[dict]:
    """
    Generate dummy reports for testing.
    
    Args:
        count_per_category: How many reports per category to generate
    
    Returns:
        List of report dictionaries ready for API
    """
    reports = []
    
    for category, texts in DUMMY_TEXTS.items():
        # Generate reports for this category
        for i in range(count_per_category):
            # Pick random text from category (with repetition allowed)
            text = random.choice(texts)
            lat, lng = generate_random_coordinate()
            
            reports.append({
                "text": text,
                "latitude": lat,
                "longitude": lng,
                "source": "dummy"
            })
    
    # Shuffle to mix categories
    random.shuffle(reports)
    return reports


def upload_reports_one_by_one(reports: List[dict]) -> dict:
    """
    Upload reports one by one (for systems without batch endpoint).
    
    Args:
        reports: List of report dictionaries
    
    Returns:
        Summary statistics
    """
    success_count = 0
    failed_count = 0
    
    for i, report in enumerate(reports, 1):
        try:
            response = requests.post(
                f"{API_BASE}/reports",
                json=report,
                timeout=10
            )
            if response.ok:
                success_count += 1
                print(f"  ✓ [{i}/{len(reports)}] Uploaded: {report['text'][:50]}...")
            else:
                failed_count += 1
                print(f"  ✗ [{i}/{len(reports)}] Failed: {response.status_code}")
        except Exception as e:
            failed_count += 1
            print(f"  ✗ [{i}/{len(reports)}] Error: {e}")
    
    return {
        "success": success_count,
        "failed": failed_count,
        "total": len(reports)
    }


def main():
    print("=" * 70)
    print("RideAware - Dummy Data Generator (SRS S3)")
    print("=" * 70)
    
    # Check if backend is running
    print("\n1. Checking backend connection...")
    try:
        response = requests.get(f"{API_BASE}/health", timeout=5)
        if response.ok:
            print(f"   ✓ Backend is running at {API_BASE}")
        else:
            print(f"   ✗ Backend returned status {response.status_code}")
            return
    except Exception as e:
        print(f"   ✗ Cannot connect to backend: {e}")
        print("\n   Please start the backend first:")
        print("   cd backend")
        print("   uvicorn main:app --reload --port 8000")
        return
    
    # Generate dummy data
    print("\n2. Generating dummy reports...")
    reports = generate_dummy_reports(count_per_category=8)
    print(f"   ✓ Generated {len(reports)} dummy reports")
    
    # Show sample
    print("\n3. Sample reports:")
    for i, r in enumerate(reports[:5], 1):
        print(f"   {i}. {r['text'][:60]}... @ ({r['latitude']:.4f}, {r['longitude']:.4f})")
    print(f"   ... and {len(reports) - 5} more")
    
    # Upload
    print(f"\n4. Uploading to {API_BASE}/reports ...")
    print("   (This may take a moment...)")
    result = upload_reports_one_by_one(reports)
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"✓ Successfully uploaded: {result['success']}/{result['total']}")
    if result['failed'] > 0:
        print(f"✗ Failed: {result['failed']}/{result['total']}")
    print("\n" + "=" * 70)
    print("Done! Check your frontend map to see the dummy data.")
    print("All dummy reports are marked with source='dummy'")
    print("=" * 70)


if __name__ == "__main__":
    main()
