export type Language = 'en' | 'de';

export const translations = {
  en: {
    // Header
    appTitle: "RideAware",
    appSubtitle: "Near-miss & Hazard Reporting (Prototype)",
    backend: "Backend",
    
    // Form section
    newReport: "New Report",
    clickOnMap: "click on map",
    reportLabel: "Report (5–150 characters)",
    reportPlaceholder: "e.g. Glass on bike lane at intersection…",
    submitButton: "Submit Report",
    submitting: "Sending…",
    tipText: "Tip: First click on the map, then submit. Markers appear directly on the map.",
    
    // Reports section
    latestReports: "Latest Reports",
    entries: "entries",
    noReports: "No reports yet. Create a new report using the form.",
    
    // Map legend
    legend: "Legend",
    obstacle: "Obstacle",
    dangerSpot: "Danger Spot",
    infrastructureProblem: "Infrastructure Problem",
    positiveFeedback: "Positive Feedback",
    
    // Map popup
    selectedPosition: "Selected Position",
    
    // Error messages
    errorLength: "Text must be between 5 and 150 characters.",
    errorNoLocation: "Please select a position on the map first.",
    errorSendFailed: "Sending failed.",
    errorBackendUnreachable: "Backend unreachable (is FastAPI running on port 8000?).",
    errorLoadReports: "Reports could not be loaded.",
  },
  de: {
    // Header
    appTitle: "RideAware",
    appSubtitle: "Beinahe-Unfälle & Gefahrenmeldungen (Prototyp)",
    backend: "Backend",
    
    // Form section
    newReport: "Neue Meldung",
    clickOnMap: "auf Karte klicken",
    reportLabel: "Meldung (5–150 Zeichen)",
    reportPlaceholder: "z. B. Glas auf dem Radweg bei der Kreuzung…",
    submitButton: "Meldung absenden",
    submitting: "Senden…",
    tipText: "Tipp: Erst auf die Karte klicken, dann absenden. Marker erscheinen direkt auf der Map.",
    
    // Reports section
    latestReports: "Letzte Meldungen",
    entries: "Einträge",
    noReports: "Noch keine Meldungen. Erstelle eine neue Meldung über das Formular.",
    
    // Map legend
    legend: "Legende",
    obstacle: "Hindernis",
    dangerSpot: "Gefahrenstelle",
    infrastructureProblem: "Infrastrukturproblem",
    positiveFeedback: "Positives Feedback",
    
    // Map popup
    selectedPosition: "Ausgewählte Position",
    
    // Error messages
    errorLength: "Text muss zwischen 5 und 150 Zeichen haben.",
    errorNoLocation: "Bitte zuerst eine Position auf der Karte auswählen.",
    errorSendFailed: "Senden fehlgeschlagen.",
    errorBackendUnreachable: "Backend nicht erreichbar (läuft FastAPI auf Port 8000?).",
    errorLoadReports: "Reports konnten nicht geladen werden.",
  }
};

// Category translations (for display)
export const categoryTranslations = {
  en: {
    "Hindernis": "Obstacle",
    "Gefahrenstelle": "Danger Spot",
    "Infrastrukturproblem": "Infrastructure Problem",
    "Positives Feedback": "Positive Feedback",
    "Spam": "Spam"
  },
  de: {
    "Hindernis": "Hindernis",
    "Gefahrenstelle": "Gefahrenstelle",
    "Infrastrukturproblem": "Infrastrukturproblem",
    "Positives Feedback": "Positives Feedback",
    "Spam": "Spam"
  }
};