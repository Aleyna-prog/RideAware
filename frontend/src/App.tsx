import { useEffect, useState } from "react";
import { MapContainer, Marker, Popup, TileLayer, useMapEvents } from "react-leaflet";
import type { LeafletMouseEvent } from "leaflet";
import L from "leaflet";
import "./App.css";

type Report = {
  id: number;
  text: string;
  latitude: number;
  longitude: number;
  timestamp: string;
  category: "Hindernis" | "Infrastrukturproblem" | "Gefahrenstelle" | "Positives Feedback" | "Spam" | string;
  confidence: number;
  source: string;
  is_corrected: boolean;
  // if you later add these in backend output, it will still work:
  model_name?: string;
  model_version?: string;
};

const API_BASE = "http://127.0.0.1:8000";

function ClickToSetMarker({ onPick }: { onPick: (lat: number, lng: number) => void }) {
  useMapEvents({
    click(e: LeafletMouseEvent) {
      onPick(e.latlng.lat, e.latlng.lng);
    },
  });
  return null;
}

/** SRS label set -> stable colors */
function markerColor(category: string) {
  switch ((category ?? "").trim()) {
    case "Hindernis":
      return "#ff5a5f"; // red
    case "Gefahrenstelle":
      return "#ffb020"; // orange
    case "Infrastrukturproblem":
      return "#2dd4bf"; // teal
    case "Positives Feedback":
      return "#4f8cff"; // blue
    case "Spam":
      return "#94a3b8"; // grey (normally not shown)
    default:
      return "#a78bfa"; // fallback purple
  }
}

function makeDotIcon(color: string) {
  return L.divIcon({
    className: "",
    html: `
      <div style="
        width: 18px; height: 18px;
        background: ${color};
        border: 2px solid rgba(255,255,255,0.95);
        border-radius: 999px;
        box-shadow: 0 6px 14px rgba(0,0,0,0.35);
      "></div>
    `,
    iconSize: [18, 18],
    iconAnchor: [9, 9],
    popupAnchor: [0, -8],
  });
}

export default function App() {
  const [text, setText] = useState("");
  const [picked, setPicked] = useState<{ lat: number; lng: number } | null>(null);
  const [reports, setReports] = useState<Report[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const center: [number, number] = [48.2082, 16.3738]; // Wien

  async function loadReports() {
    const res = await fetch(`${API_BASE}/reports`);
    const data = (await res.json()) as Report[];

    // Backend already hides Spam by default, but keep this as a safety-net:
    setReports(data.filter((r) => r.category !== "Spam"));
  }

  useEffect(() => {
    loadReports().catch(() => setError("Reports konnten nicht geladen werden."));
  }, []);

  async function submit() {
    setError(null);

    const trimmed = text.trim();
    if (trimmed.length < 5 || trimmed.length > 150) {
      setError("Text muss zwischen 5 und 150 Zeichen haben.");
      return;
    }
    if (!picked) {
      setError("Bitte zuerst eine Position auf der Karte auswählen.");
      return;
    }

    setBusy(true);
    try {
      const res = await fetch(`${API_BASE}/reports`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          text: trimmed,
          latitude: picked.lat,
          longitude: picked.lng,
          source: "real",
        }),
      });

      if (!res.ok) {
        const msg = await res.json().catch(() => null);
        setError(msg?.detail ?? "Senden fehlgeschlagen.");
        return;
      }

      setText("");
      await loadReports();
    } catch {
      setError("Backend nicht erreichbar (läuft FastAPI auf Port 8000?).");
    } finally {
      setBusy(false);
    }
  }

  const CardStyle: React.CSSProperties = {
    border: "1px solid var(--border-soft)",
    borderRadius: 14,
    background: "var(--bg-card)",
    padding: 16,
    boxShadow: "0 10px 30px rgba(0,0,0,0.35)",
  };

  return (
    <div
      style={{
        height: "100vh",
        width: "100vw",
        display: "flex",
        flexDirection: "column",
        background: "var(--bg-main)",
        color: "var(--text-main)",
      }}
    >
      {/* Header */}
      <header
        style={{
          padding: "18px 20px",
          borderBottom: "1px solid var(--border-soft)",
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          background: "linear-gradient(180deg, rgba(79,140,255,0.10), rgba(0,0,0,0))",
        }}
      >
        <div>
          <div style={{ fontSize: 20, fontWeight: 800, letterSpacing: 0.2 }}>RideAware</div>
          <div style={{ fontSize: 12, color: "var(--text-muted)" }}>Near-miss & Hazard Reporting (Prototype)</div>
        </div>

        <div style={{ fontSize: 12, color: "var(--text-muted)" }}>
          Backend: <span style={{ color: "var(--text-main)" }}>{API_BASE}</span>
        </div>
      </header>

      {/* Top content */}
      <div style={{ padding: 16, display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
        {/* Form card */}
        <section style={CardStyle}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline" }}>
            <h3 style={{ margin: 0, fontSize: 14, fontWeight: 800 }}>Neue Meldung</h3>
            <div style={{ fontSize: 12, color: "var(--text-muted)" }}>
              {picked ? `📍 ${picked.lat.toFixed(5)}, ${picked.lng.toFixed(5)}` : "📍 auf Karte klicken"}
            </div>
          </div>

          <div style={{ marginTop: 10, fontSize: 12, color: "var(--text-muted)" }}>Meldung (5–150 Zeichen)</div>

          <textarea
            value={text}
            onChange={(e) => setText(e.target.value)}
            rows={4}
            placeholder="z. B. Glas auf dem Radweg bei der Kreuzung …"
            style={{
              width: "100%",
              marginTop: 8,
              borderRadius: 12,
              padding: 12,
              border: "1px solid var(--border-soft)",
              background: "var(--bg-card-soft)",
              color: "var(--text-main)",
              outline: "none",
              resize: "vertical",
            }}
          />

          {error && (
            <div
              style={{
                marginTop: 10,
                padding: "10px 12px",
                borderRadius: 12,
                background: "rgba(255,70,70,0.12)",
                border: "1px solid rgba(255,70,70,0.25)",
                color: "rgba(255,190,190,0.95)",
                fontSize: 12,
              }}
            >
              {error}
            </div>
          )}

          <button
            onClick={submit}
            disabled={busy}
            style={{
              marginTop: 12,
              width: "100%",
              padding: "10px 12px",
              borderRadius: 12,
              border: "1px solid var(--border-soft)",
              background: busy ? "rgba(255,255,255,0.06)" : "rgba(79,140,255,0.22)",
              color: "var(--text-main)",
              cursor: busy ? "not-allowed" : "pointer",
            }}
          >
            {busy ? "Senden…" : "Meldung absenden"}
          </button>

          <div style={{ marginTop: 10, fontSize: 12, color: "var(--text-muted)" }}>
            Tipp: Erst auf die Karte klicken, dann absenden. Marker erscheinen direkt auf der Map.
          </div>
        </section>

        {/* Reports card */}
        <section style={{ ...CardStyle, overflow: "hidden" }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline" }}>
            <h3 style={{ margin: 0, fontSize: 14, fontWeight: 800 }}>Letzte Meldungen</h3>
            <div style={{ fontSize: 12, color: "var(--text-muted)" }}>{reports.length} Einträge</div>
          </div>

          <div style={{ marginTop: 10, maxHeight: 190, overflow: "auto", paddingRight: 6 }}>
            {reports.length === 0 ? (
              <div style={{ fontSize: 12, color: "var(--text-muted)" }}>
                Noch keine Meldungen. Erstelle eine neue Meldung über das Formular.
              </div>
            ) : (
              <ul style={{ listStyle: "none", padding: 0, margin: 0 }}>
                {reports.map((r) => {
                  const col = markerColor(r.category);
                  return (
                    <li
                      key={r.id}
                      style={{
                        padding: "10px 0",
                        borderBottom: "1px solid var(--border-soft)",
                      }}
                    >
                      <div style={{ display: "flex", justifyContent: "space-between", gap: 10 }}>
                        <div style={{ fontWeight: 800, fontSize: 12, display: "flex", alignItems: "center", gap: 8 }}>
                          <span
                            style={{
                              width: 10,
                              height: 10,
                              borderRadius: 999,
                              background: col,
                              border: "1px solid rgba(255,255,255,0.7)",
                              display: "inline-block",
                            }}
                          />
                          {r.category}{" "}
                          <span style={{ color: "var(--text-muted)", fontWeight: 600 }}>
                            ({r.confidence.toFixed(2)})
                          </span>
                        </div>

                        <div style={{ fontSize: 12, color: "var(--text-muted)", whiteSpace: "nowrap" }}>
                          {new Date(r.timestamp).toLocaleString()}
                        </div>
                      </div>

                      <div style={{ marginTop: 6, fontSize: 13, color: "var(--text-main)" }}>{r.text}</div>

                      <div style={{ marginTop: 6, fontSize: 12, color: "var(--text-muted)" }}>
                        📍 {r.latitude.toFixed(5)}, {r.longitude.toFixed(5)}
                      </div>
                    </li>
                  );
                })}
              </ul>
            )}
          </div>
        </section>
      </div>

      {/* Map panel */}
      <div style={{ flex: 1, padding: "0 16px 16px 16px" }}>
        <section
          style={{
            height: "100%",
            borderRadius: 14,
            overflow: "hidden",
            border: "1px solid var(--border-soft)",
            background: "var(--bg-card)",
            boxShadow: "0 10px 30px rgba(0,0,0,0.35)",
            position: "relative",
          }}
        >
          {/* Legend */}
          <div
            style={{
              position: "absolute",
              zIndex: 999,
              right: 16,
              top: 12,
              background: "rgba(17, 26, 46, 0.92)",
              border: "1px solid var(--border-soft)",
              borderRadius: 12,
              padding: "10px 12px",
              fontSize: 12,
              color: "var(--text-main)",
              boxShadow: "0 10px 30px rgba(0,0,0,0.35)",
              backdropFilter: "blur(6px)",
            }}
          >
            <div style={{ fontWeight: 800, marginBottom: 6 }}>Legende</div>
            <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
              <span style={{ width: 10, height: 10, borderRadius: 999, background: markerColor("Hindernis"), display: "inline-block" }} />
              Hindernis
            </div>
            <div style={{ display: "flex", gap: 8, alignItems: "center", marginTop: 4 }}>
              <span style={{ width: 10, height: 10, borderRadius: 999, background: markerColor("Gefahrenstelle"), display: "inline-block" }} />
              Gefahrenstelle
            </div>
            <div style={{ display: "flex", gap: 8, alignItems: "center", marginTop: 4 }}>
              <span style={{ width: 10, height: 10, borderRadius: 999, background: markerColor("Infrastrukturproblem"), display: "inline-block" }} />
              Infrastrukturproblem
            </div>
            <div style={{ display: "flex", gap: 8, alignItems: "center", marginTop: 4 }}>
              <span style={{ width: 10, height: 10, borderRadius: 999, background: markerColor("Positives Feedback"), display: "inline-block" }} />
              Positives Feedback
            </div>
          </div>

          <MapContainer center={center} zoom={12} style={{ height: "100%", width: "100%" }}>
            <TileLayer attribution="© OpenStreetMap contributors" url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />

            <ClickToSetMarker onPick={(lat, lng) => setPicked({ lat, lng })} />

            {picked && (
              <Marker position={[picked.lat, picked.lng]} icon={makeDotIcon("#4f8cff")}>
                <Popup>Ausgewählte Position</Popup>
              </Marker>
            )}

            {reports.map((r) => (
              <Marker key={r.id} position={[r.latitude, r.longitude]} icon={makeDotIcon(markerColor(r.category))}>
                <Popup>
                  <strong>{r.category}</strong>
                  <br />
                  {r.text}
                </Popup>
              </Marker>
            ))}
          </MapContainer>
        </section>
      </div>
    </div>
  );
}
