// src/App.jsx
import { useEffect, useState } from "react";
import { fetchMeta, simulateStrategy } from "./api/client";

const defaultPitLoss = 22.0;

export default function App() {
  const [meta, setMeta] = useState(null);
  const [loadingMeta, setLoadingMeta] = useState(true);
  const [metaError, setMetaError] = useState("");

  const [gp, setGp] = useState("");
  const [raceLaps, setRaceLaps] = useState(57);
  const [pitLoss, setPitLoss] = useState(defaultPitLoss);
  const [enabledCompounds, setEnabledCompounds] = useState(["Soft", "Medium", "Hard"]);

  const [simLoading, setSimLoading] = useState(false);
  const [simError, setSimError] = useState("");
  const [simResult, setSimResult] = useState(null);

  // ---- Load metadata on mount ----
  useEffect(() => {
    async function load() {
      try {
        setLoadingMeta(true);
        const data = await fetchMeta();
        setMeta(data);
        const gpNames = Object.keys(data.gps || {}).filter((k) => k !== "_global");
        if (gpNames.length > 0) {
          setGp(gpNames[0]);
        }
      } catch (err) {
        console.error(err);
        setMetaError(err.message || "Failed to load metadata");
      } finally {
        setLoadingMeta(false);
      }
    }
    load();
  }, []);

  const toggleCompound = (comp) => {
    setEnabledCompounds((prev) => {
      if (prev.includes(comp)) {
        // keep at least 2 compounds
        if (prev.length <= 2) return prev;
        return prev.filter((c) => c !== comp);
      } else {
        return [...prev, comp];
      }
    });
  };

  const handleSimulate = async (e) => {
    e.preventDefault();
    if (!gp) return;

    setSimError("");
    setSimLoading(true);
    setSimResult(null);
    try {
      const payload = {
        gp,
        race_laps: Number(raceLaps),
        pit_loss_s: Number(pitLoss),
        enabled_compounds: enabledCompounds,
        top_n: 10,
      };
      const data = await simulateStrategy(payload);
      setSimResult(data);
    } catch (err) {
      console.error(err);
      setSimError(err.message || "Simulation failed");
    } finally {
      setSimLoading(false);
    }
  };

  return (
    <div style={styles.page}>
      <div style={styles.container}>
        <h1 style={styles.title}>Tyre Degradation Strategy Explorer</h1>
        <p style={styles.subtitle}>
          Backend: FastAPI · Frontend: React · Data: FastF1 (2022–2024 ground-effect era)
        </p>

        {/* --- META LOADING STATE --- */}
        {loadingMeta && <p>Loading track list…</p>}
        {metaError && <p style={styles.error}>⚠ {metaError}</p>}

        {/* --- FORM --- */}
        {meta && !loadingMeta && (
          <form onSubmit={handleSimulate} style={styles.form}>
            <div style={styles.fieldRow}>
              <label style={styles.label}>Grand Prix</label>
              <select
                value={gp}
                onChange={(e) => setGp(e.target.value)}
                style={styles.select}
              >
                {Object.keys(meta.gps || {})
                  .filter((k) => k !== "_global")
                  .map((name) => (
                    <option key={name} value={name}>
                      {name}
                    </option>
                  ))}
              </select>
            </div>

            <div style={styles.fieldRow}>
              <label style={styles.label}>Race laps</label>
              <input
                type="number"
                min="10"
                max="80"
                value={raceLaps}
                onChange={(e) => setRaceLaps(e.target.value)}
                style={styles.input}
              />
            </div>

            <div style={styles.fieldRow}>
              <label style={styles.label}>Pit loss (s)</label>
              <input
                type="number"
                step="0.1"
                value={pitLoss}
                onChange={(e) => setPitLoss(e.target.value)}
                style={styles.input}
              />
            </div>

            <div style={styles.fieldRow}>
              <label style={styles.label}>Compounds</label>
              <div>
                {["Soft", "Medium", "Hard"].map((comp) => (
                  <label key={comp} style={styles.checkboxLabel}>
                    <input
                      type="checkbox"
                      checked={enabledCompounds.includes(comp)}
                      onChange={() => toggleCompound(comp)}
                    />{" "}
                    {comp}
                  </label>
                ))}
                <div style={styles.hint}>
                  (Keep at least two enabled to satisfy FIA 2-compound rule.)
                </div>
              </div>
            </div>

            <button type="submit" style={styles.button} disabled={simLoading}>
              {simLoading ? "Simulating…" : "Simulate strategies"}
            </button>
          </form>
        )}

        {/* --- SIMULATION OUTPUT --- */}
        {simError && <p style={styles.error}>⚠ {simError}</p>}

        {simResult && (
          <div style={styles.resultsSection}>
            <h2 style={styles.sectionTitle}>
              {simResult.gp} — {simResult.race_laps} laps
            </h2>
            <p style={styles.metaLine}>
              Pit loss: {simResult.pit_loss_s.toFixed(1)} s · Compounds:{" "}
              {Object.keys(simResult.compounds).join(", ")}
            </p>

            {/* Compound table */}
            <h3 style={styles.subheading}>Compound parameters</h3>
            <table style={styles.table}>
              <thead>
                <tr>
                  <th>Compound</th>
                  <th>Baseline (s)</th>
                  <th>Slope (s/lap)</th>
                  <th>Cap (laps)</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(simResult.compounds).map(([name, cp]) => (
                  <tr key={name}>
                    <td>{name}</td>
                    <td>{cp.baseline_s.toFixed(3)}</td>
                    <td>{cp.slope_s_per_lap.toFixed(4)}</td>
                    <td>{cp.cap_laps}</td>
                  </tr>
                ))}
              </tbody>
            </table>

            {/* Strategy table */}
            <h3 style={styles.subheading}>Top strategies (lower is better)</h3>
            <table style={styles.table}>
              <thead>
                <tr>
                  <th>#</th>
                  <th>Stops</th>
                  <th>Stints (laps)</th>
                  <th>Compounds</th>
                  <th>Total time (s)</th>
                </tr>
              </thead>
              <tbody>
                {simResult.top_strategies.map((s, idx) => (
                  <tr key={idx}>
                    <td>{idx + 1}</td>
                    <td>{s.stops}</td>
                    <td>{s.stints.join("-")}</td>
                    <td>{s.compounds.join("-")}</td>
                    <td>{s.total_time_s.toFixed(1)}</td>
                  </tr>
                ))}
              </tbody>
            </table>

            {/* Stint curves (simple textual summary for now;
                later we can plot with a chart library like Recharts) */}
            <h3 style={styles.subheading}>Stint curves (predicted lap time)</h3>
            <ul>
              {Object.entries(simResult.stint_curves).map(([comp, curve]) => (
                <li key={comp}>
                  <strong>{comp}</strong>: ages 1–{curve.ages[curve.ages.length - 1]}  
                  &nbsp;| first lap ≈ {curve.times[0].toFixed(3)} s,  
                  last ≈ {curve.times[curve.times.length - 1].toFixed(3)} s
                </li>
              ))}
            </ul>
          </div>
        )}

        <footer style={styles.footer}>
          <span>
            Built by you • Tyre Degradation Model · Stores no personal data.
          </span>
        </footer>
      </div>
    </div>
  );
}

const styles = {
  page: {
    minHeight: "100vh",
    background: "#0b1020",
    color: "#f5f5f5",
    padding: "2rem 1rem",
    fontFamily: "system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
  },
  container: {
    maxWidth: "960px",
    margin: "0 auto",
  },
  title: {
    fontSize: "2.2rem",
    marginBottom: "0.25rem",
    fontWeight: 700,
  },
  subtitle: {
    fontSize: "0.95rem",
    opacity: 0.8,
    marginBottom: "1.5rem",
  },
  form: {
    background: "#141a33",
    padding: "1.5rem",
    borderRadius: "12px",
    marginBottom: "1.5rem",
    boxShadow: "0 10px 25px rgba(0,0,0,0.3)",
  },
  fieldRow: {
    display: "flex",
    alignItems: "center",
    marginBottom: "0.9rem",
    gap: "0.75rem",
  },
  label: {
    minWidth: "110px",
    fontSize: "0.95rem",
  },
  select: {
    flex: 1,
    padding: "0.4rem 0.5rem",
    borderRadius: "6px",
    border: "1px solid #3a4375",
    background: "#0e1224",
    color: "#f5f5f5",
  },
  input: {
    flex: 1,
    padding: "0.4rem 0.5rem",
    borderRadius: "6px",
    border: "1px solid #3a4375",
    background: "#0e1224",
    color: "#f5f5f5",
  },
  checkboxLabel: {
    marginRight: "0.75rem",
    fontSize: "0.9rem",
  },
  hint: {
    fontSize: "0.8rem",
    opacity: 0.7,
    marginTop: "0.25rem",
  },
  button: {
    marginTop: "0.8rem",
    padding: "0.55rem 1.2rem",
    borderRadius: "999px",
    border: "none",
    background: "#ff4f4f",
    color: "white",
    fontWeight: 600,
    cursor: "pointer",
  },
  resultsSection: {
    background: "#141a33",
    padding: "1.5rem",
    borderRadius: "12px",
    boxShadow: "0 10px 25px rgba(0,0,0,0.3)",
  },
  sectionTitle: {
    fontSize: "1.4rem",
    marginBottom: "0.25rem",
  },
  metaLine: {
    fontSize: "0.9rem",
    opacity: 0.8,
    marginBottom: "1rem",
  },
  table: {
    width: "100%",
    borderCollapse: "collapse",
    marginBottom: "1.2rem",
    fontSize: "0.9rem",
  },
  subheading: {
    marginTop: "0.8rem",
    marginBottom: "0.4rem",
    fontSize: "1.05rem",
  },
  error: {
    color: "#ff7777",
    marginBottom: "0.75rem",
  },
  footer: {
    marginTop: "1.5rem",
    fontSize: "0.8rem",
    opacity: 0.7,
    textAlign: "center",
  },
};
