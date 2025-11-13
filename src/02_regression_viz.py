# ============================================
# 02_regression_viz.py — PROFESSOR MODE v2 👩‍🏫
# Tyre degradation on Δ-time with driver FE
# Adds:
#   • Within-stint baseline (Δ-time)
#   • Driver fixed effects (demean per driver)
#   • Linear vs Quadratic vs Piecewise (BIC selection)
#   • HC3 robust SE
#   • Strong NaN/Inf hardening + global fallbacks
#   • Safe working directory
# ============================================

# ---- Safe working directory (run from anywhere) ----
import os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)

# ---- Imports ----
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.tools.tools import add_constant
from sklearn.metrics import r2_score, mean_absolute_error
from config import YEAR, GRAND_PRIX, DEGRADATION_THRESHOLD, RANDOM_SEED

np.random.seed(RANDOM_SEED)
sns.set(context="notebook", style="whitegrid")

# ---- Files ----
DATA_FILE = Path(f"data/processed/laps_{GRAND_PRIX.lower()}{YEAR}.csv")
FIG_DIR = Path("figures"); FIG_DIR.mkdir(parents=True, exist_ok=True)
assert DATA_FILE.exists(), "Run src/01_extract_prepare.py first!"
print(f"📦 Using dataset: {DATA_FILE}")

# ------------------------------------
# STEP 1 — Load & basic prep
# ------------------------------------
df = pd.read_csv(DATA_FILE)
need = {"lap_time_s","tyre_age_laps","compound_norm"}
missing = need - set(df.columns)
if missing:
    raise ValueError(f"Dataset missing columns: {missing}")

df = df.dropna(subset=["lap_time_s","tyre_age_laps","compound_norm"]).copy()

# Fill optional controls roughly (we’ll harden again later)
for col in ["track_temp_c","session_min"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Global medians as fallbacks if a subset has all-NaN controls
GLOBAL_MED = {
    "track_temp_c": (df["track_temp_c"].median() if "track_temp_c" in df.columns else 0.0),
    "session_min":  (df["session_min"].median()  if "session_min"  in df.columns else 0.0),
}

# ------------------------------------
# STEP 2 — Δ-time within stint + driver FE
# ------------------------------------
def stint_baseline(g):
    # baseline: median of laps 2–3 (avoid pit-exit). Fallback: first 3 laps.
    ref = g.loc[g["tyre_age_laps"].isin([2,3]), "lap_time_s"]
    base = ref.median() if len(ref) else g["lap_time_s"].head(3).median()
    g["delta_lap_time"] = g["lap_time_s"] - base
    return g

df = (df.sort_values(["Driver","Stint","LapNumber"])
        .groupby(["Driver","Stint"], as_index=False)
        .apply(stint_baseline)
        .reset_index(drop=True))

driver_med = df.groupby("Driver")["delta_lap_time"].median().rename("drv_med")
df = df.merge(driver_med, on="Driver", how="left")
df["delta_fe"] = df["delta_lap_time"] - df["drv_med"]

# mild trimming for stability
df = df[df["delta_fe"].between(-5, 5)]
df = df[df["tyre_age_laps"] >= 1]

# Coerce numerics and replace infinities
for c in ["tyre_age_laps","track_temp_c","session_min","delta_fe"]:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce").replace([np.inf, -np.inf], np.nan)

# Fill controls with per-compound medians → fallback to global
for c in ["track_temp_c","session_min"]:
    if c in df.columns:
        df[c] = df.groupby("compound_norm")[c].transform(
            lambda s: s.fillna(s.median())
        )
        df[c] = df[c].fillna(GLOBAL_MED[c])

# Essential columns must be clean
df = df.dropna(subset=["tyre_age_laps","delta_fe"])

# ------------------------------------
# STEP 3 — Model builders
# ------------------------------------
def design_matrix(sub, form="linear", knot=None):
    """Return a finite X matrix with controls filled safely."""
    X = pd.DataFrame({"tyre_age_laps": sub["tyre_age_laps"].values})

    if form == "quadratic":
        X["tyre_age_laps2"] = sub["tyre_age_laps"].values ** 2
    if form == "piecewise" and knot is not None:
        X["hinge"] = np.clip(sub["tyre_age_laps"].values - knot, 0, None)

    # Controls (robust fills)
    if "track_temp_c" in sub.columns:
        s = pd.to_numeric(sub["track_temp_c"], errors="coerce").replace([np.inf, -np.inf], np.nan)
        med = s.median();  med = GLOBAL_MED["track_temp_c"] if pd.isna(med) else med
        X["track_temp_c"] = s.fillna(med).values
    else:
        X["track_temp_c"] = GLOBAL_MED["track_temp_c"]

    if "session_min" in sub.columns:
        s = pd.to_numeric(sub["session_min"], errors="coerce").replace([np.inf, -np.inf], np.nan)
        med = s.median();  med = GLOBAL_MED["session_min"] if pd.isna(med) else med
        X["session_min"] = s.fillna(med).values
    else:
        X["session_min"] = GLOBAL_MED["session_min"]

    # Final hardening + constant
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    X = add_constant(X, has_constant="add")
    return X

def fit_ols(y, X):
    # HC3 robust covariance to handle heteroskedasticity
    return sm.OLS(y, X).fit(cov_type="HC3")

def select_form(sub):
    """Pick linear/quadratic/piecewise (knot ∈ [4..12]) by lowest BIC."""
    y = sub["delta_fe"].values

    X_lin = design_matrix(sub, "linear")
    m_lin = fit_ols(y, X_lin); best = ("linear", m_lin, None)

    X_quad = design_matrix(sub, "quadratic")
    m_quad = fit_ols(y, X_quad)
    if m_quad.bic < best[1].bic:
        best = ("quadratic", m_quad, None)

    for k in range(4, 13):
        X_pw = design_matrix(sub, "piecewise", knot=k)
        m_pw = fit_ols(y, X_pw)
        if m_pw.bic < best[1].bic:
            best = ("piecewise", m_pw, k)

    return best  # (form, model, knot)

def predict_curve(model, form, knot, sub):
    xgrid = np.linspace(1, sub["tyre_age_laps"].max(), 100)
    grid = pd.DataFrame({"tyre_age_laps": xgrid})
    if form == "quadratic":
        grid["tyre_age_laps2"] = xgrid**2
    if form == "piecewise" and knot is not None:
        grid["hinge"] = np.clip(xgrid - knot, 0, None)
    grid["track_temp_c"] = sub["track_temp_c"].mean() if "track_temp_c" in sub else GLOBAL_MED["track_temp_c"]
    grid["session_min"]  = sub["session_min"].mean()  if "session_min"  in sub else GLOBAL_MED["session_min"]
    grid = add_constant(grid, has_constant="add")
    yhat = model.predict(grid)
    return xgrid, yhat

def slope_at(model, form, knot, x_at=10):
    """Marginal slope (s/lap) at a reference lap for interpretability."""
    p = model.params
    b1 = p.get("tyre_age_laps", np.nan)
    if form == "linear":
        return b1
    if form == "quadratic":
        b2 = p.get("tyre_age_laps2", 0.0)
        return b1 + 2*b2*x_at
    if form == "piecewise":
        b_hinge = p.get("hinge", 0.0)
        return b1 + (b_hinge if x_at > (knot or 0) else 0.0)
    return np.nan

# ------------------------------------
# STEP 4 — Fit per compound & plot
# ------------------------------------
results = []
for comp in ["Soft","Medium","Hard"]:
    sub = df[df["compound_norm"] == comp].copy()
    if sub.empty:
        print(f"⚠️  No data for {comp}"); continue

    # Final scrub per compound (ensure numeric & finite)
    for c in ["tyre_age_laps","delta_fe","track_temp_c","session_min"]:
        if c in sub.columns:
            sub[c] = pd.to_numeric(sub[c], errors="coerce").replace([np.inf, -np.inf], np.nan)

    for c in ["track_temp_c","session_min"]:
        if c in sub.columns:
            med = sub[c].median()
            if pd.isna(med): med = GLOBAL_MED[c]
            sub[c] = sub[c].fillna(med)

    sub = sub.dropna(subset=["tyre_age_laps","delta_fe"])
    sub = sub[sub["tyre_age_laps"] >= 1]
    if len(sub) < 10:
        print(f"⚠️  Too few clean rows for {comp}; skipping.")
        continue

    # Select best functional form & fit
    form, model, knot = select_form(sub)

    # Fit quality on Δ-time
    X_eval = design_matrix(sub, form, knot)
    y = sub["delta_fe"].values
    yhat = model.predict(X_eval)
    r2 = r2_score(y, yhat); mae = mean_absolute_error(y, yhat)
    slope10 = slope_at(model, form, knot, x_at=10)

    results.append({
        "Compound": comp,
        "BestForm": form,
        "Knot": (knot if knot is not None else ""),
        "MarginalSlope_atLap10_s_per_lap": slope10,
        "R2": r2, "MAE": mae, "n": len(sub)
    })

    # ---- Plot (Δ-time) ----
    plt.figure(figsize=(8,5))
    sns.scatterplot(data=sub, x="tyre_age_laps", y="delta_fe",
                    s=18, alpha=0.45, edgecolor=None)
    med = sub.groupby("tyre_age_laps")["delta_fe"].median()
    plt.plot(med.index, med.values, color="red", lw=1.6, label="Median Δ-time")

    xgrid, ygrid = predict_curve(model, form, knot, sub)
    label = f"{form} fit" + (f" (k={knot})" if knot else "")
    plt.plot(xgrid, ygrid, color="blue", lw=2, label=label)

    plt.title(f"{YEAR} {GRAND_PRIX} — {comp} (Δ-time with FE)")
    plt.xlabel("Laps on tyre")
    plt.ylabel("Δ lap time vs stint baseline (s)")
    plt.legend()
    plt.tight_layout()
    out = FIG_DIR / f"deg_{GRAND_PRIX.lower()}{YEAR}_{comp.lower()}_deltaFE.png"
    plt.savefig(out, dpi=150); plt.close()
    print(f"🖼  saved {out}")

# ------------------------------------
# STEP 5 — Save summary
# ------------------------------------
if results:
    res = pd.DataFrame(results).sort_values("Compound")
    outcsv = FIG_DIR / f"summary_{GRAND_PRIX.lower()}{YEAR}_deltaFE.csv"
    res.to_csv(outcsv, index=False)
    print("\n==============================")
    print("Δ-time + Driver FE summary (interpret at Lap 10):")
    print(res.to_string(index=False))
    print("==============================")
    print(f"📄 wrote {outcsv.resolve()}")
else:
    print("No compound models — check filters/data.")
