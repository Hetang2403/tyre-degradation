# ============================================
# 02b_fit_global_models.py — PROFESSOR MODE 👩‍🏫
# Fit per-track + global degradation parameters (2022–2024 corpus)
# Uses:
#   • Baselines from laps 2–4 (absolute lap_time_s)
#   • Slopes from Δ-time with driver FE (Huber linear)
# Handles:
#   • Missing compounds per GP via Soft/Hard interpolation
#   • Final fallback to global means per compound
# Saves:
#   • data/processed/models_ground_effect.json
# ============================================

import os
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import HuberRegressor

# ---- safe working dir ----
ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
print(f"📂 WD: {ROOT}")

CORPUS_FILE = Path("data/processed/laps_corpus_2022_2024.parquet")
MODEL_FILE  = Path("data/processed/models_ground_effect.json")

assert CORPUS_FILE.exists(), "Run src/00_build_corpus.py first to create the corpus."
df = pd.read_parquet(CORPUS_FILE)
print(f"📦 Loaded corpus: {len(df):,} rows")

# ------------------------------------
# Δ-time within stint + driver FE
# ------------------------------------
def stint_baseline(g):
    ref = g.loc[g["tyre_age_laps"].between(2,4), "lap_time_s"]
    base = ref.median() if len(ref) else g["lap_time_s"].head(4).median()
    g["delta_lap_time"] = g["lap_time_s"] - base
    return g

df = (df.sort_values(["Driver","Stint","LapNumber"])
        .groupby(["Driver","Stint"], as_index=False)
        .apply(stint_baseline)
        .reset_index(drop=True))

drv_med = df.groupby("Driver")["delta_lap_time"].median().rename("drv_med")
df = df.merge(drv_med, on="Driver", how="left")
df["delta_fe"] = df["delta_lap_time"] - df["drv_med"]
df = df[df["tyre_age_laps"] >= 1]
df = df[df["delta_fe"].between(-5, 5)]

# ------------------------------------
# Baselines & slopes (GP × compound)
# ------------------------------------
# Baseline = median absolute pace at laps 2–4
baseline_tbl = (
    df[df["tyre_age_laps"].between(2,4)]
    .groupby(["GP","compound_norm"])["lap_time_s"]
    .median()
    .reset_index(name="baseline_s")
)

def fit_slope(group):
    if len(group) < 30:
        return np.nan
    X = group[["tyre_age_laps"]].values
    y = group["delta_fe"].values
    m = HuberRegressor(alpha=0.0, epsilon=1.35)
    m.fit(X, y)
    return float(m.coef_[0])

slope_tbl = (
    df.groupby(["GP","compound_norm"])
      .apply(fit_slope)
      .reset_index(name="slope_s_per_lap")
)

params = baseline_tbl.merge(slope_tbl, on=["GP","compound_norm"], how="outer")

# ------------------------------------
# Fill unused/missing compounds per GP
# ------------------------------------
ALL_COMPS = ["Soft","Medium","Hard"]

def interpolate_for_gp(gp: str, grp: pd.DataFrame):
    """
    grp: rows for one GP (cols: GP, compound_norm, baseline_s, slope_s_per_lap).
    Returns list of rows covering Soft/Medium/Hard; Medium is interpolated
    from Soft/Hard when missing; Soft/Hard can remain NaN here (filled later
    with global means).
    """
    rows = []

    def get(col, comp):
        sub = grp[grp["compound_norm"] == comp]
        if sub.empty:
            return np.nan
        return float(sub.iloc[0][col])

    soft_base = get("baseline_s", "Soft")
    hard_base = get("baseline_s", "Hard")
    soft_slope = get("slope_s_per_lap", "Soft")
    hard_slope = get("slope_s_per_lap", "Hard")

    med_base  = np.nanmean([soft_base, hard_base])
    med_slope = np.nanmean([soft_slope, hard_slope])

    for comp in ALL_COMPS:
        base  = get("baseline_s", comp)
        slope = get("slope_s_per_lap", comp)

        if comp == "Medium":
            if np.isnan(base):
                base = med_base
            if np.isnan(slope):
                slope = med_slope

        rows.append({
            "GP": gp,
            "compound_norm": comp,
            "baseline_s": base,
            "slope_s_per_lap": slope,
        })
    return rows

per_gp_rows = []
for gp, grp in params.groupby("GP"):
    per_gp_rows.extend(interpolate_for_gp(gp, grp))

filled = pd.DataFrame(per_gp_rows)

# ------------------------------------
# Global means per compound as final fallback
# ------------------------------------
global_means = (
    filled.groupby("compound_norm")[["baseline_s","slope_s_per_lap"]]
          .mean(numeric_only=True)
)

def fill_from_global(row):
    g = global_means.loc[row["compound_norm"]]
    if np.isnan(row["baseline_s"]):
        row["baseline_s"] = float(g["baseline_s"])
    if np.isnan(row["slope_s_per_lap"]):
        row["slope_s_per_lap"] = float(g["slope_s_per_lap"])
    return row

filled = filled.apply(fill_from_global, axis=1)

# ------------------------------------
# Also build GLOBAL (era-average) params
# ------------------------------------
global_pack_rows = []
for comp in ALL_COMPS:
    g_comp = df[df["compound_norm"] == comp]
    if g_comp.empty:
        continue
    # baseline: median of laps 2–4 for that compound across era
    base = g_comp[g_comp["tyre_age_laps"].between(2,4)]["lap_time_s"].median()
    if g_comp["delta_fe"].dropna().shape[0] < 30:
        continue
    X = g_comp[["tyre_age_laps"]].values
    y = g_comp["delta_fe"].values
    m = HuberRegressor(alpha=0.0, epsilon=1.35).fit(X, y)
    slope = float(m.coef_[0])
    global_pack_rows.append({
        "compound_norm": comp,
        "baseline_s": float(base),
        "slope_s_per_lap": slope,
    })

global_pack = pd.DataFrame(global_pack_rows)

# ------------------------------------
# Pack everything into JSON
# ------------------------------------
model = {
    "meta": {
        "era": "2022-2024",
        "notes": (
            "Baselines from laps 2–4 (absolute lap_time_s); "
            "slopes from Δ-time with driver FE (Huber linear). "
            "Per-track Medium interpolated from Soft/Hard where missing; "
            "remaining NaNs filled with era means."
        )
    },
    "global": {
        row["compound_norm"]: {
            "baseline_s": float(row["baseline_s"]),
            "slope_s_per_lap": float(row["slope_s_per_lap"]),
        }
        for _, row in global_pack.iterrows()
    },
    "per_track": {}
}

for gp, grp in filled.groupby("GP"):
    model["per_track"][gp] = {
        row["compound_norm"]: {
            "baseline_s": float(row["baseline_s"]),
            "slope_s_per_lap": float(row["slope_s_per_lap"]),
        }
        for _, row in grp.iterrows()
    }

MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
MODEL_FILE.write_text(json.dumps(model, indent=2))
print(f"✅ wrote {MODEL_FILE.resolve()}")
