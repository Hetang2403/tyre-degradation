# ============================================
# 03_strategy_sim.py — PROFESSOR MODE (Step 5) 👩‍🏫
# Adds:
#   • Piecewise degradation (early/late slopes) with auto knot (from GP CSV)
#   • Slope-aware stint age caps
#   • Optional per-lap stochastic noise (to diversify near-ties)
#   • Still prefers era model pack; falls back to CSV estimation where needed
# ============================================

# ---- Safe working dir ----
import os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
print(f"📂 Working directory set to: {ROOT}")

# ---- Imports ----
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product
from sklearn.linear_model import HuberRegressor
from config import YEAR, GRAND_PRIX, DEGRADATION_THRESHOLD, RANDOM_SEED

np.random.seed(RANDOM_SEED)
sns.set(context="notebook", style="whitegrid")

# ------------------------------------
# Paths
# ------------------------------------
DATA_FILE  = Path(f"data/processed/laps_{GRAND_PRIX.lower()}{YEAR}.csv")
MODEL_FILE = Path("data/processed/models_ground_effect.json")
FIG_DIR    = Path("figures"); FIG_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------
# User knobs (tweak freely)
# ------------------------------------
DEFAULT_RACE_LAPS = 57 if GRAND_PRIX.lower() == "bahrain" else 50
PIT_LOSS_S = 22.0
BASE_MAX_AGE = {"Soft": 22, "Medium": 30, "Hard": 40}

# Enable stochasticity to break ties (seconds per lap std dev)
NOISE_STD_PER_LAP = 0.00   # try 0.01 for more diversity

# Use piecewise when available (needs local GP CSV to estimate early/late slopes)
USE_PIECEWISE = True

# ============================================
# STEP 1 — Parameter sources
# ============================================

def load_params_from_model_pack(gp_name: str):
    """Load baselines/slopes from era model pack (per-track if present, else global)."""
    if not MODEL_FILE.exists():
        return None, None, None
    pack = json.loads(MODEL_FILE.read_text())
    per = pack.get("per_track", {}).get(gp_name)
    if per is None:
        print("ℹ️  Using GLOBAL era averages from model pack.")
        per = pack.get("global")
    if not per:
        return None, None, None
    baselines = {k: float(v["baseline_s"]) for k, v in per.items()}
    slopes    = {k: float(v["slope_s_per_lap"]) for k, v in per.items()}
    meta_note = pack.get("meta", {}).get("notes", "")
    return baselines, slopes, meta_note

def _prep_delta_fe(df):
    """Δ-time within stint (median laps 2–4) + driver FE."""
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
    return df

def estimate_params_from_gp_csv(csv_path: Path):
    """Fallback linear baselines/slopes from this GP CSV."""
    assert csv_path.exists(), f"Missing {csv_path}. Run src/01_extract_prepare.py."
    df = pd.read_csv(csv_path)
    df = _prep_delta_fe(df)

    baselines = (df[df["tyre_age_laps"].between(2,4)]
                   .groupby("compound_norm")["lap_time_s"]
                   .median().to_dict())

    slopes = {}
    for comp in ["Soft","Medium","Hard"]:
        sub = df[df["compound_norm"] == comp]
        if len(sub) < 20:
            continue
        X = sub[["tyre_age_laps"]].values
        y = sub["delta_fe"].values
        m = HuberRegressor(alpha=0.0, epsilon=1.35).fit(X, y)
        slopes[comp] = float(m.coef_[0])

    return baselines, slopes, df, "Estimated from single-GP CSV (Δ-time + driver-FE, Huber)"

def estimate_piecewise_from_df(df):
    """Return {comp: (knot, slope_early, slope_late)} using Huber and MAE grid search."""
    out = {}
    for comp in ["Soft","Medium","Hard"]:
        g = df[df["compound_norm"] == comp]
        if len(g) < 40:
            continue
        x = g["tyre_age_laps"].values.reshape(-1,1)
        y = g["delta_fe"].values

        # Candidate knots (common range; skip extremes)
        ks = np.arange(5, max(6, int(np.nanmax(g["tyre_age_laps"])) - 3))
        if len(ks) == 0:
            continue

        best = None
        for k in ks:
            mask_early = g["tyre_age_laps"] <= k
            mask_late  = g["tyre_age_laps"] >  k
            if mask_early.sum() < 10 or mask_late.sum() < 10:
                continue
            m1 = HuberRegressor(alpha=0.0, epsilon=1.35).fit(g.loc[mask_early, ["tyre_age_laps"]].values,
                                                             g.loc[mask_early, "delta_fe"].values)
            m2 = HuberRegressor(alpha=0.0, epsilon=1.35).fit(g.loc[mask_late, ["tyre_age_laps"]].values,
                                                             g.loc[mask_late, "delta_fe"].values)
            yhat = np.empty_like(y, dtype=float)
            yhat[mask_early.values] = m1.predict(g.loc[mask_early, ["tyre_age_laps"]].values)
            yhat[mask_late.values]  = m2.predict(g.loc[mask_late,  ["tyre_age_laps"]].values)
            mae = np.mean(np.abs(y - yhat))
            if (best is None) or (mae < best[0]):
                best = (mae, int(k), float(m1.coef_[0]), float(m2.coef_[0]))
        if best:
            out[comp] = (best[1], best[2], best[3])  # (knot, early, late)
    return out

# Preferred: model pack
baselines, slopes, source_note = load_params_from_model_pack(GRAND_PRIX)

piecewise = {}
df_csv = None

if baselines is None or slopes is None:
    # Fall back to single GP
    print("ℹ️  Model pack missing → estimating from local CSV.")
    baselines, slopes, df_csv, source_note = estimate_params_from_gp_csv(DATA_FILE)
else:
    # If we also have the local CSV and USE_PIECEWISE, try to get per-GP piecewise
    if USE_PIECEWISE and DATA_FILE.exists():
        df_csv = _prep_delta_fe(pd.read_csv(DATA_FILE))

# Ensure Medium even if unused
if "Medium" not in slopes and {"Soft","Hard"}.issubset(slopes):
    slopes["Medium"]    = (slopes["Soft"] + slopes["Hard"]) / 2.0
if "Medium" not in baselines and {"Soft","Hard"}.issubset(baselines):
    baselines["Medium"] = (baselines["Soft"] + baselines["Hard"]) / 2.0

# Piecewise (if requested and data available)
if USE_PIECEWISE and df_csv is not None:
    piecewise = estimate_piecewise_from_df(df_csv)  # {comp: (knot, early, late)}

COMPOUNDS = [c for c in ["Soft","Medium","Hard"] if c in baselines and c in slopes]
assert len(COMPOUNDS) >= 2, "Need at least two compounds."

print("\n=== Parameter source ===")
print(source_note)
print("Baselines (s):", {k: round(v,3) for k,v in baselines.items()})
print("Linear slopes (s/lap):", {k: round(v,4) for k,v in slopes.items()})
if USE_PIECEWISE and piecewise:
    print("Piecewise:", {k: {"knot":v[0], "early":round(v[1],4), "late":round(v[2],4)} for k,v in piecewise.items()})

# ============================================
# STEP 2 — Race laps & slope-aware caps
# ============================================

def infer_race_laps_from_csv(csv_path: Path):
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path, usecols=["Driver","LapNumber"])
    laps_per_driver = df.groupby("Driver")["LapNumber"].max()
    if laps_per_driver.empty:
        return None
    return int(np.median(laps_per_driver.values))

RACE_LAPS = infer_race_laps_from_csv(DATA_FILE) or DEFAULT_RACE_LAPS
print(f"\nRace laps used: {RACE_LAPS}")

MAX_STINT_AGE = {}
for comp in COMPOUNDS:
    base_cap = BASE_MAX_AGE.get(comp, 35)
    slope_lin = max(abs(slopes[comp]), 1e-6)
    limit = int(np.clip(np.floor(1.6 / slope_lin), 8, 45))  # smaller slope → longer allowable stint
    MAX_STINT_AGE[comp] = min(base_cap, limit)
print("Max stint ages:", MAX_STINT_AGE)

# ============================================
# STEP 3 — stint time calculators
# ============================================

def stint_time_linear(compound: str, L: int) -> float:
    base  = baselines.get(compound, np.nan)
    slope = slopes.get(compound, np.nan)
    if np.isnan(base) or np.isnan(slope) or L <= 0:
        return np.inf
    drive = L * base + slope * (L * (L - 1) / 2.0)
    if NOISE_STD_PER_LAP > 0:
        drive += np.random.normal(0.0, NOISE_STD_PER_LAP) * L
    return drive

def stint_time_piecewise(compound: str, L: int) -> float:
    """Piecewise: early slope to knot k, then late slope. Continuous at the knot."""
    if compound not in piecewise:
        return stint_time_linear(compound, L)
    base = baselines.get(compound, np.nan)
    k, s1, s2 = piecewise[compound]
    if np.isnan(base) or L <= 0:
        return np.inf
    k = int(max(1, k))
    if L <= k:
        # sum i=1..L [base + s1*(i-1)]
        drive = L * base + s1 * (L * (L - 1) / 2.0)
    else:
        # segment 1: i=1..k
        sum1 = k * base + s1 * (k * (k - 1) / 2.0)
        # segment 2: i=k+1..L
        m = L - k
        # value at i = base + s1*(k-1) + s2*(i - k)
        sum2 = m * (base + s1*(k - 1)) + s2 * (m * (m + 1) / 2.0)
        drive = sum1 + sum2
    if NOISE_STD_PER_LAP > 0:
        drive += np.random.normal(0.0, NOISE_STD_PER_LAP) * L
    return drive

def stint_time(compound: str, L: int) -> float:
    return stint_time_piecewise(compound, L) if (USE_PIECEWISE and compound in piecewise) else stint_time_linear(compound, L)

# Helpers
def valid_two_compounds(seq):
    return len(set(seq)) >= 2

def splits_two(total, max_len):
    for a in range(1, total):
        b = total - a
        if a <= max_len and b <= max_len:
            yield (a, b)

def splits_three(total, max_len):
    for a in range(1, total-1):
        for b in range(1, total-a):
            c = total - a - b
            if a <= max_len and b <= max_len and c <= max_len:
                yield (a, b, c)

def strategy_time(stint_laps, compounds):
    if not valid_two_compounds(compounds):
        return np.inf
    for L, comp in zip(stint_laps, compounds):
        if L > MAX_STINT_AGE.get(comp, 99):
            return np.inf
    drive = sum(stint_time(comp, L) for L, comp in zip(stint_laps, compounds))
    pits  = (len(stint_laps) - 1) * PIT_LOSS_S
    return drive + pits

# ============================================
# STEP 4 — Search strategies
# ============================================
records = []
cap = max(MAX_STINT_AGE.values())

# 1-stop
for (a, b) in splits_two(RACE_LAPS, cap):
    for c1, c2 in product(COMPOUNDS, repeat=2):
        t = strategy_time((a, b), (c1, c2))
        if np.isfinite(t):
            records.append({"Stops": 1, "Stints": (a, b), "Compounds": (c1, c2), "TotalTime_s": t})

# 2-stop
for (a, b, c) in splits_three(RACE_LAPS, cap):
    for c1, c2, c3 in product(COMPOUNDS, repeat=3):
        t = strategy_time((a, b, c), (c1, c2, c3))
        if np.isfinite(t):
            records.append({"Stops": 2, "Stints": (a, b, c), "Compounds": (c1, c2, c3), "TotalTime_s": t})

res = pd.DataFrame(records).sort_values("TotalTime_s").reset_index(drop=True)

# ============================================
# STEP 5 — Output & plots
# ============================================
print("\n=== Per-compound parameters used ===")
for c in COMPOUNDS:
    pw = piecewise.get(c)
    if USE_PIECEWISE and pw:
        print(f"{c:>6} | base={baselines[c]:.3f} s | early={pw[1]:+.4f} | late={pw[2]:+.4f} | knot={pw[0]} | cap={MAX_STINT_AGE[c]}")
    else:
        print(f"{c:>6} | base={baselines[c]:.3f} s | slope={slopes[c]:+.4f} | cap={MAX_STINT_AGE[c]}")
print(f"Pit loss assumed: {PIT_LOSS_S:.1f} s")

topN = 12
print(f"\n=== Top {topN} strategies (lower total time is better) ===")
for i, row in res.head(topN).iterrows():
    st = "-".join(map(str, row["Stints"]))
    cp = "-".join(row["Compounds"])
    print(f"{i+1:>2}. {row['Stops']}-stop | stints {st} | {cp} | total {row['TotalTime_s']:.1f} s")

out_csv = FIG_DIR / f"strategies_{GRAND_PRIX.lower()}{YEAR}.csv"
res.to_csv(out_csv, index=False)
print(f"\n📄 wrote {out_csv.resolve()}")

# Stint curves
for comp in COMPOUNDS:
    xmax = min(MAX_STINT_AGE[comp], 45)
    ages = np.arange(1, xmax+1)
    if USE_PIECEWISE and comp in piecewise:
        k, s1, s2 = piecewise[comp]
        # Build curve
        curve = []
        for i in ages:
            if i <= k:
                curve.append(baselines[comp] + s1 * (i - 1))
            else:
                curve.append(baselines[comp] + s1 * (k - 1) + s2 * (i - k))
        y = np.array(curve)
        title_extra = f"piecewise (k={k})"
    else:
        y = baselines[comp] + slopes[comp] * (ages - 1)
        title_extra = "linear"
    plt.figure(figsize=(7,4))
    plt.plot(ages, y, lw=2)
    plt.title(f"{YEAR} {GRAND_PRIX} — {comp} predicted stint curve ({title_extra})")
    plt.xlabel("Laps on tyre"); plt.ylabel("Lap time (s)")
    plt.tight_layout()
    out = FIG_DIR / f"stint_curves_{GRAND_PRIX.lower()}{YEAR}_{comp.lower()}.png"
    plt.savefig(out, dpi=150); plt.close()

# Strategy bar chart
plt.figure(figsize=(11,6))
top = res.head(8).copy()
labels = [f"{r.Stops}-stop | { '-'.join(map(str,r.Stints)) } | { '-'.join(r.Compounds) }" for _, r in top.iterrows()]
plt.barh(range(len(top)), top["TotalTime_s"])
plt.gca().invert_yaxis()
plt.yticks(range(len(top)), labels, fontsize=9)
plt.xlabel("Total race time (s)")
plt.title(f"{YEAR} {GRAND_PRIX}: strategy comparison (lower is better)")
plt.tight_layout()
out = FIG_DIR / f"strategy_compare_{GRAND_PRIX.lower()}{YEAR}.png"
plt.savefig(out, dpi=150); plt.close()
print(f"🖼  saved {out}")
