from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)

from sklearn.linear_model import HuberRegressor
from sklearn.metrics import r2_score, mean_absolute_error

from config import YEAR, GRAND_PRIX, DEGRADATION_THRESHOLD, RANDOM_SEED

sns.set(context="notebook", style="whitegrid")
np.random.seed(RANDOM_SEED)

DATA_FILE = Path(f"data/processed/laps_{GRAND_PRIX.lower()}{YEAR}.csv")
FIG_DIR = Path("figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_FILE)

# ✅ Professor check:
expected_cols = {"compound_norm","tyre_age_laps","lap_time_s"}
missing = expected_cols - set(df.columns)
if missing:
    raise ValueError(f"Dataset is missing columns: {missing}. "
                     "Re-run step 01 to rebuild the tidy CSV.")

# Keep only the essentials we model on
df = df.dropna(subset=["lap_time_s", "tyre_age_laps", "compound_norm"]).copy()

COMPOUNDS = ["Soft", "Medium", "Hard"]

def fit_compound_linear_robust(sub_df):
    """
    Fit a robust linear regression: lap_time_s ~ tyre_age_laps.

    Returns:
        dict with slope, intercept, r2, mae, n.
    """
    X = sub_df[["tyre_age_laps"]].values
    y = sub_df["lap_time_s"].values

    # Huber is robust to outliers (epsilon ~ how tolerant we are)
    model = HuberRegressor(alpha=0.0, epsilon=1.35)
    model.fit(X, y)
    yhat = model.predict(X)

    out = {
        "Slope_s_per_lap": float(model.coef_[0]),     # degradation rate
        "Intercept": float(model.intercept_),
        "R2": float(r2_score(y, yhat)),
        "MAE": float(mean_absolute_error(y, yhat)),
        "n": int(len(sub_df))
    }
    return model, out

def plot_compound(sub_df, model, comp_name):
    """
    Scatter: lap_time_s vs tyre_age_laps + fitted line.
    Saves figures/deg_<gp><year>_<compound>.png
    """
    xmax = int(np.nanmax(sub_df["tyre_age_laps"]))
    grid = np.arange(1, xmax + 1)
    pred = model.predict(grid.reshape(-1, 1))

    plt.figure(figsize=(8, 5))
    sns.scatterplot(
        data=sub_df, x="tyre_age_laps", y="lap_time_s",
        s=18, alpha=0.55, edgecolor=None
    )
    plt.plot(grid, pred, linewidth=2)
    plt.title(f"{YEAR} {GRAND_PRIX} — {comp_name} degradation")
    plt.xlabel("Laps on tyre")
    plt.ylabel("Lap time (s)")
    out = FIG_DIR / f"deg_{GRAND_PRIX.lower()}{YEAR}_{comp_name.lower()}.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()

results = []
for comp in COMPOUNDS:
    sub = df[df["compound_norm"] == comp].copy()
    if sub.empty:
        print(f"⚠️  No laps for {comp}; skipping.")
        continue

    # Professor Tip:
    # Consider trimming very early out-laps (tyre_age_laps == 1) if you see pit-exit noise.
    # For now, we keep them because step 01 already removed pit in/out laps.

    model, stats = fit_compound_linear_robust(sub)
    stats["Compound"] = comp
    results.append(stats)

    # Human-readable console summary
    slope = stats["Slope_s_per_lap"]
    r2 = stats["R2"]
    mae = stats["MAE"]
    n = stats["n"]
    print(f"\n🧪 {comp} — n={n}")
    print(f"   slope (s/lap): {slope:.3f}   (this IS the degradation rate)")
    print(f"   R²: {r2:.3f}   MAE: {mae:.2f}")

    # Plot and save
    plot_compound(sub, model, comp)

if not results:
    raise SystemExit("No compound models were produced. Check the data or filters in step 01.")

summary = (
    pd.DataFrame(results)[["Compound","Slope_s_per_lap","Intercept","R2","MAE","n"]]
    .sort_values("Compound")
    .reset_index(drop=True)
)

# Pit-window heuristic (simple, per-compound)
summary["PitWindow_trigger?"] = summary["Slope_s_per_lap"] >= DEGRADATION_THRESHOLD

out_csv = FIG_DIR / f"summary_{GRAND_PRIX.lower()}{YEAR}.csv"
summary.to_csv(out_csv, index=False)

print("\n==============================")
print("Degradation summary (per compound)")
print(summary.to_string(index=False))
print("==============================")
print(f"📄 wrote {out_csv.resolve()}")