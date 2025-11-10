from pathlib import Path
import pandas as pd 
import fastf1
from fastf1 import plotting
import os

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)

from config import (
    YEAR, GRAND_PRIX, SESSION,
    GREEN_FLAG_ONLY, INCLUDE_WEATHER, OUTLIER_LAPTIME_RANGE_S
)

CACHE_DIR = Path("data/cache")
OUT_DIR = Path("data/processed")
OUT_FILE = OUT_DIR / f"laps_{GRAND_PRIX.lower()}{YEAR}.csv"

CACHE_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

fastf1.Cache.enable_cache(CACHE_DIR)
plotting.setup_mpl(misc_mpl_mods=False)

session = fastf1.get_session(YEAR, GRAND_PRIX, SESSION)
# We don't need telemetry for degradation; weather=True is useful
session.load(telemetry=False, weather=True)

laps_raw = session.laps.copy()
weather_raw = session.weather_data.copy()

keep_cols = [
    "Driver","Team","LapNumber","LapTime","LapStartTime",
    "PitInTime","PitOutTime","IsAccurate","Compound","Stint",
    "FreshTyre","TrackStatus"
]

have_cols = [c for c in keep_cols if c in laps_raw.columns]
laps = (
    laps_raw[have_cols]
    .sort_values(["Driver","LapNumber"])
    .reset_index(drop=True)
)

before = len(laps)
laps = laps[laps["LapTime"].notna()]
after = len(laps)

before = len(laps)
laps = laps[laps["IsAccurate"].fillna(False) == True]
after = len(laps)

before = len(laps)
laps = laps[laps["PitInTime"].isna() & laps["PitOutTime"].isna()]
after = len(laps)

valid_compounds = {"HARD","MEDIUM","SOFT","Hard","Medium","Soft"}
before = len(laps)
laps = laps[laps["Compound"].isin(valid_compounds)]
after = len(laps)

laps["compound_norm"] = laps["Compound"].str.upper().map({
    "HARD": "Hard", "MEDIUM": "Medium", "SOFT": "Soft",
    "Hard": "Hard", "Medium": "Medium", "Soft": "Soft"
})

def is_green(track_status):
    if pd.isna(track_status):
        return True
    # TrackStatus is a ';'-separated string of status codes (e.g., '1;6')
    tokens = str(track_status).split(";")
    return "1" in tokens

if GREEN_FLAG_ONLY and "TrackStatus" in laps.columns:
    before = len(laps)
    laps = laps[laps["TrackStatus"].apply(is_green)].copy()
    after = len(laps)
    print(f"🟢 Dropped non-green laps: {before - after}")

laps["lap_time_s"] = laps["LapTime"].dt.total_seconds()

# 4b) Tyre age in laps (within each Driver-Stint block, starting at 1)
laps = laps.sort_values(["Driver","Stint","LapNumber"])
laps["tyre_age_laps"] = laps.groupby(["Driver","Stint"]).cumcount() + 1

# 4c) Session clock (seconds/minutes from race start)
if "LapStartTime" in laps.columns:
    laps["lap_start_s"] = laps["LapStartTime"].dt.total_seconds()
    laps["session_min"] = laps["lap_start_s"] / 60.0
else:
    laps["lap_start_s"] = pd.NA
    laps["session_min"] = pd.NA

if INCLUDE_WEATHER and weather_raw is not None and not weather_raw.empty:
    wcols = ["Time","AirTemp","TrackTemp","Humidity","Pressure","Rainfall","WindSpeed","WindDirection"]
    wcols = [c for c in wcols if c in weather_raw.columns]
    weather = weather_raw[wcols].copy().sort_values("Time")
    weather["time_s"] = weather["Time"].dt.total_seconds()

    # Prepare laps for asof-merge
    laps = laps.sort_values("lap_start_s")
    laps = pd.merge_asof(
        left=laps, right=weather,
        left_on="lap_start_s", right_on="time_s",
        direction="backward", allow_exact_matches=True
    ).drop(columns=[c for c in ["time_s"] if c in weather.columns])

    # Friendly column names
    laps = laps.rename(columns={"TrackTemp":"track_temp_c","AirTemp":"air_temp_c"})
    print("🌡️  Weather merged (asof on LapStartTime).")
else:
    # Ensure columns exist even if weather not merged
    for c in ["track_temp_c","air_temp_c","Humidity","Pressure","Rainfall","WindSpeed","WindDirection"]:
        if c not in laps.columns:
            laps[c] = pd.NA
    print("🌤  Weather merge skipped or unavailable.")

cols_out = [
    "Driver","Team","compound_norm","Stint","tyre_age_laps",
    "LapNumber","lap_time_s","session_min",
    "track_temp_c","air_temp_c","Humidity","Pressure","Rainfall","WindSpeed","WindDirection"
]
# Keep only columns that exist (robust across FastF1 versions)
cols_out = [c for c in cols_out if c in laps.columns]
dataset = laps[cols_out].copy()

# Outlier sanity window (track-specific bounds from config)
lo, hi = OUTLIER_LAPTIME_RANGE_S
before = len(dataset)
dataset = dataset[(dataset["lap_time_s"] >= lo) & (dataset["lap_time_s"] <= hi)]
after = len(dataset)

dataset = dataset.sort_values(["compound_norm","Driver","LapNumber"]).reset_index(drop=True)

dataset.to_csv(OUT_FILE, index=False)
print(f"\n✅ Wrote {OUT_FILE.resolve()}")
print("\n👀 Preview:")
print(dataset.head(10).to_string(index=False))