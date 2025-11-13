# ============================================
# 00_build_corpus.py — hardened version 👩‍🏫
# Build a multi-season F1 degradation corpus (2022–2024).
# Safely handles sessions missing LapTime / variant schemas.
# ============================================

import os
from pathlib import Path
import numpy as np
import pandas as pd
import fastf1
from fastf1 import plotting
from fastf1.core import DataNotLoadedError


ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
print(f"📂 WD: {ROOT}")

YEARS = [2022, 2023, 2024]
SESSION = "R"  # race only
CACHE_DIR = Path("data/cache"); CACHE_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR   = Path("data/processed"); OUT_DIR.mkdir(parents=True, exist_ok=True)
CORPUS_FILE = OUT_DIR / "laps_corpus_2022_2024.parquet"

fastf1.Cache.enable_cache(CACHE_DIR)
plotting.setup_mpl(misc_mpl_mods=False)

def _to_seconds(series):
    """Robustly convert a pandas Series to seconds (float)."""
    if series is None:
        return None
    s = series.copy()
    # already timedelta?
    if np.issubdtype(s.dtype, np.timedelta64):
        return s.dt.total_seconds()
    # sometimes comes as string like '0:01:32.345000'
    try:
        return pd.to_timedelta(s, errors="coerce").dt.total_seconds()
    except Exception:
        return None


def clean_session(year, gp_name):
    """Load one race, return tidy DataFrame or None (skip if unusable)."""
    try:
        sess = fastf1.get_session(year, gp_name, SESSION)
        # Ask explicitly for laps; don't load telemetry (faster), do load weather.
        sess.load(telemetry=False, weather=True)   # laps should be included
    except Exception as e:
        print(f"⚠️  Skip {year} {gp_name}: load error -> {e}")
        return None

    # Some sessions still won't have laps; guard access
    try:
        laps = sess.laps.copy()
    except DataNotLoadedError as e:
        print(f"⚠️  Skip {year} {gp_name}: laps not available -> {e}")
        return None
    except Exception as e:
        print(f"⚠️  Skip {year} {gp_name}: laps access failed -> {e}")
        return None

    if laps is None or laps.empty:
        print(f"⚠️  Skip {year} {gp_name}: no laps in dataset")
        return None

    # Columns seen across versions
    possible = [
        "Driver","Team","LapNumber","LapTime","Time",
        "LapStartTime","PitInTime","PitOutTime","IsAccurate",
        "Compound","Stint","FreshTyre","TrackStatus"
    ]
    keep = [c for c in possible if c in laps.columns]
    laps = laps[keep].sort_values(["Driver","LapNumber"]).reset_index(drop=True)

    # Compute lap_time_s robustly
    lap_time_s = None
    if "LapTime" in laps.columns:
        lap_time_s = _to_seconds(laps["LapTime"])
    if lap_time_s is None or lap_time_s.isna().all():
        # Some sessions expose only "Time" deltas per lap
        if "Time" in laps.columns:
            lap_time_s = _to_seconds(laps["Time"])
    if lap_time_s is None or lap_time_s.isna().all():
        print(f"⚠️  Skip {year} {gp_name}: missing usable LapTime/Time")
        return None
    laps["lap_time_s"] = lap_time_s

    # Filter: valid, accurate, not pit-in/out
    if "IsAccurate" in laps.columns:
        laps = laps[laps["IsAccurate"].fillna(False) == True]
    if "PitInTime" in laps.columns:
        laps = laps[laps["PitInTime"].isna()]
    if "PitOutTime" in laps.columns:
        laps = laps[laps["PitOutTime"].isna()]

    # Normalise compounds
    valid_compounds = {"HARD","MEDIUM","SOFT","Hard","Medium","Soft"}
    if "Compound" not in laps.columns:
        print(f"⚠️  Skip {year} {gp_name}: no Compound column")
        return None
    laps = laps[laps["Compound"].isin(valid_compounds)]
    if laps.empty:
        print(f"⚠️  Skip {year} {gp_name}: no primary compound laps")
        return None
    laps["compound_norm"] = laps["Compound"].str.upper().map(
        {"HARD":"Hard","MEDIUM":"Medium","SOFT":"Soft",
         "HARD":"Hard","MEDIUM":"Medium","SOFT":"Soft"}
    )

    # Green-flag only (optional; keep if available)
    def is_green(ts):
        if pd.isna(ts): return True
        return "1" in str(ts).split(";")
    if "TrackStatus" in laps.columns:
        laps = laps[laps["TrackStatus"].apply(is_green)]

    # Tyre age (per stint)
    if "Stint" not in laps.columns:
        # create synthetic stint index if missing (rare)
        laps["Stint"] = laps.groupby("Driver")["Compound"].ne(
            laps.groupby("Driver")["Compound"].shift()
        ).cumsum()
    laps = laps.sort_values(["Driver","Stint","LapNumber"])
    laps["tyre_age_laps"] = laps.groupby(["Driver","Stint"]).cumcount() + 1

    # Session minute
    if "LapStartTime" in laps.columns:
        ls = _to_seconds(laps["LapStartTime"])
        laps["lap_start_s"] = ls
        laps["session_min"] = ls / 60.0
    else:
        laps["session_min"] = np.nan

    # Merge weather (nearest prior) if present
    weather = getattr(sess, "weather_data", None)
    if weather is not None and not weather.empty and "Time" in weather.columns:
        w = weather.sort_values("Time").copy()
        w["time_s"] = _to_seconds(w["Time"])
        cols = [c for c in ["AirTemp","TrackTemp","Humidity","Pressure","Rainfall",
                            "WindSpeed","WindDirection"] if c in w.columns]
        if "lap_start_s" in laps.columns:
            laps = laps.sort_values("lap_start_s")
            laps = pd.merge_asof(
                laps, w[["time_s"]+cols], left_on="lap_start_s",
                right_on="time_s", direction="backward"
            )
            laps = laps.rename(columns={"TrackTemp":"track_temp_c","AirTemp":"air_temp_c"})
        else:
            # cannot align safely, set NaN weather
            for c in ["track_temp_c","air_temp_c","Humidity","Pressure","Rainfall","WindSpeed","WindDirection"]:
                laps[c] = np.nan
    else:
        for c in ["track_temp_c","air_temp_c","Humidity","Pressure","Rainfall","WindSpeed","WindDirection"]:
            laps[c] = np.nan

    out = laps[[
        "Driver","Team","compound_norm","Stint","tyre_age_laps","LapNumber",
        "lap_time_s","session_min","track_temp_c","air_temp_c",
        "Humidity","Pressure","Rainfall","WindSpeed","WindDirection"
    ]].copy()

    # Guardrails
    out = out[out["lap_time_s"].between(50, 200)]
    out["Year"]  = year
    out["GP"]    = sess.event["EventName"] if "EventName" in sess.event else gp_name
    out["Round"] = sess.event.get("RoundNumber", np.nan) if hasattr(sess, "event") else np.nan
    return out.reset_index(drop=True)

rows = []
for yr in YEARS:
    print(f"\n=== {yr} schedule ===")
    try:
        schedule = fastf1.get_event_schedule(yr)
    except Exception as e:
        print(f"⚠️ schedule issue {yr}: {e}")
        continue

    gps = schedule["EventName"].tolist()
    for gp in gps:
        print(f"→ {yr} {gp}")
        df_gp = clean_session(yr, gp)
        if df_gp is None or df_gp.empty:
            continue
        short = f"{gp.lower().replace(' ','_')}{yr}"
        per_file = OUT_DIR / f"laps_{short}.csv"
        df_gp.to_csv(per_file, index=False)
        rows.append(df_gp)

# Save corpus
if rows:
    corpus = pd.concat(rows, ignore_index=True)
    CORPUS_FILE.parent.mkdir(parents=True, exist_ok=True)
    corpus.to_parquet(CORPUS_FILE, index=False)
    print(f"\n✅ wrote {CORPUS_FILE.resolve()}  (rows={len(corpus):,})")
else:
    print("No sessions saved. Check network/cache and try again.")
