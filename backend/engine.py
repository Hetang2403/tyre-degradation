# backend/engine.py
# ============================================
# Core tyre strategy engine as reusable functions
# Uses:
#   • data/processed/models_ground_effect.json (era model pack)
#   • simple linear degradation per compound
# ============================================

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import json
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODEL_FILE = ROOT / "data/processed/models_ground_effect.json"


@dataclass
class CompoundParams:
    baseline_s: float        # base rolling lap time on fresh tyre
    slope_s_per_lap: float   # degradation rate (Δ s / lap)
    cap_laps: int            # recommended maximum stint length


@dataclass
class StrategyResult:
    stops: int
    stints: Tuple[int, ...]         # e.g. (16, 40)
    compounds: Tuple[str, ...]      # e.g. ("Soft", "Hard")
    total_time_s: float


@dataclass
class SimulationOutput:
    gp: str
    race_laps: int
    pit_loss_s: float
    compounds: Dict[str, CompoundParams]
    top_strategies: List[StrategyResult]
    stint_curves: Dict[str, Dict[str, List[float]]]  # {comp: {ages, times}}


# ---------- Model pack loading ----------

def load_model_pack() -> dict:
    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Model pack not found at {MODEL_FILE}. "
            "Run 00_build_corpus.py and 02b_fit_global_models.py first."
        )
    return json.loads(MODEL_FILE.read_text())


def get_params_for_gp(pack: dict, gp_name: str) -> Dict[str, Dict[str, float]]:
    """
    Returns dict: {compound: {baseline_s, slope_s_per_lap}}.
    Prefers per-track; falls back to global averages.
    """
    per_track = pack.get("per_track", {})
    params = per_track.get(gp_name)
    if params is None:
        params = pack.get("global")
    if params is None:
        raise ValueError(f"No parameters for GP '{gp_name}' and no global fallback.")
    return params


# ---------- Strategy maths ----------

def compute_caps(slopes: Dict[str, float]) -> Dict[str, int]:
    """
    Simple slope-aware caps: steeper slope → shorter cap.
    Heuristic: cap ≈ min(base_cap, floor(1.6 / |slope|)), clipped [8, 45].
    """
    base_max = {"Soft": 22, "Medium": 30, "Hard": 40}
    caps: Dict[str, int] = {}

    for comp, slope in slopes.items():
        base_cap = base_max.get(comp, 35)
        s = abs(float(slope))
        s = max(s, 1e-6)
        limit = int(np.clip(np.floor(1.6 / s), 8, 45))
        caps[comp] = min(base_cap, limit)
    return caps


def stint_time_linear(baseline: float, slope: float, L: int) -> float:
    """Sum_{i=1..L} (baseline + slope*(i-1))."""
    if L <= 0:
        return float("inf")
    return L * baseline + slope * (L * (L - 1) / 2.0)


def _valid_two_compounds(seq: Tuple[str, ...]) -> bool:
    return len(set(seq)) >= 2


def _splits_two(total: int, max_len: int):
    for a in range(1, total):
        b = total - a
        if a <= max_len and b <= max_len:
            yield (a, b)


def _splits_three(total: int, max_len: int):
    for a in range(1, total - 1):
        for b in range(1, total - a):
            c = total - a - b
            if a <= max_len and b <= max_len and c <= max_len:
                yield (a, b, c)


def _strategy_time(
    stint_laps: Tuple[int, ...],
    compounds: Tuple[str, ...],
    baselines: Dict[str, float],
    slopes: Dict[str, float],
    caps: Dict[str, int],
    pit_loss_s: float,
) -> float:
    if not _valid_two_compounds(compounds):
        return float("inf")

    drive = 0.0
    for L, comp in zip(stint_laps, compounds):
        if L > caps.get(comp, 99):
            return float("inf")
        base = baselines.get(comp)
        slope = slopes.get(comp)
        if base is None or slope is None:
            return float("inf")
        drive += stint_time_linear(base, slope, L)

    pits = (len(stint_laps) - 1) * pit_loss_s
    return drive + pits


# ---------- Public API ----------

def list_gps_and_compounds(pack: dict) -> Dict[str, List[str]]:
    """Return {gp_name: [compounds]} based on per_track section."""
    gps = {}
    per_track = pack.get("per_track", {})
    for gp_name, comp_dict in per_track.items():
        gps[gp_name] = sorted(comp_dict.keys())
    global_dict = pack.get("global", {})
    gps["_global"] = sorted(global_dict.keys())
    return gps


def simulate_strategy(
    gp_name: str,
    race_laps: int,
    pit_loss_s: float,
    enabled_compounds: Optional[List[str]] = None,
    top_n: int = 10,
) -> SimulationOutput:
    """
    Main function used by FastAPI & CLI.
    """
    pack = load_model_pack()
    raw_params = get_params_for_gp(pack, gp_name)

    baselines: Dict[str, float] = {}
    slopes: Dict[str, float] = {}

    for comp, vals in raw_params.items():
        baselines[comp] = float(vals["baseline_s"])
        slopes[comp] = float(vals["slope_s_per_lap"])

    if "Medium" not in baselines and {"Soft", "Hard"}.issubset(baselines):
        baselines["Medium"] = (baselines["Soft"] + baselines["Hard"]) / 2.0
        slopes["Medium"] = (slopes["Soft"] + slopes["Hard"]) / 2.0

    if enabled_compounds:
        comps = [c for c in enabled_compounds if c in baselines]
    else:
        comps = [c for c in ["Soft", "Medium", "Hard"] if c in baselines]

    if len(comps) < 2:
        raise ValueError("Need at least two enabled compounds for a legal dry strategy.")

    caps = compute_caps({c: slopes[c] for c in comps})

    records: List[StrategyResult] = []
    max_cap = max(caps.values())

    # 1-stop
    for a, b in _splits_two(race_laps, max_cap):
        for c1 in comps:
            for c2 in comps:
                t = _strategy_time((a, b), (c1, c2), baselines, slopes, caps, pit_loss_s)
                if np.isfinite(t):
                    records.append(StrategyResult(1, (a, b), (c1, c2), float(t)))

    # 2-stop
    for a, b, c in _splits_three(race_laps, max_cap):
        for c1 in comps:
            for c2 in comps:
                for c3 in comps:
                    t = _strategy_time((a, b, c), (c1, c2, c3), baselines, slopes, caps, pit_loss_s)
                    if np.isfinite(t):
                        records.append(StrategyResult(2, (a, b, c), (c1, c2, c3), float(t)))

    records_sorted = sorted(records, key=lambda r: r.total_time_s)[:top_n]

    # stint curves for plotting
    stint_curves: Dict[str, Dict[str, List[float]]] = {}
    for comp in comps:
        cap = caps[comp]
        ages = list(range(1, min(cap, 45) + 1))
        base = baselines[comp]
        slope = slopes[comp]
        times = [base + slope * (i - 1) for i in ages]
        stint_curves[comp] = {"ages": ages, "times": times}

    compounds_out: Dict[str, CompoundParams] = {}
    for comp in comps:
        compounds_out[comp] = CompoundParams(
            baseline_s=baselines[comp],
            slope_s_per_lap=slopes[comp],
            cap_laps=caps[comp],
        )

    return SimulationOutput(
        gp=gp_name,
        race_laps=race_laps,
        pit_loss_s=pit_loss_s,
        compounds=compounds_out,
        top_strategies=records_sorted,
        stint_curves=stint_curves,
    )
