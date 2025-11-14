# backend/main.py
from __future__ import annotations

from typing import List, Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .engine import (
    load_model_pack,
    list_gps_and_compounds,
    simulate_strategy,
)


app = FastAPI(
    title="Tyre Degradation Strategy API",
    description="FastAPI backend for F1 tyre degradation & strategy simulation.",
    version="0.1.0",
)

# CORS: allow React dev + future Firebase
origins = [
    "http://localhost:5173",
    "http://localhost:3000",
    "https://your-firebase-domain.web.app",
    "https://your-firebase-domain.firebaseapp.com",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------- Pydantic models ----------

class SimulateRequest(BaseModel):
    gp: str = Field(..., description="Grand Prix name as in model pack")
    race_laps: int = Field(..., gt=0)
    pit_loss_s: float = Field(..., gt=0)
    enabled_compounds: List[str] | None = Field(
        default=None,
        description="Subset of ['Soft','Medium','Hard']; None = all available.",
    )
    top_n: int = Field(10, gt=0, le=30)


class StrategyItem(BaseModel):
    stops: int
    stints: List[int]
    compounds: List[str]
    total_time_s: float


class CompoundParamsOut(BaseModel):
    baseline_s: float
    slope_s_per_lap: float
    cap_laps: int


class SimulateResponse(BaseModel):
    gp: str
    race_laps: int
    pit_loss_s: float
    compounds: Dict[str, CompoundParamsOut]
    top_strategies: List[StrategyItem]
    stint_curves: Dict[str, Dict[str, List[float]]]


# ---------- Routes ----------

@app.get("/api/meta")
def get_metadata() -> Dict[str, Any]:
    pack = load_model_pack()
    gps_map = list_gps_and_compounds(pack)
    meta = pack.get("meta", {})
    return {"gps": gps_map, "meta": meta}


@app.post("/api/simulate", response_model=SimulateResponse)
def post_simulate(req: SimulateRequest):
    try:
        sim = simulate_strategy(
            gp_name=req.gp,
            race_laps=req.race_laps,
            pit_loss_s=req.pit_loss_s,
            enabled_compounds=req.enabled_compounds,
            top_n=req.top_n,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))

    compounds_out = {
        name: CompoundParamsOut(
            baseline_s=cp.baseline_s,
            slope_s_per_lap=cp.slope_s_per_lap,
            cap_laps=cp.cap_laps,
        )
        for name, cp in sim.compounds.items()
    }

    top_strats = [
        StrategyItem(
            stops=s.stops,
            stints=list(s.stints),
            compounds=list(s.compounds),
            total_time_s=s.total_time_s,
        )
        for s in sim.top_strategies
    ]

    return SimulateResponse(
        gp=sim.gp,
        race_laps=sim.race_laps,
        pit_loss_s=sim.pit_loss_s,
        compounds=compounds_out,
        top_strategies=top_strats,
        stint_curves=sim.stint_curves,
    )
