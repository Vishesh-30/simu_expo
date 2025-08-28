from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List, Optional, Union, Dict
from monte_carlo import MonteCarloSimulator
from range_forecast import range_forecast_calc

app = FastAPI()

# --- Pydantic Models for API Request Structure ---

class DistributionParam(BaseModel):
    dist_type: str
    low: Optional[float] = None
    high: Optional[float] = None
    base: Optional[float] = None
    mean: Optional[float] = None
    std: Optional[float] = None

class PrimaryMarketConfig(BaseModel):
    market_name: str
    region: str
    baseline_volume: float = Field(..., gt=0)
    event_params: Optional[List[DistributionParam]] = []
    class_share_param: DistributionParam
    product_share_param: DistributionParam
    gross_price_param: Union[DistributionParam, List[DistributionParam]]
    gtn_param: Union[DistributionParam, List[DistributionParam]]
    sku_splits: Optional[List[float]] = None

# --- MODIFIED MODEL ---
class SecondaryMarketConfig(BaseModel):
    """Configuration for a secondary market (simplified)."""
    market_name: str
    scale_up_region: str
    scale_up_param: DistributionParam 

class SimulationRequest(BaseModel):
    n_simulations: int = Field(..., gt=0)
    primary_market_configs: List[PrimaryMarketConfig]
    secondary_market_configs: Optional[List[SecondaryMarketConfig]] = None

@app.post("/simulate")
async def simulate(request: SimulationRequest):
    primary_configs = [market.dict() for market in request.primary_market_configs]
    secondary_configs = [market.dict() for market in request.secondary_market_configs] if request.secondary_market_configs else None
    
    sim = MonteCarloSimulator(
        n_simulations=request.n_simulations,
        primary_market_configs=primary_configs,
        secondary_market_configs=secondary_configs
    )
    result = sim.run()
    return result

# ---------- NEW REQUEST MODEL FOR RANGE FORECAST ----------
class RangeForecastRequest(BaseModel):
    base_peak: float
    base_netsales: List[float]
    summary: Dict[str, float]

@app.post("/range_forecast")
async def range_forecast(request: RangeForecastRequest):
    range_forecasts = range_forecast_calc(request.base_peak, request.base_netsales, request.summary)
    return range_forecasts





# Input format for the API

# {
#   "n_simulations": 1000,
#   "final_net_sales": 10000,
#   "event_params": [
#     {"dist_type": "uniform", "low": 0.04, "high": 0.15, "base": 0.10},
#     {"dist_type": "uniform", "low": 0.06, "high": 0.10, "base": 0.08},
#     {"dist_type": "uniform", "low": 0.10, "high": 0.10, "base": 0.10}
#   ],
#   "class_share_param": {"dist_type": "uniform", "low": 0.2, "high": 0.8, "base": 0.3},
#   "product_share_param": {"dist_type": "uniform", "low": 0.4, "high": 0.5, "base": 0.3},
#   "gross_price_param": {"dist_type": "uniform", "low": 200, "high": 200, "base": 200},
#   "gtn_param": {"dist_type": "uniform", "low": 0.1, "high": 0.3, "base": 0.2}
# }


# {
#   "n_simulations": 1000,
#   "final_net_sales": 10000,
#   "event_params": [
#     {"dist_type": "normal", "low": 0.04, "high": 0.15, "base": 0.10},
#     {"dist_type": "uniform", "low": 0.06, "high": 0.10, "base": 0.08},
#     {"dist_type": "uniform", "low": 0.10, "high": 0.10, "base": 0.10}
#   ],
#   "class_share_param": {"dist_type": "uniform", "low": 0.2, "high": 0.8, "base": 0.3},
#   "product_share_param": {"dist_type": "uniform", "low": 0.4, "high": 0.5, "base": 0.3},
#   "sku_splits": [0.3, 0.7],
#   "gross_price_param": [
#     {"dist_type": "uniform", "low": 180, "high": 220, "base": 200},
#     {"dist_type": "uniform", "low": 150, "high": 250, "base": 200}
#   ],
#   "gtn_param": [
#     {"dist_type": "uniform", "low": 0.1, "high": 0.2, "base": 0.15},
#     {"dist_type": "uniform", "low": 0.2, "high": 0.3, "base": 0.25}
#   ]
# }



# distribution_params = {
#     "uniform": ["low", "high"],
#     "normal": ["mean", "std"],  # or optionally "low", "high"             mode==base
#     "triangular": ["low", "high", "mode"],
#     "discrete_uniform": ["low", "mode", "high"], # define probab
#     "beta_pert": ["low", "high", "mode"],
# }


# {
#   "base_peak": 1500,
#   "base_netsales": [1000, 1050, 1100, 1200, 1300, 1250, 1400, 1350, 1450, 1500],
#   "summary": {
#     "p5": 1020.0,
#     "p10": 1111.0,
#     "p25": 1200.0,
#     "p50": 1300.0,
#     "p75": 1400.0,
#     "p90": 1450.0,
#     "p95": 1480.0
#   }
# }


# {
# "n_simulations": 1000,
# "final_net_sales": 10000,
# "event_params": [
# {"dist_type": "normal", "low": 0.04, "high": 0.04, "base": 0.04},
# {"dist_type": "uniform", "low": 0.04, "high": 0.04, "base": 0.04},
# {"dist_type": "uniform", "low": 0.04, "high": 0.04, "base": 0.04}
# ],
# "class_share_param":{"dist_type":"uniform","low":0.2,"high":0.2,"base":0.2},
# "product_share_param":{"dist_type":"uniform","low":0.5,"high":0.5,"base":0.5},
# "sku_splits":[0.3,0.7],
# "gross_price_param":[
# {"dist_type":"uniform","low":200,"high":200,"base":200},
# {"dist_type":"uniform","low":250,"high":250,"base":250}
# ],
# "gtn_param":[
# {"dist_type":"uniform","low":0.2,"high":0.2,"base":0.2},
# {"dist_type":"uniform","low":0.2,"high":0.2,"base":0.2}
# ]
# }
