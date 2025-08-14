from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional, Union, Dict
from monte_carlo import MonteCarloSimulator
from range_forecast import range_forecast as rf_func

app = FastAPI()

class DistributionParam(BaseModel):
    dist_type: str
    low: Optional[float] = None
    high: Optional[float] = None
    mean: Optional[float] = None
    std: Optional[float] = None
    base: Optional[float] = None

class SimulationRequest(BaseModel):
    n_simulations: int
    final_baseline_trend: float
    event_params: List[DistributionParam]
    class_share_param: Union[DistributionParam, List[DistributionParam]]
    product_share_param: Union[DistributionParam, List[DistributionParam]]
    gross_price_param: Union[DistributionParam, List[DistributionParam]]
    gtn_param: Union[DistributionParam, List[DistributionParam]]

@app.post("/simulate")
async def simulate(request: SimulationRequest):
    def convert_param(param):
        if isinstance(param, list):
            return [p.dict() for p in param]
        return param.dict()
    
    sim = MonteCarloSimulator(
        n_simulations=request.n_simulations,
        final_baseline_trend=request.final_baseline_trend,
        event_params=[param.dict() for param in request.event_params],
        class_share_param=convert_param(request.class_share_param),
        product_share_param=convert_param(request.product_share_param),
        gross_price_param=convert_param(request.gross_price_param),
        gtn_param=convert_param(request.gtn_param)
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
    range_forecasts = rf_func(request.base_peak, request.base_netsales, request.summary)
    return range_forecasts





# Input format for the API
# {
#   "n_simulations": 10000,
#   "final_baseline_trend": 10146472.64,        
#   "event_params": [
#         {"dist_type": "uniform", "low": 0.10, "high": 0.15, "base": 0.12},                                  #N number of events
#         {"dist_type": "normal", "low": 0.04, "high": 0.05, "base": 0.045},
#         {"dist_type": "uniform", "low": -0.16, "high": -0.10, "base": -0.13},
#         {"dist_type": "uniform", "low": 0.03, "high": 0.05, "base": 0.04}
#     ],
#   "class_share_param": [
#         {"dist_type": "uniform", "low": 0.2, "high": 0.3, "base": 0.25},                                    # SKU split(only one row in case of no SKU Split)
#         {"dist_type": "uniform", "low": 0.3, "high": 0.4, "base": 0.35}
#     ],
#   "product_share_param": [
#         {"dist_type": "uniform", "low": 0.1, "high": 0.2, "base": 0.15},
#         {"dist_type": "uniform", "low": 0.14, "high": 0.2, "base": 0.17}
#     ],
#   "gross_price_param": [
#         {"dist_type": "uniform", "low": 240.63, "high": 245.63, "base": 243.13},
#         {"dist_type": "uniform", "low": 260, "high": 280, "base": 270}
#     ],
#   "gtn_param": [
#         {"dist_type": "uniform", "low": 0.05, "high": 0.06, "base": 0.055},
#         {"dist_type": "uniform", "low": 0.05, "high": 0.08, "base": 0.065}
#     ]
# }



# Output format:
# {
#     "summary": {
#         "mean": 249330220.03,
#         "median": 253284240.73,
#         "std": 10108038.28,
#         "p5": 234853058.27,
#         "p10": 236916556.28,
#         "p25": 243107050.3,
#         "p50": 253284240.73,
#         "p75": 257002757.7,
#         "p90": 259081597.78,
#         "p95": 259774544.47,
#         "n_simulations": 10000,
#     },
#     "all_results": [
#         232789560.27,             We can plot histogram using this data
#         257002757.7,
#         260467491.17,
#         243107050.3,
#         253284240.73
#         ...
#
#     ]
# }


# distribution_params = {
#     "uniform": ["low", "high"],
#     "normal": ["mean", "std"],  # or optionally "low", "high"             mode==base
#     "triangular": ["low", "high", "mode"],
#     "discrete_uniform": ["low", "mode", "high"], # define probab
#     "beta_pert": ["low", "high", "mode"],
# }