import numpy as np
from typing import List, Dict

def range_forecast_calc(
        base_peak: float,
        base_netsales: List[float],
        summary: Dict[str, float]
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Generate range forecasts based on base peak and summary statistics.

    Args:
        base_peak (float): The base peak value for scaling.
        base_netsales (List[float]): List of base net sales values.
        summary (Dict[str, float]): Summary statistics containing percentiles.

    Returns:
        Dict[str, Dict[str, np.ndarray]]:
            - 'percentiles': scaled percentile curves
            - 'bands': grouped certainty bands (10%, 25%, 50%, 90%)
    """
    base_netsales = np.array(base_netsales, dtype=float)
    percentiles = {}

    # --- Scale percentiles across all months ---
    for p in summary:
        key = p.lower() if not p.startswith("p") else p  # ensure consistent keys
        scaling_factor = summary[p] / base_peak
        percentiles[key] = (base_netsales * scaling_factor).tolist()

    # --- Construct certainty bands ---
    bands = {
        "10%": {"lower": percentiles["p45"],   "upper": percentiles["p55"]},
        "25%": {"lower": percentiles["p37_5"], "upper": percentiles["p62_5"]},
        "50%": {"lower": percentiles["p25"],   "upper": percentiles["p75"]},
        "90%": {"lower": percentiles["p5"],    "upper": percentiles["p95"]},
    }

    return {"bands": bands}