import numpy as np
from typing import List, Dict

def range_forecast(
        base_peak: float,
        base_netsales: List[float],
        summary: Dict[str, float]
) -> Dict[str, np.ndarray]:
    """
    Generate range forecasts based on base peak and summary statistics.

    Args:
        base_peak (float): The base peak value for scaling.
        base_netsales (List[float]): List of base net sales values.
        summary (Dict[str, float]): Summary statistics containing percentiles, etc.

    Returns:
        Dict[str, np.ndarray]: A dictionary with range forecasts for each summary statistic.
    """
    base_netsales = np.array(base_netsales)
    range_forecasts = {}

    for p in summary:
        scaling_factor = summary[p] / base_peak
        range_forecasts[p] = (base_netsales * scaling_factor).tolist()

    return range_forecasts
