import numpy as np
from scipy.stats import beta
import json

class MonteCarloSimulator:
    def __init__(self, n_simulations, primary_market_configs, secondary_market_configs=None):
        self.n_simulations = n_simulations
        self.primary_configs = primary_market_configs
        self.secondary_configs = secondary_market_configs or []

        if not isinstance(self.primary_configs, list) or not self.primary_configs:
            raise ValueError("'primary_market_configs' must be a non-empty list.")
        for i, config in enumerate(self.primary_configs):
            if "region" not in config or "market_name" not in config:
                raise ValueError(f"Missing 'region' or 'market_name' key in primary config at index {i}.")
        for i, config in enumerate(self.secondary_configs):
            required = ["market_name", "scale_up_region", "scale_up_param"]
            if not all(k in config for k in required):
                raise ValueError(f"Missing one of {required} in secondary config at index {i}.")

    # Methods _validate_low_high and sample_from_distribution are unchanged
    @staticmethod
    def _validate_low_high(low, high):
        if (low is not None) and (high is not None) and (low > high):
            raise ValueError(f"'low' ({low}) must be <= 'high' ({high}).")

    def sample_from_distribution(
        self, dist_type, low=None, high=None, base=None, mean=None, std=None, probs=None
    ):
        self._validate_low_high(low, high)
        if dist_type == "uniform":
            if low is None or high is None: raise ValueError("Uniform requires 'low' and 'high'.")
            return np.random.uniform(low, high)
        elif dist_type == "normal":
            if low is None or high is None: raise ValueError("Normal requires 'low' and 'high'.")
            mean_val, std_val = (low + high) / 2, abs(high - low) / 4
            return np.random.normal(mean_val, std_val)
        elif dist_type == "triangular":
            if low is None or high is None or base is None: raise ValueError("Triangular requires 'low', 'high', and 'base'.")
            if not (low <= base <= high): raise ValueError("Triangular requires low <= base <= high.")
            return np.random.triangular(low, base, high)
        elif dist_type == "beta_pert":
            if low is None or high is None or base is None: raise ValueError("Beta-PERT requires 'low', 'high', and 'base'.")
            mean_pert = (low + 4 * base + high) / 6
            if np.isclose(base, mean_pert) or np.isclose(high, low): alpha, beta_val = 2.0, 2.0
            else:
                alpha = 1 + 4 * (base - low) / (high - low)
                beta_val = 1 + 4 * (high - base) / (high - low)
                if alpha <= 0: alpha = 2.0
                if beta_val <= 0: beta_val = 2.0
            return low + beta.rvs(alpha, beta_val) * (high - low)
        elif dist_type == "discrete_uniform":
            if low is None or high is None or base is None: raise ValueError("Discrete uniform requires 'low', 'high', and 'base'.")
            probs = probs if probs is not None else [1/3, 1/3, 1/3]
            if len(probs) != 3: raise ValueError("Discrete uniform 'probs' must be a list of 3 values.")
            if not np.isclose(sum(probs), 1.0): raise ValueError("Discrete uniform 'probs' must sum to 1.0.")
            return np.random.choice([low, base, high], p=probs)
        else:
            raise ValueError(f"Unsupported distribution type: {dist_type}")

    def _sample_param(self, spec):
        if isinstance(spec, dict) and 'dist_type' in spec:
            return self.sample_from_distribution(**spec)
        return spec

    def _run_single_primary_market_simulation(self, market_config):
        baseline_volume = float(market_config["baseline_volume"])
        event_params = market_config.get("event_params", [])
        cs = float(self._sample_param(market_config["class_share_param"]))
        ps = float(self._sample_param(market_config["product_share_param"]))
        gp_spec = market_config["gross_price_param"]
        gtn_spec = market_config["gtn_param"]
        sku_splits = market_config.get("sku_splits")

        # --- MODIFIED EVENT SAMPLING ---
        # Capture detailed event info instead of just the consolidated value
        sampled_events_details = []
        for spec in event_params:
            sampled_value = float(self._sample_param(spec))
            sampled_events_details.append({
                "input_params": spec,
                "sampled_value": sampled_value
            })
        
        event_factors = sum(e["sampled_value"] for e in sampled_events_details)
        total_volume = baseline_volume * (1.0 + event_factors) * cs * ps

        sku_sales_values = []
        sku_details = []
        if sku_splits:
            for i, split in enumerate(sku_splits):
                sku_volume = total_volume * split
                gp = float(self._sample_param(gp_spec[i] if isinstance(gp_spec, list) else gp_spec))
                gtn = float(self._sample_param(gtn_spec[i] if isinstance(gtn_spec, list) else gtn_spec))
                net_sales = sku_volume * gp * (1.0 - gtn)
                sku_sales_values.append(net_sales)
                sku_details.append({"gross_price": gp, "gtn": gtn, "net_sales": net_sales})
        else:
            gp = float(self._sample_param(gp_spec))
            gtn = float(self._sample_param(gtn_spec))
            net_sales = total_volume * gp * (1.0 - gtn)
            sku_sales_values.append(net_sales)
            sku_details.append({"gross_price": gp, "gtn": gtn, "net_sales": net_sales})
        
        result = {
            "overall_net_sales": float(np.sum(sku_sales_values)),
            "sku_splits": sku_splits,
            "skus": sku_details,
            "sampled_values": {
                "events": sampled_events_details, # Pass the new detailed list
                "class_share": cs,
                "product_share": ps,
                "calculated_total_volume": total_volume
            }
        }
        return result

    def run(self):
        # This method is unchanged
        all_results = []
        for _ in range(self.n_simulations):
            primary_market_results = {
                cfg["market_name"]: self._run_single_primary_market_simulation(cfg)
                for cfg in self.primary_configs
            }
            secondary_market_results = {}
            if self.secondary_configs:
                regional_total_sales = {}
                for cfg in self.primary_configs:
                    region = cfg["region"]
                    if region not in regional_total_sales:
                        regional_total_sales[region] = 0.0
                    regional_total_sales[region] += primary_market_results[cfg["market_name"]]["overall_net_sales"]
                for cfg in self.secondary_configs:
                    name = cfg["market_name"]
                    region = cfg["scale_up_region"]
                    regional_base_sales = regional_total_sales.get(region, 0.0)
                    sampled_factor = self._sample_param(cfg["scale_up_param"])
                    scaled_up_sales = regional_base_sales * sampled_factor
                    secondary_market_results[name] = {
                        "overall_net_sales": scaled_up_sales,
                        "sampled_scale_up_factor": sampled_factor
                    }
            world_total_sales = sum(res["overall_net_sales"] for res in primary_market_results.values()) + \
                                sum(res["overall_net_sales"] for res in secondary_market_results.values())
            all_results.append({
                "world_net_sales": world_total_sales,
                "primary_markets": primary_market_results,
                "secondary_markets": secondary_market_results
            })

        net_sales_array = np.array([r["world_net_sales"] for r in all_results], dtype=float)
        summary = {
            "mean": round(float(np.mean(net_sales_array)), 10),
            "median": round(float(np.median(net_sales_array)), 10),
            "std": round(float(np.std(net_sales_array)), 10),
            "p5": round(float(np.percentile(net_sales_array, 5)), 10),
            "p10": round(float(np.percentile(net_sales_array, 10)), 10),
            "p25": round(float(np.percentile(net_sales_array, 25)), 10),
            "p37_5": round(float(np.percentile(net_sales_array, 37.5)), 10),
            "p45": round(float(np.percentile(net_sales_array, 45)), 10),
            "p50": round(float(np.percentile(net_sales_array, 50)), 10),
            "p55": round(float(np.percentile(net_sales_array, 55)), 10),
            "p62_5": round(float(np.percentile(net_sales_array, 62.5)), 10),
            "p75": round(float(np.percentile(net_sales_array, 75)), 10),
            "p90": round(float(np.percentile(net_sales_array, 90)), 10),
            "p95": round(float(np.percentile(net_sales_array, 95)), 10),
            "n_simulations": int(self.n_simulations),
        }
        return {"summary": summary, "all_results": all_results}