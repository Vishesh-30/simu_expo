import numpy as np
from scipy.stats import beta


class MonteCarloSimulator:
    def __init__(
        self,
        n_simulations,
        final_net_sales,
        event_params,
        class_share_param,
        product_share_param,
        gross_price_param,
        gtn_param,
        sku_splits=None,
    ):
        """
        Monte Carlo Simulator for net sales with optional SKU split.
        """
        self.n_simulations = n_simulations
        self.final_net_sales = float(final_net_sales)
        self.event_params = event_params or []
        self.param_dict = {
            "class_share": class_share_param,
            "product_share": product_share_param,
            "gross_price": gross_price_param,
            "gtn": gtn_param,
        }
        self.sku_splits = sku_splits

        # Validate SKU splits
        if self.sku_splits is not None:
            if len(self.sku_splits) == 0:
                raise ValueError("sku_splits cannot be an empty list.")
            if not np.isclose(sum(self.sku_splits), 1.0):
                raise ValueError("SKU splits must sum to 1.0")
            if any(p < 0 for p in self.sku_splits):
                raise ValueError("SKU splits must be non-negative.")

        # Basic param presence checks
        for key in ("class_share", "product_share", "gross_price", "gtn"):
            if self.param_dict.get(key) is None:
                raise ValueError(f"Missing parameter spec for '{key}'.")

    @staticmethod
    def _validate_low_high(low, high):
        if (low is not None) and (high is not None) and (low > high):
            raise ValueError(f"'low' ({low}) must be <= 'high' ({high}).")

    def sample_from_distribution(
        self, dist_type, low=None, high=None, base=None, mean=None, std=None, probs=None
    ):
        """
        Supported dist_type: 'uniform', 'normal', 'triangular', 'beta_pert', 'discrete_uniform'
        """
        self._validate_low_high(low, high)

        if dist_type == "uniform":
            if low is None or high is None:
                raise ValueError("Uniform requires 'low' and 'high'.")
            return np.random.uniform(low, high)

        elif dist_type == "normal":
            if low is None or high is None:
                raise ValueError("Normal requires 'low' and 'high'.")
            mean = (low + high) / 2
            std = abs(high - low) / 4
            return np.random.normal(mean, std)

        elif dist_type == "triangular":
            if low is None or high is None or base is None:
                raise ValueError("Triangular requires 'low', 'high', and 'base'.")
            if not (low <= base <= high):
                raise ValueError("Triangular requires low <= base <= high.")
            return np.random.triangular(low, base, high)

        elif dist_type == "beta_pert":
            if low is None or high is None or base is None:
                raise ValueError("Beta-PERT requires 'low', 'high', and 'base'.")
            # Mean with λ = 4 (classic PERT weighting)
            mean_pert = (low + 4 * base + high) / 6
            if np.isclose(base, mean_pert) or np.isclose(high, low):
                alpha, beta_val = 2.0, 2.0
            else:
                alpha = 1 + 4 * (base - low) / (high - low)
                beta_val = 1 + 4 * (high - base) / (high - low)
                if alpha <= 0:
                    alpha = 2.0
                if beta_val <= 0:
                    beta_val = 2.0
            sample = beta.rvs(alpha, beta_val)
            return low + sample * (high - low)


        elif dist_type == "discrete_uniform":
            if low is None or high is None or base is None:
                raise ValueError("Discrete uniform requires 'low', 'high', and 'base'.")
            
            # Default to equal probability
            if probs is None:
                probs = [1 / 3, 1 / 3, 1 / 3]

            if len(probs) != 3:
                raise ValueError("Discrete uniform 'probs' must be a list of 3 values.")
            if not np.isclose(sum(probs), 1.0):
                raise ValueError("Discrete uniform 'probs' must sum to 1.0.")

            return np.random.choice([low, base, high], p=probs)

        else:
            raise ValueError(f"Unsupported distribution type: {dist_type}")

    def _sample_param(self, spec):
        return self.sample_from_distribution(**spec)

    def run(self):
        all_results = []

        use_sku = self.sku_splits is not None
        gp_spec = self.param_dict["gross_price"]
        gtn_spec = self.param_dict["gtn"]

        if use_sku:
            n_skus = len(self.sku_splits)
            if isinstance(gp_spec, list) and len(gp_spec) != n_skus:
                raise ValueError("Length of gross_price_param list must match number of sku_splits.")
            if isinstance(gtn_spec, list) and len(gtn_spec) != n_skus:
                raise ValueError("Length of gtn_param list must match number of sku_splits.")

        for _ in range(self.n_simulations):
            # Sample individual event factors
            individual_events = []
            for i, event_spec in enumerate(self.event_params):
                event_value = float(self._sample_param(event_spec))
                individual_events.append(event_value)
            
            # Total event factors (for calculation)
            event_factors = sum(individual_events)

            # Brand-level params
            cs = float(self._sample_param(self.param_dict["class_share"]))
            ps = float(self._sample_param(self.param_dict["product_share"]))

            brand_volume = self.final_net_sales * (1.0 + event_factors) * cs * ps

            sku_sales_values = []
            sku_details = []
            if use_sku:
                for i, split in enumerate(self.sku_splits):
                    sku_volume = brand_volume * split
                    gp = float(self._sample_param(gp_spec[i] if isinstance(gp_spec, list) else gp_spec))
                    gtn = float(self._sample_param(gtn_spec[i] if isinstance(gtn_spec, list) else gtn_spec))
                    net_sales = sku_volume * gp * (1.0 - gtn)
                    sku_sales_values.append(net_sales)

                    sku_details.append({
                        "gross_price": round(gp, 10),
                        "gtn": round(gtn, 10),
                        "net_sales": round(net_sales, 10)
                    })
            else:
                gp = float(self._sample_param(gp_spec))
                gtn = float(self._sample_param(gtn_spec))
                net_sales = brand_volume * gp * (1.0 - gtn)
                sku_sales_values.append(net_sales)

                sku_details.append({
                    "gross_price": round(gp, 10),
                    "gtn": round(gtn, 10),
                    "net_sales": round(net_sales, 10)
                })

            overall_net_sales = float(np.sum(sku_sales_values))

            result_entry = {}
            
            # Add individual events at the top
            for i, event_value in enumerate(individual_events):
                result_entry[f"event_{i+1}"] = round(event_value, 10)
            
            # Add total event factors
            result_entry["total_event_factors"] = round(event_factors, 10)
            
            # Add other parameters
            result_entry.update({
                "class_share": round(cs, 10),
                "product_share": round(ps, 10),
                "overall_net_sales": round(overall_net_sales, 10),
                "sku_split": self.sku_splits if use_sku else None,
                "skus": sku_details
            })

            all_results.append(result_entry)

        # Use overall net sales for histogram + summary
        net_sales_array = np.array([r["overall_net_sales"] for r in all_results], dtype=float)
        summary = {
            "mean": round(float(np.mean(net_sales_array)), 10),
            "median": round(float(np.median(net_sales_array)), 10),
            "std": round(float(np.std(net_sales_array)), 10 ),

            # Percentiles needed for certainty bands
            "p5": round(float(np.percentile(net_sales_array, 5)), 10),
            "p25": round(float(np.percentile(net_sales_array, 25)), 10),
            "p37_5": round(float(np.percentile(net_sales_array, 37.5)), 10),
            "p45": round(float(np.percentile(net_sales_array, 45)), 10),
            "p50": round(float(np.percentile(net_sales_array, 50)), 10),
            "p55": round(float(np.percentile(net_sales_array, 55)), 10),
            "p62_5": round(float(np.percentile(net_sales_array, 62.5)), 10),
            "p75": round(float(np.percentile(net_sales_array, 75)), 10),
            "p95": round(float(np.percentile(net_sales_array, 95)), 10),
            "p10": round(float(np.percentile(net_sales_array, 10)), 10),
            "p90": round(float(np.percentile(net_sales_array, 90)), 10),

            "n_simulations": int(self.n_simulations),
        }

        return {"summary": summary, "all_results": all_results}
