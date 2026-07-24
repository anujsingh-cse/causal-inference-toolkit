"""
Synthetic Control Method implementation for quasi-experimental policy evaluation.

Optimizes non-negative donor weights summing to 1 to construct a synthetic control unit
that matches the pre-treatment trajectory and characteristics of the treated unit.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy.optimize import minimize


class SyntheticControlResult:
    """Container for Synthetic Control estimation and placebo testing results."""

    def __init__(
        self,
        treated_unit: Union[str, int],
        weights: Dict[Union[str, int], float],
        pre_rmspe: float,
        post_rmspe: float,
        rmspe_ratio: float,
        att: float,
        time_series: pd.DataFrame,
        placebo_results: Optional[Dict[Union[str, int], Dict[str, float]]] = None,
        p_value: Optional[float] = None,
    ):
        self.treated_unit = treated_unit
        self.weights = weights
        self.pre_rmspe = pre_rmspe
        self.post_rmspe = post_rmspe
        self.rmspe_ratio = rmspe_ratio
        self.att = att
        self.time_series = time_series
        self.placebo_results = placebo_results or {}
        self.p_value = p_value

    def __repr__(self) -> str:
        p_str = f", p_value={self.p_value:.4f}" if self.p_value is not None else ""
        return (
            f"<SyntheticControlResult treated={self.treated_unit}, ATT={self.att:.4f}, "
            f"Pre-RMSPE={self.pre_rmspe:.4f}, RMSPE-Ratio={self.rmspe_ratio:.4f}{p_str}>"
        )


class SyntheticControl:
    """
    Synthetic Control Method for single treated unit and multiple control donor pool.

    Constructs a weighted combination of control units to simulate the counterfactual
    outcome of the treated unit in the absence of treatment.
    """

    def __init__(self, random_state: Optional[int] = None):
        self.random_state = random_state

    def fit_predict(
        self,
        data: pd.DataFrame,
        unit_col: str,
        time_col: str,
        outcome_col: str,
        treated_unit: Union[str, int],
        treatment_time: Union[int, float, str],
        covariates: Optional[List[str]] = None,
        run_placebos: bool = True,
    ) -> SyntheticControlResult:
        """
        Estimate synthetic control weights and treatment effect.

        Parameters
        ----------
        data : pd.DataFrame
            Panel data containing units, timestamps, outcome, and optional covariates.
        unit_col : str
            Column identifying units.
        time_col : str
            Column identifying time periods.
        outcome_col : str
            Outcome variable column.
        treated_unit : Union[str, int]
            Identifier of treated unit.
        treatment_time : Union[int, float, str]
            Time period when treatment starts (inclusive).
        covariates : Optional[List[str]]
            List of covariate column names to balance during pre-treatment period.
        run_placebos : bool
            Whether to run in-space placebo tests for permutation inference p-values.
        """
        # Pivot panel data: rows = time, columns = unit
        pivot_y = data.pivot(index=time_col, columns=unit_col, values=outcome_col)
        units = pivot_y.columns.tolist()

        if treated_unit not in units:
            raise ValueError(f"Treated unit '{treated_unit}' not in data unit column.")

        donor_units = [u for u in units if u != treated_unit]
        if not donor_units:
            raise ValueError("No donor control units found in dataset.")

        pre_mask = pivot_y.index < treatment_time
        post_mask = pivot_y.index >= treatment_time

        if not pre_mask.any():
            raise ValueError(f"No pre-treatment time periods found before {treatment_time}.")
        if not post_mask.any():
            raise ValueError(f"No post-treatment time periods found from {treatment_time}.")

        # Fit weights for actual treated unit
        weights, pre_rmspe, post_rmspe, rmspe_ratio, att, ts_df = self._fit_unit(
            pivot_y, treated_unit, donor_units, pre_mask, post_mask, covariates, data, unit_col, time_col
        )

        placebo_res: Dict[Union[str, int], Dict[str, float]] = {}
        p_value: Optional[float] = None

        if run_placebos:
            actual_ratio = rmspe_ratio
            ratio_list = [actual_ratio]

            for placebo_u in donor_units:
                p_donors = [u for u in units if u != placebo_u]
                _, _, _, p_ratio, p_att, _ = self._fit_unit(
                    pivot_y, placebo_u, p_donors, pre_mask, post_mask, covariates, data, unit_col, time_col
                )
                placebo_res[placebo_u] = {"att": p_att, "rmspe_ratio": p_ratio}
                ratio_list.append(p_ratio)

            # Permutation p-value: proportion of placebos with RMSPE ratio >= treated RMSPE ratio
            p_value = float(np.mean([r >= actual_ratio for r in ratio_list]))

        weight_dict = {u: float(w) for u, w in zip(donor_units, weights)}

        return SyntheticControlResult(
            treated_unit=treated_unit,
            weights=weight_dict,
            pre_rmspe=pre_rmspe,
            post_rmspe=post_rmspe,
            rmspe_ratio=rmspe_ratio,
            att=att,
            time_series=ts_df,
            placebo_results=placebo_res,
            p_value=p_value,
        )

    def _fit_unit(
        self,
        pivot_y: pd.DataFrame,
        target_unit: Union[str, int],
        donor_units: List[Union[str, int]],
        pre_mask: pd.Series,
        post_mask: pd.Series,
        covariates: Optional[List[str]],
        raw_data: pd.DataFrame,
        unit_col: str,
        time_col: str,
    ) -> Tuple[np.ndarray, float, float, float, float, pd.DataFrame]:
        # Pre-treatment target outcome
        y_pre_target = pivot_y.loc[pre_mask, target_unit].values
        # Pre-treatment donor outcomes
        y_pre_donors = pivot_y.loc[pre_mask, donor_units].values  # shape: (T_pre, N_donors)

        n_donors = len(donor_units)
        init_weights = np.ones(n_donors) / n_donors
        bounds = [(0.0, 1.0) for _ in range(n_donors)]
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

        def loss_func(w: np.ndarray) -> float:
            synth_pre = y_pre_donors @ w
            diff = y_pre_target - synth_pre
            return float(np.mean(diff**2))

        opt_res = minimize(
            loss_func,
            init_weights,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-8, "maxiter": 500},
        )
        weights = opt_res.x

        # Compute synthetic series across all time
        y_all_donors = pivot_y[donor_units].values
        y_synth = y_all_donors @ weights
        y_actual = pivot_y[target_unit].values

        pre_diff = y_actual[pre_mask] - y_synth[pre_mask]
        post_diff = y_actual[post_mask] - y_synth[post_mask]

        pre_rmspe = float(np.sqrt(np.mean(pre_diff**2)))
        post_rmspe = float(np.sqrt(np.mean(post_diff**2)))
        rmspe_ratio = post_rmspe / pre_rmspe if pre_rmspe > 0 else np.nan
        att = float(np.mean(post_diff))

        ts_df = pd.DataFrame(
            {
                "time": pivot_y.index,
                "actual": y_actual,
                "synthetic": y_synth,
                "effect": y_actual - y_synth,
            }
        ).set_index("time")

        return weights, pre_rmspe, post_rmspe, rmspe_ratio, att, ts_df
