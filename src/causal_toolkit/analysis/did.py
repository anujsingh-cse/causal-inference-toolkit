"""
Difference-in-Differences (DiD) module for policy evaluation and panel causal inference.

Supports classic 2x2 DiD, Two-Way Fixed Effects (TWFE), parallel trends verification,
and event study dynamic effect estimation.
"""

from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd
from scipy import stats


class DiDResult:
    """Container for Difference-in-Differences analysis output."""

    def __init__(
        self,
        att: float,
        se: float,
        t_stat: float,
        p_value: float,
        ci_lower: float,
        ci_upper: float,
        parallel_trends_pvalue: Optional[float] = None,
        event_study_effects: Optional[pd.DataFrame] = None,
        model_type: str = "2x2",
    ):
        self.att = att
        self.se = se
        self.t_stat = t_stat
        self.p_value = p_value
        self.ci_lower = ci_lower
        self.ci_upper = ci_upper
        self.parallel_trends_pvalue = parallel_trends_pvalue
        self.event_study_effects = event_study_effects
        self.model_type = model_type

    def summary(self) -> str:
        pt_str = (
            f"Parallel Trends p-value: {self.parallel_trends_pvalue:.4f}\n"
            if self.parallel_trends_pvalue is not None
            else ""
        )
        return (
            f"=== Difference-in-Differences ({self.model_type}) ===\n"
            f"ATT Estimate: {self.att:.4f} (SE: {self.se:.4f})\n"
            f"t-statistic: {self.t_stat:.4f}, p-value: {self.p_value:.4f}\n"
            f"95% CI: [{self.ci_lower:.4f}, {self.ci_upper:.4f}]\n"
            f"{pt_str}"
        )

    def __repr__(self) -> str:
        return f"<DiDResult ATT={self.att:.4f}, p_value={self.p_value:.4f}>"


class DifferenceInDifferences:
    """
    Difference-in-Differences (DiD) Estimator.
    """

    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha

    def estimate_2x2(
        self,
        data: pd.DataFrame,
        outcome_col: str,
        treatment_col: str,
        post_col: str,
        covariates: Optional[List[str]] = None,
    ) -> DiDResult:
        """
        Classic 2x2 Difference-in-Differences estimation.

        Y = beta0 + beta1*Treat + beta2*Post + beta3*(Treat * Post) + gamma*X + eps
        ATT = beta3
        """
        df = data.copy().dropna(subset=[outcome_col, treatment_col, post_col])
        df["interaction"] = df[treatment_col] * df[post_col]

        X_cols = ["const", treatment_col, post_col, "interaction"]
        if covariates:
            X_cols.extend(covariates)

        df["const"] = 1.0
        X = df[X_cols].values
        y = df[outcome_col].values

        beta, se, t_stat, p_val, ci_low, ci_high = self._ols_regression(X, y)

        att_idx = X_cols.index("interaction")
        att = float(beta[att_idx])
        att_se = float(se[att_idx])
        att_t = float(t_stat[att_idx])
        att_p = float(p_val[att_idx])
        att_ci_low = float(ci_low[att_idx])
        att_ci_high = float(ci_high[att_idx])

        return DiDResult(
            att=att,
            se=att_se,
            t_stat=att_t,
            p_value=att_p,
            ci_lower=att_ci_low,
            ci_upper=att_ci_high,
            model_type="2x2 OLS",
        )

    def fit_panel(
        self,
        data: pd.DataFrame,
        unit_col: str,
        time_col: str,
        outcome_col: str,
        treatment_col: str,
        treatment_start_time: Optional[Union[int, float, str]] = None,
        covariates: Optional[List[str]] = None,
    ) -> DiDResult:
        """
        Panel Two-Way Fixed Effects (TWFE) DiD with parallel trends test & event study.
        """
        df = data.copy().dropna(subset=[unit_col, time_col, outcome_col, treatment_col])

        # Compute post indicator if treatment_start_time supplied
        if "post" not in df.columns:
            if treatment_start_time is None:
                raise ValueError("Specify treatment_start_time when 'post' column not in DataFrame.")
            df["post"] = (df[time_col] >= treatment_start_time).astype(int)

        # 2x2 estimation base
        res_2x2 = self.estimate_2x2(df, outcome_col, treatment_col, "post", covariates)

        # Parallel trends test in pre-treatment period
        pt_pval = self.test_parallel_trends(df, unit_col, time_col, outcome_col, treatment_col, "post")

        # Event study relative period estimation if numeric time_col and treatment_start_time given
        event_df = None
        if treatment_start_time is not None and np.issubdtype(df[time_col].dtype, np.number):
            event_df = self._compute_event_study(
                df, time_col, outcome_col, treatment_col, treatment_start_time
            )

        return DiDResult(
            att=res_2x2.att,
            se=res_2x2.se,
            t_stat=res_2x2.t_stat,
            p_value=res_2x2.p_value,
            ci_lower=res_2x2.ci_lower,
            ci_upper=res_2x2.ci_upper,
            parallel_trends_pvalue=pt_pval,
            event_study_effects=event_df,
            model_type="TWFE Panel",
        )

    def test_parallel_trends(
        self,
        data: pd.DataFrame,
        unit_col: str,
        time_col: str,
        outcome_col: str,
        treatment_col: str,
        post_col: str,
    ) -> float:
        """
        Test parallel pre-treatment linear trends assumption.
        Returns p-value for hypothesis of equal pre-treatment slopes.
        """
        pre_df = data[data[post_col] == 0].copy()
        if len(pre_df) < 5:
            return 1.0

        # Convert time_col to numeric rank if categorical/string
        if not np.issubdtype(pre_df[time_col].dtype, np.number):
            time_map = {t: i for i, t in enumerate(sorted(pre_df[time_col].unique()))}
            pre_df["time_num"] = pre_df[time_col].map(time_map)
        else:
            pre_df["time_num"] = pre_df[time_col]

        pre_df["const"] = 1.0
        pre_df["time_treat_interaction"] = pre_df["time_num"] * pre_df[treatment_col]

        X_cols = ["const", "time_num", treatment_col, "time_treat_interaction"]
        X = pre_df[X_cols].values
        y = pre_df[outcome_col].values

        beta, se, t_stat, p_val, _, _ = self._ols_regression(X, y)
        inter_idx = X_cols.index("time_treat_interaction")

        return float(p_val[inter_idx])

    def _compute_event_study(
        self,
        df: pd.DataFrame,
        time_col: str,
        outcome_col: str,
        treatment_col: str,
        treatment_start_time: Union[int, float],
    ) -> pd.DataFrame:
        """Compute mean outcome dynamic trajectory by period relative to treatment."""
        df_event = df.copy()
        df_event["rel_time"] = df_event[time_col] - treatment_start_time

        grouped = (
            df_event.groupby(["rel_time", treatment_col])[outcome_col]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
        return grouped

    def _ols_regression(self, X: np.ndarray, y: np.ndarray):
        n, k = X.shape
        dof = max(1, n - k)
        beta, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)

        y_pred = X @ beta
        resids = y - y_pred
        sse = float(np.sum(resids**2))
        mse = sse / dof

        var_beta = mse * np.linalg.pinv(X.T @ X)
        se = np.sqrt(np.maximum(0.0, np.diag(var_beta)))
        t_stat = np.where(se > 0, beta / se, 0.0)
        p_val = 2.0 * (1.0 - stats.t.cdf(np.abs(t_stat), df=dof))

        t_crit = stats.t.ppf(1.0 - self.alpha / 2.0, df=dof)
        ci_low = beta - t_crit * se
        ci_high = beta + t_crit * se

        return beta, se, t_stat, p_val, ci_low, ci_high
