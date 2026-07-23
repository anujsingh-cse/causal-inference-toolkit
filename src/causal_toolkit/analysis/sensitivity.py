"""
Sensitivity Analysis Module

Implements multiple sensitivity analysis methods for causal inference:
- Rosenbaum bounds (binary treatment)
- Cinelli-Hazlett sensitivity analysis (continuous treatment)
- E-value computation
- Robustness Value (RV) and R² decompositions
- TIPS (Treatment Effect Sensitivity) curves
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import warnings

from causal_toolkit.core.base import Assumptions, CausalEstimate, EstimatorType


@dataclass
class SensitivityResult:
    """Result of sensitivity analysis."""

    method: str
    gamma: Optional[float] = None  # Rosenbaum sensitivity parameter
    robustness_value: Optional[float] = None  # RV: min confounding strength to change conclusion
    r2_yz_dx: Optional[float] = None  # R²_{Y←Z|X,D} - outcome confounding
    r2_zd_x: Optional[float] = None  # R²_{Z←D|X} - treatment confounding
    e_value: Optional[float] = None  # E-value for point estimate
    e_value_ci: Optional[float] = None  # E-value for CI limit
    benchmark_covariate: Optional[str] = None
    benchmark_r2_yz: Optional[float] = None
    benchmark_r2_zd: Optional[float] = None
    conclusion_reversed: bool = False
    details: Dict[str, Any] = None

    def __post_init__(self):
        if self.details is None:
            self.details = {}

    def __str__(self) -> str:
        if self.method == "rosenbaum":
            return f"Rosenbaum bounds: Γ={self.gamma:.3f}, conclusion_reversed={self.conclusion_reversed}"
        elif self.method == "cinelli_hazlett":
            return (f"Cinelli-Hazlett: RV={self.robustness_value:.4f}, "
                    f"R²_yz={self.r2_yz_dx:.4f}, R²_zd={self.r2_zd_x:.4f}")
        elif self.method == "evalue":
            return f"E-value: {self.e_value:.3f} (CI: {self.e_value_ci:.3f})"
        return f"{self.method}: conclusion_reversed={self.conclusion_reversed}"


class SensitivityAnalyzer:
    """
    Unified sensitivity analysis interface.

    Supports:
    - Rosenbaum bounds for binary treatment with matched/stratified data
    - Cinelli-Hazlett for continuous treatment/binary outcome
    - E-value for point estimates and confidence intervals
    - Benchmark confounding against observed covariates
    """

    def __init__(self, causal_model: Any = None):
        self.model = causal_model
        self.results: List[SensitivityResult] = []

    def rosenbaum_bounds(
        self,
        estimate: CausalEstimate,
        gamma_range: Tuple[float, float] = (1.0, 3.0),
        n_points: int = 50,
        alpha: float = 0.05,
    ) -> SensitivityResult:
        """
        Rosenbaum sensitivity analysis for binary treatment.

        Computes bounds on p-value as function of unobserved confounding
        strength Γ (Gamma). Γ=1 means no hidden bias.

        Args:
            estimate: CausalEstimate with point estimate, CI, p-value
            gamma_range: Range of Γ to evaluate
            n_points: Number of Γ values to test
            alpha: Significance level

        Returns:
            SensitivityResult with critical Γ where conclusion reverses
        """
        gammas = np.linspace(gamma_range[0], gamma_range[1], n_points)

        # Get test statistic and standard error
        if estimate.standard_error is None:
            if estimate.ci_lower is not None and estimate.ci_upper is not None:
                se = (estimate.ci_upper - estimate.ci_lower) / (2 * 1.96)
            else:
                se = 0.1  # fallback
        else:
            se = estimate.standard_error

        point_estimate = estimate.value
        z_obs = abs(point_estimate / se) if se > 0 else 0

        # Rosenbaum bounds: p-value upper bound as function of Γ
        # For large sample: p_upper(Γ) = 1 - Φ(z_obs / Γ + Φ⁻¹(1/Γ))
        from scipy import stats

        p_values = []
        conclusion_reversed = False
        critical_gamma = gamma_range[1]

        for gamma in gammas:
            # Upper bound on p-value
            z_adj = z_obs / gamma
            # Simplified approximation
            if gamma == 1:
                p_upper = 2 * (1 - stats.norm.cdf(z_obs))
            else:
                p_upper = 2 * (1 - stats.norm.cdf(z_adj * stats.norm.ppf(1 - 1/(2*gamma))))

            p_values.append(p_upper)

            if p_upper > alpha and not conclusion_reversed:
                conclusion_reversed = True
                critical_gamma = gamma

        result = SensitivityResult(
            method="rosenbaum",
            gamma=critical_gamma,
            conclusion_reversed=conclusion_reversed,
            details={
                "gammas": gammas.tolist(),
                "p_values": p_values,
                "alpha": alpha,
                "observed_z": z_obs,
                "point_estimate": point_estimate,
            },
        )
        self.results.append(result)
        return result

    def cinelli_hazlett(
        self,
        estimate: CausalEstimate,
        benchmark_covariate: str = None,
        r2_yz_dx: float = None,
        r2_zd_x: float = None,
        k_yz: int = 1,
        k_zd: int = 1,
        alpha: float = 0.05,
    ) -> SensitivityResult:
        """
        Cinelli & Hazlett (2020) sensitivity analysis for linear models.

        Computes Robustness Value (RV) - minimum confounding strength
        needed to change conclusion.

        Args:
            estimate: CausalEstimate from linear regression
            benchmark_covariate: Name of benchmark covariate in model
            r2_yz_dx: Partial R² of outcome ~ confounder | X,D
            r2_zd_x: Partial R² of treatment ~ confounder | X
            k_yz: Number of confounders for outcome
            k_zd: Number of confounders for treatment
            alpha: Significance level

        Returns:
            SensitivityResult with RV and benchmark comparison
        """
        from scipy import stats

        t_stat = estimate.value / estimate.standard_error if estimate.standard_error else 1.96
        df = estimate.n_samples - len(getattr(self.model, 'common_causes', [])) - 2

        if df <= 0:
            df = max(estimate.n_samples - 2, 1)

        # Critical t-value
        t_crit = stats.t.ppf(1 - alpha/2, df)

        # Robustness Value for significance
        # RV = f(t, df) = sqrt(t²/(t²+df)) * (some adjustment)
        # From Cinelli & Hazlett: RV = |t|/sqrt(df + t²) for k=1
        rv_sig = abs(t_stat) / np.sqrt(df + t_stat**2)

        # RV for estimate to reduce to zero
        rv_est = abs(estimate.value) / np.sqrt(df * estimate.standard_error**2 + estimate.value**2 / df)

        robustness_value = min(rv_sig, rv_est)  # Conservative

        # Benchmark against observed covariate
        benchmark_r2_yz = None
        benchmark_r2_zd = None
        if benchmark_covariate and self.model is not None:
            try:
                benchmark_r2_yz, benchmark_r2_zd = self._compute_benchmark_r2(benchmark_covariate)
            except Exception:
                pass

        # Check if benchmark explains away effect
        if benchmark_r2_yz is not None and benchmark_r2_zd is not None:
            r2_needed = (rv_sig**2) / (1 - rv_sig**2)
            conclusion_reversed = (benchmark_r2_yz * benchmark_r2_zd) > r2_needed
        else:
            conclusion_reversed = False

        result = SensitivityResult(
            method="cinelli_hazlett",
            robustness_value=robustness_value,
            r2_yz_dx=r2_yz_dx,
            r2_zd_x=r2_zd_x,
            benchmark_covariate=benchmark_covariate,
            benchmark_r2_yz=benchmark_r2_yz,
            benchmark_r2_zd=benchmark_r2_zd,
            conclusion_reversed=conclusion_reversed,
            details={
                "t_statistic": t_stat,
                "degrees_freedom": df,
                "rv_significance": rv_sig,
                "rv_estimate": rv_est,
                "alpha": alpha,
                "critical_r2": rv_sig**2 / (1 - rv_sig**2) if rv_sig < 1 else np.inf,
            },
        )
        self.results.append(result)
        return result

    def _compute_benchmark_r2(self, covariate: str) -> Tuple[float, float]:
        """Compute partial R² for a benchmark covariate."""
        if self.model is None or not hasattr(self.model, 'data'):
            return (0.0, 0.0)

        from sklearn.linear_model import LinearRegression
        from sklearn.preprocessing import StandardScaler

        data = self.model.data
        X_cols = self.model.common_causes + [self.model.treatment]
        X = data[X_cols].values
        y = data[self.model.outcome].values
        z = data[covariate].values

        # Partial R²: outcome ~ Z | X, D
        reg_full = LinearRegression().fit(X, y)
        reg_reduced = LinearRegression().fit(X[:, :-1], y)  # without treatment
        r2_full = reg_full.score(X, y)
        r2_reduced = reg_reduced.score(X[:, :-1], y)
        r2_yz_dx = (r2_full - r2_reduced) / (1 - r2_reduced) if r2_reduced < 1 else 1.0

        # Partial R²: treatment ~ Z | X
        t = data[self.model.treatment].values
        X_no_t = X[:, :-1]  # without treatment
        reg_t_full = LinearRegression().fit(np.column_stack([X_no_t, z]), t)
        reg_t_reduced = LinearRegression().fit(X_no_t, t)
        r2_t_full = reg_t_full.score(np.column_stack([X_no_t, z]), t)
        r2_t_reduced = reg_t_reduced.score(X_no_t, t)
        r2_zd_x = (r2_t_full - r2_t_reduced) / (1 - r2_t_reduced) if r2_t_reduced < 1 else 1.0

        return (max(0, r2_yz_dx), max(0, r2_zd_x))

    def e_value(
        self,
        estimate: CausalEstimate,
        true_effect: float = 0.0,
        alpha: float = 0.05,
    ) -> SensitivityResult:
        """
        Compute E-value (VanderWeele & Ding, 2017).

        Minimum strength of unmeasured confounder (risk ratio scale)
        needed to explain away the observed association.

        Args:
            estimate: CausalEstimate (assumes risk ratio or can convert)
            true_effect: True causal effect to shift to (default 0 = null)
            alpha: Confidence level for CI E-value

        Returns:
            SensitivityResult with E-value for point estimate and CI
        """
        # E-value for point estimate
        # For risk ratio: E = RR + sqrt(RR*(RR-1))
        # For coefficient: approximate with exp(|coef|)
        from scipy import stats

        # Convert to relative risk scale if needed
        # Assume estimate.value is log-RR or coefficient
        rr_estimate = np.exp(abs(estimate.value))
        e_value = rr_estimate + np.sqrt(rr_estimate * (rr_estimate - 1))

        # E-value for CI limit
        if estimate.ci_lower is not None and estimate.ci_upper is not None:
            # Use the CI limit closer to null
            ci_limit = estimate.ci_lower if estimate.value > 0 else estimate.ci_upper
            rr_ci = np.exp(abs(ci_limit))
            e_value_ci = rr_ci + np.sqrt(rr_ci * (rr_ci - 1))
        else:
            e_value_ci = e_value

        # Conclusion reversed if E-value < some threshold (e.g., 1.25 for mild confounding)
        conclusion_reversed = e_value < 1.25

        result = SensitivityResult(
            method="evalue",
            e_value=e_value,
            e_value_ci=e_value_ci,
            conclusion_reversed=conclusion_reversed,
            details={
                "rr_estimate": rr_estimate,
                "rr_ci": rr_ci if 'rr_ci' in locals() else rr_estimate,
                "true_effect": true_effect,
                "alpha": alpha,
            },
        )
        self.results.append(result)
        return result

    def tip_curve(
        self,
        estimate: CausalEstimate,
        r2_yz_range: Tuple[float, float] = (0, 0.5),
        r2_zd_range: Tuple[float, float] = (0, 0.5),
        n_points: int = 30,
    ) -> Dict[str, Any]:
        """
        Generate TIPS (Treatment Effect Sensitivity) curve data.

        Shows how estimate changes as function of confounding parameters.
        """
        r2_yz_vals = np.linspace(r2_yz_range[0], r2_yz_range[1], n_points)
        r2_zd_vals = np.linspace(r2_zd_range[0], r2_zd_range[1], n_points)

        R2_yz, R2_zd = np.meshgrid(r2_yz_vals, r2_zd_vals)

        # Bias formula from Cinelli & Hazlett
        # Bias = sign(r_yz_zd) * sqrt(R2_yz * R2_zd / ((1-R2_yz)*(1-R2_zd))) * se * sqrt(n)
        # Simplified: adjusted_estimate = estimate - bias
        se = estimate.standard_error or 0.1
        bias_factor = np.sqrt((R2_yz * R2_zd) / ((1 - R2_yz) * (1 - R2_zd) + 1e-10))
        bias = bias_factor * se * np.sqrt(estimate.n_samples)
        adjusted = estimate.value - bias

        # Significance threshold
        from scipy import stats
        t_crit = stats.t.ppf(0.975, max(estimate.n_samples - 5, 1))
        significant = np.abs(adjusted) > t_crit * se

        return {
            "r2_yz": r2_yz_vals.tolist(),
            "r2_zd": r2_zd_vals.tolist(),
            "adjusted_estimates": adjusted.tolist(),
            "significant": significant.tolist(),
            "original_estimate": estimate.value,
        }

    def summarize(self) -> str:
        """Return summary of all sensitivity analyses."""
        if not self.results:
            return "No sensitivity analyses run yet."

        lines = ["Sensitivity Analysis Summary", "=" * 40]
        for r in self.results:
            lines.append(str(r))
            if r.benchmark_covariate:
                lines.append(f"  Benchmark '{r.benchmark_covariate}': "
                           f"R²_yz={r.benchmark_r2_yz:.4f}, R²_zd={r.benchmark_r2_zd:.4f}")
        return "\n".join(lines)


def run_sensitivity_suite(
    estimate: CausalEstimate,
    model: Any = None,
    benchmark_covariates: List[str] = None,
) -> SensitivityAnalyzer:
    """
    Run full sensitivity analysis suite.

    Args:
        estimate: CausalEstimate from primary analysis
        model: CausalModel (optional, for benchmark covariates)
        benchmark_covariates: List of covariate names to benchmark against

    Returns:
        SensitivityAnalyzer with all results
    """
    analyzer = SensitivityAnalyzer(model)

    # 1. Rosenbaum bounds (if binary treatment)
    analyzer.rosenbaum_bounds(estimate)

    # 2. Cinelli-Hazlett (if continuous treatment/linear model)
    analyzer.cinelli_hazlett(estimate)

    # 3. E-value
    analyzer.e_value(estimate)

    # 4. Benchmark against observed covariates
    if benchmark_covariates and model is not None:
        for cov in benchmark_covariates:
            analyzer.cinelli_hazlett(estimate, benchmark_covariate=cov)

    return analyzer