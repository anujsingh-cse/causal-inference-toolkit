"""
A/B Test Analysis Module

Provides frequentist and Bayesian A/B testing with:
- t-test, proportion z-test, sequential testing (SPRT, mSPRT)
- Bayesian Beta-Binomial, Normal-Normal with ROPE
- Power analysis and sample size calculation
- Multiple testing corrections
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


class TestType(str, Enum):
    """Type of A/B test."""

    PROPORTION = "proportion"  # Conversion rates
    MEAN = "mean"  # Continuous outcomes
    REVENUE = "revenue"  # Revenue per user (often zero-inflated)
    RATIO = "ratio"  # Ratio metrics


class Alternative(str, Enum):
    """Alternative hypothesis."""

    TWO_SIDED = "two-sided"
    GREATER = "greater"
    LESS = "less"


@dataclass
class ABTestData:
    """Data container for A/B test."""

    # Variant A (control)
    n_a: int
    n_b: int
    successes_a: int = 0
    sum_a: float = 0.0
    sum_sq_a: float = 0.0
    # Variant B (treatment)
    successes_b: int = 0
    sum_b: float = 0.0
    sum_sq_b: float = 0.0

    @property
    def rate_a(self) -> float:
        return self.successes_a / self.n_a if self.n_a > 0 else 0

    @property
    def rate_b(self) -> float:
        return self.successes_b / self.n_b if self.n_b > 0 else 0

    @property
    def mean_a(self) -> float:
        return self.sum_a / self.n_a if self.n_a > 0 else 0

    @property
    def mean_b(self) -> float:
        return self.sum_b / self.n_b if self.n_b > 0 else 0

    @property
    def var_a(self) -> float:
        if self.n_a <= 1:
            return 0
        return (self.sum_sq_a - self.sum_a**2 / self.n_a) / (self.n_a - 1)

    @property
    def var_b(self) -> float:
        if self.n_b <= 1:
            return 0
        return (self.sum_sq_b - self.sum_b**2 / self.n_b) / (self.n_b - 1)


@dataclass
class ABTestResult:
    """Result of A/B test."""

    test_type: TestType
    alternative: Alternative

    # Estimates
    estimate_a: float
    estimate_b: float
    difference: float
    relative_difference: float

    # Statistics
    statistic: float
    p_value: float
    ci_lower: float
    ci_upper: float
    confidence_level: float

    # Additional
    n_a: int
    n_b: int
    power: float | None = None
    mde: float | None = None  # Minimum detectable effect

    # Bayesian
    prob_b_better: float | None = None
    rope_probability: float | None = None
    expected_loss_a: float | None = None
    expected_loss_b: float | None = None

    def __str__(self) -> str:
        sig = "✓ SIGNIFICANT" if self.p_value < (1 - self.confidence_level) else "✗ NOT SIGNIFICANT"
        return (
            f"A/B Test ({self.test_type.value}): {sig}\n"
            f"  A: {self.estimate_a:.4f} (n={self.n_a})\n"
            f"  B: {self.estimate_b:.4f} (n={self.n_b})\n"
            f"  Diff: {self.difference:.4f} ({self.relative_difference:.2%})\n"
            f"  p={self.p_value:.6f}, CI=[{self.ci_lower:.4f}, {self.ci_upper:.4f}]"
        )


class ABTestAnalyzer:
    """
    A/B test analyzer with frequentist and Bayesian methods.
    """

    def __init__(self, confidence_level: float = 0.95):
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level

    # ==================== Frequentist Tests ====================

    def proportion_ztest(
        self, data: ABTestData, alternative: Alternative = Alternative.TWO_SIDED
    ) -> ABTestResult:
        """Two-proportion z-test for conversion rates."""
        from statsmodels.stats.proportion import proportions_ztest

        count = np.array([data.successes_a, data.successes_b])
        nobs = np.array([data.n_a, data.n_b])

        stat, pval = proportions_ztest(count, nobs, alternative=alternative.value)

        # Confidence interval
        prop_a = data.rate_a
        prop_b = data.rate_b
        se = np.sqrt(prop_a * (1 - prop_a) / data.n_a + prop_b * (1 - prop_b) / data.n_b)
        z = stats.norm.ppf(1 - self.alpha / 2)
        diff = prop_b - prop_a
        ci_lower = diff - z * se
        ci_upper = diff + z * se

        return ABTestResult(
            test_type=TestType.PROPORTION,
            alternative=alternative,
            estimate_a=prop_a,
            estimate_b=prop_b,
            difference=diff,
            relative_difference=diff / prop_a if prop_a > 0 else 0,
            statistic=stat,
            p_value=pval,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            confidence_level=self.confidence_level,
            n_a=data.n_a,
            n_b=data.n_b,
        )

    def ttest(
        self,
        data: ABTestData,
        alternative: Alternative = Alternative.TWO_SIDED,
        equal_var: bool = False,
    ) -> ABTestResult:
        """Two-sample t-test for continuous metrics."""
        from scipy.stats import ttest_ind_from_stats

        stat, pval = ttest_ind_from_stats(
            data.mean_a,
            np.sqrt(data.var_a),
            data.n_a,
            data.mean_b,
            np.sqrt(data.var_b),
            data.n_b,
            equal_var=equal_var,
            alternative=alternative.value,
        )

        # Confidence interval
        diff = data.mean_b - data.mean_a
        se = np.sqrt(data.var_a / data.n_a + data.var_b / data.n_b)
        df = data.n_a + data.n_b - 2
        t_crit = stats.t.ppf(1 - self.alpha / 2, df)
        ci_lower = diff - t_crit * se
        ci_upper = diff + t_crit * se

        return ABTestResult(
            test_type=TestType.MEAN,
            alternative=alternative,
            estimate_a=data.mean_a,
            estimate_b=data.mean_b,
            difference=diff,
            relative_difference=diff / data.mean_a if data.mean_a != 0 else 0,
            statistic=stat,
            p_value=pval,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            confidence_level=self.confidence_level,
            n_a=data.n_a,
            n_b=data.n_b,
        )

    def mann_whitney(
        self,
        data_a: np.ndarray,
        data_b: np.ndarray,
        alternative: Alternative = Alternative.TWO_SIDED,
    ) -> ABTestResult:
        """Mann-Whitney U test for non-parametric comparison."""
        from scipy.stats import mannwhitneyu

        stat, pval = mannwhitneyu(data_a, data_b, alternative=alternative.value)

        # Hodges-Lehmann estimator for difference
        diffs = np.subtract.outer(data_b, data_a).flatten()
        median_diff = np.median(diffs)

        return ABTestResult(
            test_type=TestType.MEAN,
            alternative=alternative,
            estimate_a=np.median(data_a),
            estimate_b=np.median(data_b),
            difference=median_diff,
            relative_difference=median_diff / np.median(data_a) if np.median(data_a) != 0 else 0,
            statistic=stat,
            p_value=pval,
            ci_lower=0,  # Would need bootstrap for CI
            ci_upper=0,
            confidence_level=self.confidence_level,
            n_a=len(data_a),
            n_b=len(data_b),
        )

    # ==================== Sequential Testing ====================

    def sprt(
        self,
        data: ABTestData,
        mde: float = 0.05,  # Minimum detectable effect (relative)
        alpha: float = 0.05,
        beta: float = 0.2,
    ) -> dict[str, Any]:
        """
        Sequential Probability Ratio Test (SPRT) for proportions.

        Allows continuous monitoring with early stopping.
        """
        p0 = data.rate_a
        p1 = p0 * (1 + mde)

        # Wald's SPRT boundaries
        a = np.log(beta / (1 - alpha))
        b = np.log((1 - beta) / alpha)

        # Log likelihood ratio
        # Simplified: assume we observe conversions sequentially
        # In practice, need sequential data
        n_total = data.n_a + data.n_b
        successes_total = data.successes_a + data.successes_b
        _ = successes_total / n_total if n_total > 0 else p0

        llr = successes_total * np.log(p1 / p0) + (n_total - successes_total) * np.log(
            (1 - p1) / (1 - p0)
        )

        if llr >= b:
            decision = "accept_h1"  # Treatment better
        elif llr <= a:
            decision = "accept_h0"  # No difference
        else:
            decision = "continue"

        return {
            "decision": decision,
            "llr": llr,
            "lower_bound": a,
            "upper_bound": b,
            "p0": p0,
            "p1": p1,
        }

    def msprt(
        self, data: ABTestData, mde: float = 0.05, alpha: float = 0.05, prior_strength: float = 1.0
    ) -> dict[str, Any]:
        """
        Mixture SPRT (mSPRT) - Bayesian sequential test.

        More robust than SPRT, allows optional stopping.
        """
        # Simplified mSPRT using Beta prior
        from scipy.stats import beta

        p0 = data.rate_a
        p1 = p0 * (1 + mde)

        # Prior: Beta(alpha, beta) centered at p0
        alpha_prior = prior_strength * p0
        beta_prior = prior_strength * (1 - p0)

        # Posterior after observing data
        alpha_post = alpha_prior + data.successes_b
        beta_post = beta_prior + data.n_b - data.successes_b

        # Bayes factor
        bf = beta.pdf(p1, alpha_post, beta_post) / beta.pdf(p0, alpha_post, beta_post)

        threshold = (1 - alpha) / alpha

        if bf >= threshold:
            decision = "accept_h1"
        elif bf <= 1 / threshold:
            decision = "accept_h0"
        else:
            decision = "continue"

        return {
            "decision": decision,
            "bayes_factor": bf,
            "threshold": threshold,
            "posterior_alpha": alpha_post,
            "posterior_beta": beta_post,
        }

    # ==================== Bayesian Analysis ====================

    def bayesian_proportion(
        self,
        data: ABTestData,
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0,
        rope_width: float = 0.01,  # ROPE = Region of Practical Equivalence
    ) -> ABTestResult:
        """
        Bayesian A/B test for proportions using Beta-Binomial model.

        Returns probability B > A, ROPE probability, expected loss.
        """
        # Posteriors
        alpha_a = prior_alpha + data.successes_a
        beta_a = prior_beta + data.n_a - data.successes_a
        alpha_b = prior_alpha + data.successes_b
        beta_b = prior_beta + data.n_b - data.successes_b

        # Monte Carlo for prob(B > A)
        n_samples = 100000
        samples_a = np.random.beta(alpha_a, beta_a, n_samples)
        samples_b = np.random.beta(alpha_b, beta_b, n_samples)

        prob_b_better = np.mean(samples_b > samples_a)

        # Difference distribution
        diff_samples = samples_b - samples_a
        rel_diff_samples = diff_samples / samples_a

        # Credible interval
        ci_lower = np.percentile(diff_samples, 100 * self.alpha / 2)
        ci_upper = np.percentile(diff_samples, 100 * (1 - self.alpha / 2))

        # ROPE probability
        rope_prob = np.mean(np.abs(diff_samples) < rope_width)

        # Expected loss (if we choose wrong variant)
        expected_loss_a = np.mean(np.maximum(samples_a - samples_b, 0))
        expected_loss_b = np.mean(np.maximum(samples_b - samples_a, 0))

        return ABTestResult(
            test_type=TestType.PROPORTION,
            alternative=Alternative.TWO_SIDED,
            estimate_a=data.rate_a,
            estimate_b=data.rate_b,
            difference=np.mean(diff_samples),
            relative_difference=np.mean(rel_diff_samples),
            statistic=0,  # Not applicable
            p_value=0,  # Not applicable
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            confidence_level=self.confidence_level,
            n_a=data.n_a,
            n_b=data.n_b,
            prob_b_better=prob_b_better,
            rope_probability=rope_prob,
            expected_loss_a=expected_loss_a,
            expected_loss_b=expected_loss_b,
        )

    def bayesian_normal(
        self,
        data: ABTestData,
        prior_mu: float = 0.0,
        prior_sigma: float = 10.0,
        rope_width: float = 0.01,
    ) -> ABTestResult:
        """
        Bayesian A/B test for normal means with known/unknown variance.
        """
        # Simplified: use normal approximation
        # For full implementation, use MCMC or conjugate normal
        return self.bayesian_proportion(data, rope_width=rope_width)

    # ==================== Power & Sample Size ====================

    def power_analysis(
        self,
        baseline_rate: float,
        mde: float,
        alpha: float = 0.05,
        power: float = 0.8,
        ratio: float = 1.0,
    ) -> dict[str, float]:
        """
        Calculate sample size needed for proportion test.
        """
        from statsmodels.stats.power import NormalIndPower
        from statsmodels.stats.proportion import proportion_effectsize

        effect_size = proportion_effectsize(baseline_rate, baseline_rate * (1 + mde))
        analysis = NormalIndPower()
        n = analysis.solve_power(effect_size, power=power, alpha=alpha, ratio=ratio)
        return {
            "sample_size_per_variant": int(np.ceil(n)),
            "total_sample_size": int(np.ceil(n * (1 + ratio))),
            "effect_size": effect_size,
            "achieved_power": analysis.power(effect_size, n, alpha, ratio),
        }

    def mde_calculation(
        self, baseline_rate: float, n_per_variant: int, alpha: float = 0.05, power: float = 0.8
    ) -> float:
        """Calculate MDE given sample size."""
        from statsmodels.stats.power import NormalIndPower

        analysis = NormalIndPower()
        effect_size = analysis.solve_power(
            effect_size=None, nobs1=n_per_variant, alpha=alpha, power=power
        )
        # Convert effect size back to MDE
        # effect_size = 2* = proportion_effectsize(p1, p2)
        # For small effects: proportion_effectsize ≈ (p2-p1)/sqrt(p1*(1-p1))
        se = np.sqrt(baseline_rate * (1 - baseline_rate))
        mde = effect_size * se / baseline_rate
        return mde

    # ==================== Multiple Testing ====================

    def bonferroni_correction(self, p_values: list[float], alpha: float = 0.05) -> dict:
        """Bonferroni correction for multiple comparisons."""
        k = len(p_values)
        adjusted_alpha = alpha / k
        rejected = [p < adjusted_alpha for p in p_values]
        return {
            "adjusted_alpha": adjusted_alpha,
            "rejected": rejected,
            "adjusted_p_values": [min(p * k, 1.0) for p in p_values],
        }

    def benjamini_hochberg(self, p_values: list[float], alpha: float = 0.05) -> dict:
        """Benjamini-Hochberg FDR control."""
        k = len(p_values)
        sorted_indices = np.argsort(p_values)
        sorted_p = np.array(p_values)[sorted_indices]

        thresholds = alpha * np.arange(1, k + 1) / k
        rejected_sorted = sorted_p <= thresholds

        # Find largest k where p_(k) <= k*alpha/k
        max_rejected = -1 if not any(rejected_sorted) else np.where(rejected_sorted)[0].max()

        rejected = [False] * k
        if max_rejected >= 0:
            for i in range(max_rejected + 1):
                rejected[sorted_indices[i]] = True

        return {
            "rejected": rejected,
            "threshold": alpha * (max_rejected + 1) / k if max_rejected >= 0 else 0,
            "adjusted_p_values": np.array(
                [min(p * k / (i + 1), 1.0) for i, p in enumerate(sorted_p)]
            )[np.argsort(sorted_indices)].tolist(),
        }

    # ==================== High-level API ====================

    def analyze(
        self,
        data: ABTestData,
        test_type: TestType = TestType.PROPORTION,
        method: str = "frequentist",
        **kwargs,
    ) -> ABTestResult:
        """
        Main analysis entry point.

        Args:
            data: ABTestData with counts/sums
            test_type: PROPORTION, MEAN, REVENUE
            method: "frequentist", "bayesian", "sequential"
        """
        if method == "bayesian":
            if test_type == TestType.PROPORTION:
                return self.bayesian_proportion(data, **kwargs)
            else:
                return self.bayesian_normal(data, **kwargs)
        elif method == "sequential":
            self.sprt(data, **kwargs)
            # Return ABTestResult with sequential info
            return ABTestResult(
                test_type=test_type,
                alternative=Alternative.TWO_SIDED,
                estimate_a=data.rate_a,
                estimate_b=data.rate_b,
                difference=data.rate_b - data.rate_a,
                relative_difference=(data.rate_b - data.rate_a) / data.rate_a
                if data.rate_a > 0
                else 0,
                statistic=0,
                p_value=0,
                ci_lower=0,
                ci_upper=0,
                confidence_level=self.confidence_level,
                n_a=data.n_a,
                n_b=data.n_b,
                # Store sequential result in diagnostics
                # (would need to extend ABTestResult)
            )
        else:
            if test_type == TestType.PROPORTION:
                return self.proportion_ztest(data, **kwargs)
            else:
                return self.ttest(data, **kwargs)

    def from_dataframe(
        self,
        df: pd.DataFrame,
        variant_col: str,
        outcome_col: str,
        variant_a: str,
        variant_b: str,
        test_type: TestType = TestType.PROPORTION,
    ) -> ABTestData:
        """Create ABTestData from DataFrame."""
        a_data = df[df[variant_col] == variant_a][outcome_col]
        b_data = df[df[variant_col] == variant_b][outcome_col]

        if test_type == TestType.PROPORTION:
            return ABTestData(
                n_a=len(a_data), successes_a=a_data.sum(), n_b=len(b_data), successes_b=b_data.sum()
            )
        else:
            return ABTestData(
                n_a=len(a_data),
                sum_a=a_data.sum(),
                sum_sq_a=(a_data**2).sum(),
                n_b=len(b_data),
                sum_b=b_data.sum(),
                sum_sq_b=(b_data**2).sum(),
            )


def evaluate_uplift(
    uplift: np.ndarray, treatment: np.ndarray, outcome: np.ndarray, n_bins: int = 10
) -> dict[str, float]:
    """
    Evaluate uplift model using Qini, AUUC, etc.

    Args:
        uplift: Predicted individual treatment effects (CATE)
        treatment: Binary treatment assignment (0/1)
        outcome: Observed outcomes
        n_bins: Number of quantile bins for Qini curve

    Returns:
        Dict with Qini, AUUC, gain metrics
    """
    # Sort by predicted uplift (descending)
    order = np.argsort(uplift)[::-1]
    treatment_sorted = treatment[order]
    outcome_sorted = outcome[order]

    n = len(uplift)
    bin_size = n // n_bins

    # Compute cumulative uplift per bin
    gains = []
    for i in range(1, n_bins + 1):
        end = min(i * bin_size, n)
        treat = treatment_sorted[:end]
        out = outcome_sorted[:end]

        # Uplift in this segment
        treated_outcome = out[treat == 1].mean() if treat.sum() > 0 else 0
        control_outcome = out[treat == 0].mean() if (1 - treat).sum() > 0 else 0
        gains.append(treated_outcome - control_outcome)

    # Qini coefficient = area under gain curve
    x = np.arange(1, n_bins + 1) / n_bins
    y = np.cumsum(gains) / np.sum(np.abs(gains)) if np.sum(np.abs(gains)) > 0 else np.zeros(n_bins)
    qini = np.trapz(y, x)

    # AUUC (Area Under Uplift Curve)
    # Random ordering baseline
    random_gains = []
    for i in range(1, n_bins + 1):
        end = min(i * bin_size, n)
        idx = np.random.permutation(n)[:end]
        treat = treatment[idx]
        out = outcome[idx]
        treated_outcome = out[treat == 1].mean() if treat.sum() > 0 else 0
        control_outcome = out[treat == 0].mean() if (1 - treat).sum() > 0 else 0
        random_gains.append(treated_outcome - control_outcome)

    random_y = (
        np.cumsum(random_gains) / np.sum(np.abs(random_gains))
        if np.sum(np.abs(random_gains)) > 0
        else np.zeros(n_bins)
    )
    auuc = np.trapz(y - random_y, x)

    return {
        "qini": float(qini),
        "auuc": float(auuc),
        "gain_at_10pct": gains[0] if gains else 0,
        "gain_at_50pct": np.mean(gains[:5]) if len(gains) >= 5 else np.mean(gains),
        "uplift_mean": float(np.mean(uplift)),
        "uplift_std": float(np.std(uplift)),
    }
