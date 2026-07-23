"""
Unit tests for Causal Inference Toolkit.
"""

import numpy as np
import pandas as pd
import pytest

from causal_toolkit.analysis.ab_test import (
    ABTestAnalyzer,
    ABTestData,
    ABTestResult,
    TestType,
    evaluate_uplift,
)
from causal_toolkit.analysis.sensitivity import SensitivityAnalyzer, SensitivityResult
from causal_toolkit.core.base import (
    Assumptions,
    CausalEstimand,
    CausalEstimate,
    CausalModel,
    RefutationMethod,
    RefutationResult,
)
from causal_toolkit.utils.data import compute_smd, create_synthetic_data, propensity_score


class TestCoreTypes:
    """Test core causal inference types."""

    def test_assumptions_validation(self):
        assumptions = Assumptions()
        violations = assumptions.validate()
        assert isinstance(violations, list)
        assert len(violations) == 0

    def test_causal_estimand_creation(self):
        estimand = CausalEstimand(
            expression="E[Y|do(T=1)] - E[Y|do(T=0)]",
            estimand_type="ATE",
            treatment="treatment",
            outcome="outcome",
            adjustment_set=["X1", "X2"],
        )
        assert estimand.estimand_type == "ATE"
        assert "X1" in estimand.adjustment_set

    def test_causal_estimate_properties(self):
        estimate = CausalEstimate(
            value=2.5,
            ci_lower=1.0,
            ci_upper=4.0,
            confidence_level=0.95,
            estimator="linear_regression",
            standard_error=0.75,
            p_value=0.001,
            n_samples=1000,
        )
        assert estimate.is_significant is True
        assert abs(estimate.margin_of_error - 1.5) < 1e-6

    def test_refutation_result(self):
        result = RefutationResult(
            method=RefutationMethod.PLACEBO_TREATMENT,
            null_hypothesis="Effect is zero",
            test_statistic=0.5,
            p_value=0.62,
            rejected=False,
        )
        assert result.rejected is False
        assert "p=0.6200" in str(result)

    def test_causal_model_initialization(self):
        df = pd.DataFrame({"treatment": [0, 1, 0, 1], "outcome": [1, 3, 2, 4], "X1": [1, 2, 3, 4]})
        model = CausalModel(data=df, treatment="treatment", outcome="outcome", common_causes=["X1"])
        assert model.treatment == "treatment"
        assert model.outcome == "outcome"
        assert model.common_causes == ["X1"]


class TestPreprocessing:
    """Test preprocessing utilities."""

    def test_compute_smd(self):
        treated = pd.Series([1, 2, 3, 4, 5])
        control = pd.Series([2, 3, 4, 5, 6])
        smd = compute_smd(treated, control, pooled=True)
        # Means: 3 vs 4, pooled std ~1.58, SMD ≈ -0.63
        assert isinstance(smd, float)
        assert not np.isnan(smd)

    def test_create_synthetic_data(self):
        df = create_synthetic_data(n=100, n_covariates=5, ate=2.0, heterogeneity=True)
        assert len(df) == 100
        assert "treatment" in df.columns
        assert "outcome" in df.columns
        assert "true_cate" in df.columns
        assert df["treatment"].isin([0, 1]).all()

    def test_propensity_score(self):
        df = create_synthetic_data(n=200)
        ps = propensity_score(df, "treatment", ["x0", "x1", "x2"], model="logistic")
        assert len(ps) == 200
        assert np.all((ps > 0) & (ps < 1))


class TestABTest:
    """Test A/B testing utilities."""

    def setup_method(self):
        self.analyzer = ABTestAnalyzer(confidence_level=0.95)

    def test_ab_test_data_properties(self):
        data = ABTestData(n_a=100, successes_a=10, n_b=100, successes_b=15)
        assert data.rate_a == 0.1
        assert data.rate_b == 0.15

    def test_proportion_ztest(self):
        data = ABTestData(n_a=1000, successes_a=100, n_b=1000, successes_b=120)
        result = self.analyzer.proportion_ztest(data)
        assert isinstance(result, ABTestResult)
        assert result.test_type == TestType.PROPORTION
        assert 0 <= result.p_value <= 1
        assert result.ci_lower < result.ci_upper

    def test_ttest(self):
        data = ABTestData(n_a=100, sum_a=500, sum_sq_a=3000, n_b=100, sum_b=550, sum_sq_b=3500)
        result = self.analyzer.ttest(data)
        assert isinstance(result, ABTestResult)
        assert result.test_type == TestType.MEAN
        assert result.estimate_a == 5.0
        assert result.estimate_b == 5.5

    def test_bayesian_proportion(self):
        data = ABTestData(n_a=1000, successes_a=100, n_b=1000, successes_b=120)
        result = self.analyzer.bayesian_proportion(data, rope_width=0.01)
        assert result.prob_b_better is not None
        assert 0 <= result.prob_b_better <= 1
        assert result.rope_probability is not None
        assert 0 <= result.rope_probability <= 1

    def test_power_analysis(self):
        result = self.analyzer.power_analysis(baseline_rate=0.1, mde=0.05, alpha=0.05, power=0.8)
        assert "sample_size_per_variant" in result
        assert result["sample_size_per_variant"] > 0
        assert result["total_sample_size"] > result["sample_size_per_variant"]

    def test_multiple_testing_bonferroni(self):
        p_values = [0.01, 0.03, 0.001, 0.5, 0.02]
        result = self.analyzer.bonferroni_correction(p_values, alpha=0.05)
        assert len(result["rejected"]) == 5
        assert len(result["adjusted_p_values"]) == 5
        assert result["adjusted_alpha"] == 0.01

    def test_multiple_testing_bh(self):
        p_values = [0.01, 0.03, 0.001, 0.5, 0.02]
        result = self.analyzer.benjamini_hochberg(p_values, alpha=0.05)
        assert len(result["rejected"]) == 5
        assert len(result["adjusted_p_values"]) == 5

    def test_evaluate_uplift(self):
        np.random.seed(42)
        n = 1000
        uplift = np.random.normal(0.5, 0.2, n)
        treatment = np.random.binomial(1, 0.5, n)
        outcome = np.random.normal(1.0, 0.5, n) + treatment * uplift

        metrics = evaluate_uplift(uplift, treatment, outcome)
        assert "qini" in metrics
        assert "auuc" in metrics
        assert "gain_at_10pct" in metrics
        assert isinstance(metrics["qini"], float)
        assert isinstance(metrics["auuc"], float)


class TestSensitivity:
    """Test sensitivity analysis."""

    def setup_method(self):
        self.analyzer = SensitivityAnalyzer()

    def test_rosenbaum_bounds(self):
        from causal_toolkit.core.base import CausalEstimate

        estimate = CausalEstimate(
            value=0.5, ci_lower=0.1, ci_upper=0.9, standard_error=0.2, n_samples=100
        )
        result = self.analyzer.rosenbaum_bounds(estimate, gamma_range=(1.0, 3.0))
        assert isinstance(result, SensitivityResult)
        assert result.method == "rosenbaum"
        assert result.gamma is not None

    def test_cinelli_hazlett(self):
        from causal_toolkit.core.base import CausalEstimate

        estimate = CausalEstimate(
            value=0.5, ci_lower=0.1, ci_upper=0.9, standard_error=0.2, n_samples=100
        )
        result = self.analyzer.cinelli_hazlett(estimate)
        assert isinstance(result, SensitivityResult)
        assert result.method == "cinelli_hazlett"
        assert result.robustness_value is not None

    def test_e_value(self):
        from causal_toolkit.core.base import CausalEstimate

        estimate = CausalEstimate(value=0.5, ci_lower=0.1, ci_upper=0.9, n_samples=100)
        result = self.analyzer.e_value(estimate)
        assert isinstance(result, SensitivityResult)
        assert result.method == "evalue"
        assert result.e_value is not None
        assert result.e_value > 1.0


class TestDataLoading:
    """Test dataset loading."""

    def test_load_ihdp(self):
        from causal_toolkit.utils.data import load_dataset

        df = load_dataset("ihdp")
        assert len(df) > 0
        assert "treatment" in df.columns
        assert "outcome" in df.columns

    def test_load_lalonde(self):
        from causal_toolkit.utils.data import load_dataset

        df = load_dataset("lalonde")
        assert len(df) > 0
        assert "treatment" in df.columns
        assert "outcome" in df.columns

    def test_load_criteo_uplift(self):
        from causal_toolkit.utils.data import load_dataset

        df = load_dataset("criteo_uplift")
        assert len(df) > 0
        assert "treatment" in df.columns
        assert "outcome" in df.columns


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
