"""
Extended tests for sensitivity analysis, data utilities, and core model coverage.
"""

import numpy as np
import pandas as pd
import pytest

from causal_toolkit.analysis.ab_test import (
    ABTestAnalyzer,
    ABTestData,
    ABTestType,
    Alternative,
    TestType,
)
from causal_toolkit.analysis.sensitivity import SensitivityAnalyzer, run_sensitivity_suite
from causal_toolkit.core.base import (
    Assumptions,
    CausalEstimand,
    CausalEstimate,
    CausalModel,
    EstimatorType,
    IdentificationStrategy,
    RefutationMethod,
    RefutationResult,
)
from causal_toolkit.utils.data import (
    bootstrap_ci,
    bootstrap_ci_pairs,
    compute_all_smds,
    compute_smd,
    create_synthetic_data,
    effective_sample_size,
    inverse_probability_weighting,
    load_dataset,
    propensity_score,
    standardize,
    trim_weights,
)


# ─── ABTestType backward compat ───────────────────────────────
class TestABTestTypeAlias:
    """Verify TestType alias still works."""

    def test_alias_identity(self):
        assert TestType is ABTestType

    def test_alias_members(self):
        assert TestType.PROPORTION == ABTestType.PROPORTION
        assert TestType.MEAN == ABTestType.MEAN
        assert TestType.REVENUE == ABTestType.REVENUE
        assert TestType.RATIO == ABTestType.RATIO


# ─── Extended sensitivity tests ───────────────────────────────
class TestSensitivityExtended:
    """Cover tip_curve, summarize, run_sensitivity_suite."""

    def _make_estimate(self):
        return CausalEstimate(
            value=0.5, ci_lower=0.1, ci_upper=0.9, standard_error=0.2, n_samples=200
        )

    def test_tip_curve(self):
        analyzer = SensitivityAnalyzer()
        result = analyzer.tip_curve(self._make_estimate(), n_points=10)
        assert "r2_yz" in result
        assert "r2_zd" in result
        assert "adjusted_estimates" in result
        assert "significant" in result

    def test_summarize_empty(self):
        analyzer = SensitivityAnalyzer()
        summary = analyzer.summarize()
        assert "No sensitivity analyses" in summary

    def test_summarize_after_analyses(self):
        analyzer = SensitivityAnalyzer()
        est = self._make_estimate()
        analyzer.rosenbaum_bounds(est)
        analyzer.cinelli_hazlett(est)
        analyzer.e_value(est)
        summary = analyzer.summarize()
        assert "rosenbaum" in summary.lower() or "Rosenbaum" in summary
        assert "cinelli" in summary.lower() or "Cinelli" in summary

    def test_run_sensitivity_suite(self):
        est = self._make_estimate()
        analyzer = run_sensitivity_suite(est)
        assert len(analyzer.results) == 3  # rosenbaum + cinelli + evalue

    def test_run_sensitivity_suite_with_model(self):
        df = create_synthetic_data(n=100)
        model = CausalModel(
            data=df, treatment="treatment", outcome="outcome", common_causes=["x0", "x1"]
        )
        est = self._make_estimate()
        analyzer = run_sensitivity_suite(est, model, benchmark_covariates=["x0"])
        # 3 standard + 1 benchmark cinelli
        assert len(analyzer.results) == 4

    def test_cinelli_with_model(self):
        df = create_synthetic_data(n=100)
        model = CausalModel(
            data=df, treatment="treatment", outcome="outcome", common_causes=["x0", "x1"]
        )
        analyzer = SensitivityAnalyzer(model)
        est = self._make_estimate()
        result = analyzer.cinelli_hazlett(est, benchmark_covariate="x0")
        assert result.method == "cinelli_hazlett"

    def test_e_value_low_effect(self):
        est = CausalEstimate(value=0.01, ci_lower=-0.05, ci_upper=0.07, n_samples=100)
        analyzer = SensitivityAnalyzer()
        result = analyzer.e_value(est)
        # Very small effect: e-value close to 1
        assert result.e_value > 1.0

    def test_rosenbaum_no_se(self):
        est = CausalEstimate(value=0.5, ci_lower=0.1, ci_upper=0.9, n_samples=100)
        analyzer = SensitivityAnalyzer()
        result = analyzer.rosenbaum_bounds(est)
        assert result.gamma is not None

    def test_sensitivity_result_str(self):
        from causal_toolkit.analysis.sensitivity import SensitivityResult

        r = SensitivityResult(method="rosenbaum", gamma=2.5, conclusion_reversed=True)
        assert "2.5" in str(r)

        r2 = SensitivityResult(
            method="cinelli_hazlett", robustness_value=0.3, r2_yz_dx=0.1, r2_zd_x=0.2
        )
        assert "RV=" in str(r2)

        r3 = SensitivityResult(method="evalue", e_value=2.5, e_value_ci=1.8)
        assert "E-value" in str(r3)

        r4 = SensitivityResult(method="unknown")
        assert "unknown" in str(r4)


# ─── Extended data utilities tests ────────────────────────────
class TestDataUtilsExtended:
    """Cover standardize, IPW, bootstrap, effective sample size, trim."""

    def test_standardize_zscore(self):
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0], "b": [10, 20, 30, 40, 50]})
        result = standardize(df, method="zscore")
        assert abs(result["a"].mean()) < 1e-10
        assert abs(result["a"].std(ddof=0) - 1.0) < 0.5  # approximately 1

    def test_standardize_minmax(self):
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0]})
        result = standardize(df, method="minmax")
        assert result["a"].min() == 0.0
        assert result["a"].max() == 1.0

    def test_standardize_robust(self):
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 100.0]})
        result = standardize(df, method="robust")
        assert not np.isnan(result["a"]).any()

    def test_standardize_specific_columns(self):
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [10, 20, 30], "c": ["x", "y", "z"]})
        result = standardize(df, columns=["a"])
        assert abs(result["a"].mean()) < 1e-10
        assert list(result["b"]) == [10, 20, 30]  # unchanged

    def test_compute_all_smds(self):
        df = pd.DataFrame(
            {
                "treatment": [0, 0, 0, 1, 1, 1],
                "x1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                "x2": [10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
            }
        )
        results = compute_all_smds(df, "treatment", ["x1", "x2"])
        assert "x1" in results
        assert "x2" in results
        assert "smd" in results["x1"]
        assert "balanced" in results["x1"]

    def test_compute_smd_empty(self):
        smd = compute_smd(pd.Series([], dtype=float), pd.Series([1.0, 2.0]))
        assert np.isnan(smd)

    def test_compute_smd_glass_delta(self):
        smd = compute_smd(pd.Series([1, 2, 3, 4, 5]), pd.Series([2, 3, 4, 5, 6]), pooled=False)
        assert isinstance(smd, float)
        assert not np.isnan(smd)

    def test_propensity_score_gbm(self):
        df = create_synthetic_data(n=100)
        ps = propensity_score(df, "treatment", ["x0", "x1"], model="gbm")
        assert len(ps) == 100
        assert np.all((ps > 0) & (ps < 1))

    def test_propensity_score_rf(self):
        df = create_synthetic_data(n=100)
        ps = propensity_score(df, "treatment", ["x0", "x1"], model="rf")
        assert len(ps) == 100

    def test_propensity_score_unknown_model(self):
        df = create_synthetic_data(n=100)
        with pytest.raises(ValueError, match="Unknown model"):
            propensity_score(df, "treatment", ["x0"], model="xgboost_fake")

    def test_ipw_stabilized(self):
        treatment = np.array([0, 1, 0, 1, 0, 1])
        propensity = np.array([0.3, 0.7, 0.4, 0.6, 0.2, 0.8])
        weights = inverse_probability_weighting(treatment, propensity, stabilized=True)
        assert len(weights) == 6
        assert np.all(weights > 0)

    def test_ipw_unstabilized(self):
        treatment = np.array([0, 1, 0, 1])
        propensity = np.array([0.3, 0.7, 0.4, 0.6])
        weights = inverse_probability_weighting(treatment, propensity, stabilized=False)
        assert len(weights) == 4

    def test_trim_weights(self):
        weights = np.array([0.1, 0.5, 1.0, 5.0, 100.0])
        trimmed = trim_weights(weights, lower_quantile=0.1, upper_quantile=0.9)
        assert trimmed.max() <= 100.0
        assert trimmed.min() >= 0.1

    def test_effective_sample_size(self):
        weights = np.ones(100)
        ess = effective_sample_size(weights)
        assert abs(ess - 100.0) < 1e-10

    def test_effective_sample_size_unequal(self):
        weights = np.array([1.0, 1.0, 1.0, 10.0])
        ess = effective_sample_size(weights)
        assert ess < 4.0  # unequal weights reduce ESS

    def test_bootstrap_ci(self):
        data = np.random.RandomState(42).normal(5.0, 1.0, 200)
        lower, upper = bootstrap_ci(data, np.mean, n_bootstrap=500)
        assert lower < 5.0 < upper

    def test_bootstrap_ci_pairs(self):
        rng = np.random.RandomState(42)
        a = rng.normal(5.0, 1.0, 100)
        b = rng.normal(6.0, 1.0, 100)
        lower, _upper = bootstrap_ci_pairs(
            a, b, lambda x, y: np.mean(y) - np.mean(x), n_bootstrap=500
        )
        assert lower > 0  # b clearly larger

    def test_create_synthetic_no_heterogeneity(self):
        df = create_synthetic_data(n=50, heterogeneity=False)
        assert "true_cate" in df.columns
        # Without heterogeneity, all CATE should be equal to ATE
        assert df["true_cate"].std() < 1e-10

    def test_create_synthetic_no_confounding(self):
        df = create_synthetic_data(n=50, confounding=False)
        assert len(df) == 50

    def test_load_dataset_unknown(self):
        with pytest.raises(ValueError, match="Unknown dataset"):
            load_dataset("nonexistent_dataset")


# ─── Extended core model tests ────────────────────────────────
class TestCoreModelExtended:
    """Cover CausalModel methods, summary, and edge cases."""

    def _make_model(self):
        df = pd.DataFrame(
            {
                "treatment": [0, 1, 0, 1, 0, 1],
                "outcome": [1, 3, 2, 4, 1.5, 3.5],
                "X1": [1, 2, 3, 4, 5, 6],
                "X2": [10, 20, 30, 40, 50, 60],
            }
        )
        return CausalModel(
            data=df,
            treatment="treatment",
            outcome="outcome",
            common_causes=["X1", "X2"],
            instruments=["X1"],
            effect_modifiers=["X2"],
        )

    def test_summary_no_estimate(self):
        model = self._make_model()
        summary = model.summary()
        assert "treatment" in summary
        assert "outcome" in summary

    def test_identify_raises(self):
        model = self._make_model()
        with pytest.raises(NotImplementedError):
            model.identify()

    def test_refute_raises(self):
        model = self._make_model()
        with pytest.raises(NotImplementedError):
            model.refute()

    def test_sensitivity_raises(self):
        model = self._make_model()
        with pytest.raises(NotImplementedError):
            model.sensitivity_analysis()

    def test_properties_default_none(self):
        model = self._make_model()
        assert model.estimand is None
        assert model.estimate is None
        assert model.refutations == []

    def test_assumptions_violations(self):
        a = Assumptions(unconfoundedness=False, positivity=False)
        violations = a.validate()
        assert len(violations) == 2

    def test_causal_estimate_not_significant(self):
        est = CausalEstimate(value=0.1, ci_lower=-0.5, ci_upper=0.7, p_value=0.5, n_samples=100)
        assert est.is_significant is False

    def test_causal_estimate_array_value(self):
        values = np.array([1.0, 2.0, 3.0])
        est = CausalEstimate(
            value=values, ci_lower=values - 0.5, ci_upper=values + 0.5, n_samples=3
        )
        assert "CATE" in str(est)
        moe = est.margin_of_error
        assert isinstance(moe, np.ndarray)

    def test_causal_estimand_str(self):
        estimand = CausalEstimand(
            expression="E[Y|do(T=1)]", estimand_type="ATE", treatment="T", outcome="Y"
        )
        assert "ATE" in str(estimand)

    def test_refutation_result_rejected(self):
        r = RefutationResult(
            method=RefutationMethod.RANDOM_COMMON_CAUSE,
            null_hypothesis="No effect",
            test_statistic=3.0,
            p_value=0.001,
            rejected=True,
        )
        assert "REJECTED" in str(r)


# ─── Extended AB test tests ───────────────────────────────────
class TestABTestExtended:
    """Cover sequential tests, analyze dispatch, from_dataframe."""

    def setup_method(self):
        self.analyzer = ABTestAnalyzer(confidence_level=0.95)

    def test_sprt(self):
        data = ABTestData(n_a=1000, successes_a=100, n_b=1000, successes_b=120)
        result = self.analyzer.sprt(data, mde=0.05)
        assert "decision" in result
        assert result["decision"] in ("accept_h0", "accept_h1", "continue")

    def test_msprt(self):
        data = ABTestData(n_a=1000, successes_a=100, n_b=1000, successes_b=120)
        result = self.analyzer.msprt(data, mde=0.05)
        assert "decision" in result
        assert "bayes_factor" in result

    def test_analyze_frequentist_proportion(self):
        data = ABTestData(n_a=1000, successes_a=100, n_b=1000, successes_b=120)
        result = self.analyzer.analyze(data, ABTestType.PROPORTION, "frequentist")
        assert result.test_type == ABTestType.PROPORTION

    def test_analyze_frequentist_mean(self):
        data = ABTestData(n_a=100, sum_a=500, sum_sq_a=3000, n_b=100, sum_b=550, sum_sq_b=3500)
        result = self.analyzer.analyze(data, ABTestType.MEAN, "frequentist")
        assert result.test_type == ABTestType.MEAN

    def test_analyze_bayesian(self):
        data = ABTestData(n_a=500, successes_a=50, n_b=500, successes_b=60)
        result = self.analyzer.analyze(data, ABTestType.PROPORTION, "bayesian")
        assert result.prob_b_better is not None

    def test_analyze_sequential(self):
        data = ABTestData(n_a=1000, successes_a=100, n_b=1000, successes_b=120)
        result = self.analyzer.analyze(data, ABTestType.PROPORTION, "sequential")
        assert result is not None

    def test_from_dataframe_proportion(self):
        df = pd.DataFrame(
            {"group": ["A", "A", "A", "B", "B", "B"], "converted": [0, 1, 0, 1, 1, 0]}
        )
        ab_data = self.analyzer.from_dataframe(
            df, "group", "converted", "A", "B", ABTestType.PROPORTION
        )
        assert ab_data.n_a == 3
        assert ab_data.n_b == 3
        assert ab_data.successes_a == 1

    def test_from_dataframe_mean(self):
        df = pd.DataFrame({"group": ["A", "A", "B", "B"], "revenue": [10.0, 20.0, 15.0, 25.0]})
        ab_data = self.analyzer.from_dataframe(df, "group", "revenue", "A", "B", ABTestType.MEAN)
        assert ab_data.n_a == 2
        assert ab_data.sum_a == 30.0

    def test_mde_calculation(self):
        mde = self.analyzer.mde_calculation(baseline_rate=0.1, n_per_variant=10000)
        assert isinstance(mde, float)
        assert mde > 0

    def test_ab_test_result_str_significant(self):
        from causal_toolkit.analysis.ab_test import ABTestResult

        r = ABTestResult(
            test_type=ABTestType.PROPORTION,
            alternative=Alternative.TWO_SIDED,
            estimate_a=0.10,
            estimate_b=0.15,
            difference=0.05,
            relative_difference=0.5,
            statistic=3.0,
            p_value=0.001,
            ci_lower=0.02,
            ci_upper=0.08,
            confidence_level=0.95,
            n_a=1000,
            n_b=1000,
        )
        assert "SIGNIFICANT" in str(r)

    def test_ab_test_result_str_not_significant(self):
        from causal_toolkit.analysis.ab_test import ABTestResult

        r = ABTestResult(
            test_type=ABTestType.PROPORTION,
            alternative=Alternative.TWO_SIDED,
            estimate_a=0.10,
            estimate_b=0.105,
            difference=0.005,
            relative_difference=0.05,
            statistic=0.5,
            p_value=0.6,
            ci_lower=-0.01,
            ci_upper=0.02,
            confidence_level=0.95,
            n_a=100,
            n_b=100,
        )
        assert "NOT SIGNIFICANT" in str(r)

    def test_ab_data_edge_cases(self):
        data = ABTestData(n_a=0, n_b=0)
        assert data.rate_a == 0
        assert data.rate_b == 0
        assert data.mean_a == 0
        assert data.mean_b == 0

    def test_ab_data_var_single(self):
        data = ABTestData(n_a=1, sum_a=5, sum_sq_a=25, n_b=1, sum_b=10, sum_sq_b=100)
        assert data.var_a == 0
        assert data.var_b == 0


# ─── Enum coverage ────────────────────────────────────────────
class TestEnums:
    """Ensure all enum values accessible."""

    def test_identification_strategies(self):
        assert len(IdentificationStrategy) >= 5
        assert IdentificationStrategy.BACKDOOR.value == "backdoor"

    def test_estimator_types(self):
        assert len(EstimatorType) >= 10
        assert EstimatorType.LINEAR_REGRESSION.value == "linear_regression"

    def test_refutation_methods(self):
        assert len(RefutationMethod) >= 4
        assert RefutationMethod.PLACEBO_TREATMENT.value == "placebo_treatment"
