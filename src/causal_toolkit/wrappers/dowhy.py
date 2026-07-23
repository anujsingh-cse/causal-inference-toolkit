"""
DoWhy wrapper for causal identification and estimation.

Provides unified interface to DoWhy's causal engine with our type definitions.
"""

from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd

from causal_toolkit.core.base import (
    Assumptions,
    CausalEstimate,
    CausalEstimand,
    CausalModel,
    EstimatorType,
    IdentificationStrategy,
    RefutationMethod,
    RefutationResult,
)


class DoWhyWrapper:
    """
    Wrapper around DoWhy's CausalModel for identification, estimation, refutation.

    Integrates with our CausalEstimand/CausalEstimate types.
    """

    def __init__(self, causal_model: CausalModel):
        self.model = causal_model
        self._dowhy_model = None
        self._build_dowhy_model()

    def _build_dowhy_model(self) -> None:
        """Build DoWhy CausalModel from our specification."""
        try:
            from dowhy import CausalModel as DowhyCausalModel
        except ImportError:
            raise ImportError("DoWhy not installed. Install with: pip install dowhy")

        # Build graph string if not provided
        graph_str = self._build_graph_string()

        self._dowhy_model = DowhyCausalModel(
            data=self.model.data,
            treatment=self.model.treatment,
            outcome=self.model.outcome,
            graph=graph_str,
            common_causes=self.model.common_causes,
            instruments=self.model.instruments,
            effect_modifiers=self.model.effect_modifiers,
        )

    def _build_graph_string(self) -> str:
        """Construct DOT graph string from model specification."""
        if self.model.graph is not None:
            return str(self.model.graph)

        nodes = set([self.model.treatment, self.model.outcome])
        nodes.update(self.model.common_causes)
        nodes.update(self.model.instruments)
        nodes.update(self.model.effect_modifiers)

        edges = []
        for cause in self.model.common_causes:
            edges.append(f"{cause} -> {self.model.treatment}")
            edges.append(f"{cause} -> {self.model.outcome}")
        for iv in self.model.instruments:
            edges.append(f"{iv} -> {self.model.treatment}")
        for em in self.model.effect_modifiers:
            edges.append(f"{em} -> {self.model.treatment}")
            edges.append(f"{em} -> {self.model.outcome}")
        edges.append(f"{self.model.treatment} -> {self.model.outcome}")

        return f"digraph {{\n  {';\n  '.join(edges)};\n}}"

    def identify(
        self,
        strategy: IdentificationStrategy = IdentificationStrategy.BACKDOOR,
        **kwargs,
    ) -> CausalEstimand:
        """Identify causal estimand using DoWhy."""
        if self._dowhy_model is None:
            self._build_dowhy_model()

        # Map strategy to DoWhy method
        dowhy_method = self._map_strategy(strategy)
        identified_estimand = self._dowhy_model.identify_effect(
            method_name=dowhy_method, **kwargs
        )

        # Convert to our type
        estimand = CausalEstimand(
            expression=str(identified_estimand.estimand) if hasattr(identified_estimand, 'estimand') else str(identified_estimand.estimands),
            estimand_type=str(identified_estimand.estimand_type),
            treatment=self.model.treatment,
            outcome=self.model.outcome,
            adjustment_set=list(identified_estimand.get_backdoor_variables() or []),
            instrumental_variables=list(self.model.instruments),
            mediators=list(getattr(identified_estimand, "mediator_variables", []) or []),
            assumptions=self.model.assumptions,
            identification_method=strategy,
            metadata={"dowhy_estimand": identified_estimand},
        )
        self.model._estimand = estimand
        return estimand

    def _map_strategy(self, strategy: IdentificationStrategy) -> str:
        mapping = {
            IdentificationStrategy.BACKDOOR: "default",
            IdentificationStrategy.FRONTDOOR: "frontdoor",
            IdentificationStrategy.INSTRUMENTAL_VARIABLE: "iv",
            IdentificationStrategy.MEDIATION: "mediation",
        }
        return mapping.get(strategy, "default")

    def estimate(
        self,
        estimator: EstimatorType,
        **estimator_kwargs,
    ) -> CausalEstimate:
        """Estimate causal effect using DoWhy's estimators."""
        if self.model._estimand is None:
            self.identify()

        dowhy_estimator = self._map_estimator(estimator)
        estimate = self._dowhy_model.estimate_effect(
            self.model._estimand.metadata["dowhy_estimand"],
            method_name=dowhy_estimator,
            method_params=estimator_kwargs,
        )

        # Extract point estimate and CI
        value = estimate.value
        if hasattr(estimate, "confidence_intervals") and estimate.confidence_intervals:
            ci_lower, ci_upper = estimate.confidence_intervals[0]
        else:
            se = getattr(estimate, "std_err", None)
            if se:
                from scipy import stats
                z = stats.norm.ppf(0.975)
                ci_lower = value - z * se
                ci_upper = value + z * se
            else:
                ci_lower = ci_upper = value

        causal_estimate = CausalEstimate(
            value=value,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            confidence_level=0.95,
            estimator=estimator.value,
            standard_error=getattr(estimate, "std_err", None),
            p_value=getattr(estimate, "p_value", None),
            n_samples=len(self.model.data),
            diagnostics={
                "dowhy_estimate": estimate,
                "method_params": estimator_kwargs,
            },
            estimand=self.model._estimand,
        )
        self.model._estimate = causal_estimate
        return causal_estimate

    def _map_estimator(self, estimator: EstimatorType) -> str:
        mapping = {
            EstimatorType.LINEAR_REGRESSION: "backdoor.linear_regression",
            EstimatorType.PROPENSITY_SCORE_MATCHING: "backdoor.propensity_score_matching",
            EstimatorType.PROPENSITY_SCORE_WEIGHTING: "backdoor.propensity_score_weighting",
            EstimatorType.PROPENSITY_SCORE_STRATIFICATION: "backdoor.propensity_score_stratification",
            EstimatorType.DOUBLY_ROBUST: "backdoor.doubly_robust",
            EstimatorType.TWO_STAGE_LS: "iv.instrumental_variable",
        }
        return mapping.get(estimator, "backdoor.linear_regression")

    def refute(
        self,
        methods: List[RefutationMethod] = None,
        **kwargs,
    ) -> List[RefutationResult]:
        """Run refutation tests."""
        if self.model._estimate is None:
            self.estimate(EstimatorType.LINEAR_REGRESSION)

        if methods is None:
            methods = [
                RefutationMethod.PLACEBO_TREATMENT,
                RefutationMethod.RANDOM_COMMON_CAUSE,
                RefutationMethod.DATA_SUBSET,
            ]

        results = []
        for method in methods:
            dowhy_method = self._map_refutation(method)
            try:
                refutation = self._dowhy_model.refute_estimate(
                    self.model._estimand.metadata["dowhy_estimand"],
                    self.model._estimate.diagnostics["dowhy_estimate"],
                    method_name=dowhy_method,
                    **kwargs,
                )
                result = RefutationResult(
                    method=method,
                    null_hypothesis=f"Effect is zero under {method.value}",
                    test_statistic=refutation.refutation_result.get("test_statistic", 0),
                    p_value=refutation.refutation_result.get("p_value", 1.0),
                    rejected=refutation.refutation_result.get("p_value", 1.0) < 0.05,
                    details=refutation.refutation_result,
                )
            except Exception as e:
                result = RefutationResult(
                    method=method,
                    null_hypothesis=f"Effect is zero under {method.value}",
                    test_statistic=0.0,
                    p_value=1.0,
                    rejected=False,
                    details={"error": str(e)},
                )
            results.append(result)

        self.model._refutations = results
        return results

    def _map_refutation(self, method: RefutationMethod) -> str:
        mapping = {
            RefutationMethod.PLACEBO_TREATMENT: "placebo_treatment_refuter",
            RefutationMethod.PLACEBO_OUTCOME: "placebo_outcome_refuter",
            RefutationMethod.RANDOM_COMMON_CAUSE: "random_common_cause",
            RefutationMethod.DATA_SUBSET: "data_subset_refuter",
            RefutationMethod.SIMULATED_CONFOUNDER: "simulate_unobserved_confounder",
        }
        return mapping.get(method, "random_common_cause")

    def sensitivity_analysis(
        self,
        method: str = "cinelli_hazlett",
        **kwargs,
    ) -> Any:
        """Run sensitivity analysis."""
        from causal_toolkit.analysis.sensitivity import SensitivityAnalyzer
        return SensitivityAnalyzer(self.model).analyze(method=method, **kwargs)


def create_dowhy_model(causal_model: CausalModel) -> DoWhyWrapper:
    """Factory function to create DoWhyWrapper from CausalModel."""
    return DoWhyWrapper(causal_model)