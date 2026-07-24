"""
Core causal inference types and base classes.

Defines the foundational data structures for the toolkit:
- CausalModel: Main interface for causal analysis
- CausalEstimand: Identification result (expression + assumptions)
- CausalEstimate: Estimation result (point estimate + uncertainty)
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd


class IdentificationStrategy(str, Enum):
    """Supported identification strategies."""

    BACKDOOR = "backdoor"
    FRONTDOOR = "frontdoor"
    INSTRUMENTAL_VARIABLE = "iv"
    MEDIATION = "mediation"
    REGRESSION_DISCONTINUITY = "rd"
    DIFFERENCE_IN_DIFFERENCES = "did"
    SYNTHETIC_CONTROL = "synthetic_control"


class EstimatorType(str, Enum):
    """Supported estimator types."""

    # Backdoor estimators
    LINEAR_REGRESSION = "linear_regression"
    PROPENSITY_SCORE_MATCHING = "propensity_score_matching"
    PROPENSITY_SCORE_WEIGHTING = "propensity_score_weighting"
    PROPENSITY_SCORE_STRATIFICATION = "propensity_score_stratification"
    DOUBLY_ROBUST = "doubly_robust"
    TARGETED_MAXIMUM_LIKELIHOOD = "tmle"
    CAUSAL_FOREST = "causal_forest"
    DOUBLE_ML = "double_ml"

    # IV estimators
    TWO_STAGE_LS = "2sls"
    DEEP_IV = "deepiv"
    ORTHO_IV = "orthoiv"

    # Uplift / CATE estimators
    T_LEARNER = "t_learner"
    S_LEARNER = "s_learner"
    X_LEARNER = "x_learner"
    R_LEARNER = "r_learner"
    DR_LEARNER = "dr_learner"
    CAUSAL_FOREST_CATE = "causal_forest_cate"
    METALearners = "metalearners"

    # Mediation
    MEDIATION_G_COMPUTATION = "mediation_g_computation"
    SEQUENTIAL_G_ESTIMATION = "sequential_g_estimation"


class RefutationMethod(str, Enum):
    """Supported refutation methods."""

    PLACEBO_TREATMENT = "placebo_treatment"
    PLACEBO_OUTCOME = "placebo_outcome"
    RANDOM_COMMON_CAUSE = "random_common_cause"
    DATA_SUBSET = "data_subset"
    SIMULATED_CONFOUNDER = "simulated_confounder"
    ADD_UNOBSERVED_CONFOUNDER = "add_unobserved_confounder"


@dataclass(frozen=True)
class Assumptions:
    """Causal identification assumptions."""

    unconfoundedness: bool = True
    positivity: bool = True
    consistency: bool = True
    sutva: bool = True
    no_interference: bool = True
    correct_model_specification: bool = False

    def validate(self) -> list[str]:
        """Return list of violated assumptions."""
        violations = []
        if not self.unconfoundedness:
            violations.append("Unconfoundedness violated: unobserved confounders likely")
        if not self.positivity:
            violations.append("Positivity violated: some treatment levels have zero probability")
        if not self.consistency:
            violations.append("Consistency violated: treatment definition ambiguous")
        if not self.sutva:
            violations.append("SUTVA violated: interference between units")
        return violations


@dataclass
class CausalEstimand:
    """
    Identification result: causal estimand expression + identifying assumptions.

    Represents the causal quantity to estimate (e.g., E[Y|do(T=t)]) and the
    assumptions under which it is identified from observed data.
    """

    expression: str  # e.g., "E[E[Y|T=t, X] - E[Y|T=t', X]]"
    estimand_type: str  # "ATE", "ATT", "ATC", "CATE", "LATE"
    treatment: str
    outcome: str
    adjustment_set: list[str] = field(default_factory=list)
    instrumental_variables: list[str] = field(default_factory=list)
    mediators: list[str] = field(default_factory=list)
    assumptions: Assumptions = field(default_factory=Assumptions)
    identification_method: IdentificationStrategy = IdentificationStrategy.BACKDOOR
    metadata: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return f"{self.estimand_type}: {self.expression}"


@dataclass
class CausalEstimate:
    """
    Estimation result: point estimate, confidence interval, diagnostics.

    Attributes:
        value: Point estimate (float or array for CATE)
        ci_lower: Lower confidence bound
        ci_upper: Upper confidence bound
        confidence_level: e.g., 0.95
        estimator: Name of estimator used
        standard_error: Standard error of estimate
        p_value: p-value for null hypothesis of zero effect
        n_samples: Effective sample size
        diagnostics: Estimator-specific diagnostics dict
    """

    value: float | np.ndarray
    ci_lower: float | np.ndarray
    ci_upper: float | np.ndarray
    confidence_level: float = 0.95
    estimator: str = ""
    standard_error: float | None = None
    p_value: float | None = None
    n_samples: int = 0
    diagnostics: dict[str, Any] = field(default_factory=dict)
    estimand: CausalEstimand | None = None

    def __post_init__(self) -> None:
        if isinstance(self.value, float):
            self.value = np.float64(self.value)
        if isinstance(self.ci_lower, float):
            self.ci_lower = np.float64(self.ci_lower)
        if isinstance(self.ci_upper, float):
            self.ci_upper = np.float64(self.ci_upper)

    @property
    def is_significant(self) -> bool:
        return self.p_value is not None and self.p_value < (1 - self.confidence_level)

    @property
    def margin_of_error(self) -> float | np.ndarray:
        if isinstance(self.value, np.ndarray):
            return (self.ci_upper - self.ci_lower) / 2
        return float((self.ci_upper - self.ci_lower) / 2)

    def __str__(self) -> str:
        if isinstance(self.value, np.ndarray):
            mean_val = float(np.mean(self.value))
            ci_l = float(np.mean(self.ci_lower))
            ci_u = float(np.mean(self.ci_upper))
            return (
                f"CATE estimate (n={len(self.value)}): "
                f"mean={mean_val:.4f}, CI=[{ci_l:.4f}, {ci_u:.4f}]"
            )
        return f"{self.value:.4f} [{self.ci_lower:.4f}, {self.ci_upper:.4f}]"


@dataclass
class RefutationResult:
    """Result of a refutation test."""

    method: RefutationMethod
    null_hypothesis: str
    test_statistic: float
    p_value: float
    rejected: bool
    details: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        status = "REJECTED" if self.rejected else "NOT REJECTED"
        return f"{self.method.value}: {status} (p={self.p_value:.4f})"


class CausalModel:
    """
    Main interface for causal analysis pipeline.

    Orchestrates: Graph specification -> Identification -> Estimation -> Refutation
    """

    def __init__(
        self,
        data: pd.DataFrame,
        treatment: str,
        outcome: str,
        graph: Any | None = None,
        common_causes: list[str] | None = None,
        instruments: list[str] | None = None,
        effect_modifiers: list[str] | None = None,
        assumptions: Assumptions | None = None,
    ):
        self.data = data.copy()
        self.treatment = treatment
        self.outcome = outcome
        self.graph = graph
        self.common_causes = common_causes or []
        self.instruments = instruments or []
        self.effect_modifiers = effect_modifiers or []
        self.assumptions = assumptions or Assumptions()

        self._estimand: CausalEstimand | None = None
        self._estimate: CausalEstimate | None = None
        self._refutations: list[RefutationResult] = []

    def identify(
        self, strategy: IdentificationStrategy = IdentificationStrategy.BACKDOOR, **kwargs: Any
    ) -> CausalEstimand:
        """Identify causal estimand from graph and data."""
        # Delegates to DoWhy/EconML wrappers
        raise NotImplementedError("Use DoWhyWrapper or EconMLWrapper")

    def refute(
        self, methods: list[RefutationMethod] | None = None, **kwargs: Any
    ) -> list[RefutationResult]:
        """Run refutation tests."""
        raise NotImplementedError("Use DoWhyWrapper or EconMLWrapper")

    def sensitivity_analysis(self, method: str = "cinelli_hazlett", **kwargs: Any) -> Any:
        """Run sensitivity analysis."""
        raise NotImplementedError("Use SensitivityAnalyzer")

    @property
    def estimand(self) -> CausalEstimand | None:
        return self._estimand

    @property
    def estimate(self) -> CausalEstimate | None:
        return self._estimate

    @property
    def refutations(self) -> list[RefutationResult]:
        return self._refutations

    def summary(self) -> str:
        lines = [
            f"CausalModel: {self.treatment} -> {self.outcome}",
            f"  Data: {len(self.data)} rows, {len(self.data.columns)} cols",
            f"  Common causes: {self.common_causes}",
            f"  Instruments: {self.instruments}",
            f"  Effect modifiers: {self.effect_modifiers}",
        ]
        if self._estimand:
            lines.append(f"  Estimand: {self._estimand}")
        if self._estimate:
            lines.append(f"  Estimate: {self._estimate}")
        if self._refutations:
            lines.append(f"  Refutations: {len(self._refutations)} tests")
        return "\n".join(lines)
