# API: Core

## CausalModel

```python
class CausalModel:
    def __init__(
        self,
        data: pd.DataFrame,
        treatment: str,
        outcome: str,
        graph: Optional[Any] = None,
        common_causes: Optional[List[str]] = None,
        instruments: Optional[List[str]] = None,
        effect_modifiers: Optional[List[str]] = None,
        assumptions: Optional[Assumptions] = None,
    )
```

### Properties
- `treatment: str`
- `outcome: str`
- `common_causes: List[str]`
- `instruments: List[str]`
- `effect_modifiers: List[str]`
- `assumptions: Assumptions`
- `data: pd.DataFrame`
- `estimand: Optional[CausalEstimand]`
- `estimate: Optional[CausalEstimate]`
- `refutations: List[RefutationResult]`

### Methods
```python
def identify(
    self,
    strategy: IdentificationStrategy = IdentificationStrategy.BACKDOOR,
    **kwargs,
) -> CausalEstimand:
    """Identify causal estimand. Raises NotImplementedError - use DoWhyWrapper."""

def estimate(
    self,
    estimator: EstimatorType,
    **estimator_kwargs,
) -> CausalEstimate:
    """Estimate causal effect. Raises NotImplementedError - use wrappers."""

def refute(
    self,
    methods: List[RefutationMethod] = None,
    **kwargs,
) -> List[RefutationResult]:
    """Run refutation tests. Raises NotImplementedError."""

def sensitivity_analysis(
    self,
    method: str = "cinelli_hazlett",
    **kwargs,
) -> Any:
    """Run sensitivity analysis. Raises NotImplementedError."""

def summary(self) -> str:
    """Human-readable model summary."""
```

## CausalEstimand

```python
@dataclass
class CausalEstimand:
    expression: str
    estimand_type: str          # "ATE", "ATT", "ATC", "CATE", "LATE"
    treatment: str
    outcome: str
    adjustment_set: List[str] = field(default_factory=list)
    instrumental_variables: List[str] = field(default_factory=list)
    mediators: List[str] = field(default_factory=list)
    assumptions: Assumptions = field(default_factory=Assumptions)
    identification_method: IdentificationStrategy = IdentificationStrategy.BACKDOOR
    metadata: Dict[str, Any] = field(default_factory=dict)
```

## CausalEstimate

```python
@dataclass
class CausalEstimate:
    value: Union[float, np.ndarray]
    ci_lower: Union[float, np.ndarray]
    ci_upper: Union[float, np.ndarray]
    confidence_level: float = 0.95
    estimator: str = ""
    standard_error: Optional[float] = None
    p_value: Optional[float] = None
    n_samples: int = 0
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    estimand: Optional[CausalEstimand] = None

    @property
    def is_significant(self) -> bool:
        """Whether p-value < alpha (1 - confidence_level)."""

    @property
    def margin_of_error(self) -> Union[float, np.ndarray]:
        """(ci_upper - ci_lower) / 2"""
```

## RefutationResult

```python
@dataclass
class RefutationResult:
    method: RefutationMethod
    null_hypothesis: str
    test_statistic: float
    p_value: float
    rejected: bool
    details: Dict[str, Any] = field(default_factory=dict)
```

## Assumptions

```python
@dataclass(frozen=True)
class Assumptions:
    unconfoundedness: bool = True
    positivity: bool = True
    consistency: bool = True
    sutva: bool = True
    no_interference: bool = True
    correct_model_specification: bool = False

    def validate(self) -> List[str]:
        """Return list of violated assumptions."""
```

## Enums

```python
class IdentificationStrategy(str, Enum):
    BACKDOOR = "backdoor"
    FRONTDOOR = "frontdoor"
    INSTRUMENTAL_VARIABLE = "iv"
    MEDIATION = "mediation"
    REGRESSION_DISCONTINUITY = "rd"
    DIFFERENCE_IN_DIFFERENCES = "did"
    SYNTHETIC_CONTROL = "synthetic_control"

class EstimatorType(str, Enum):
    LINEAR_REGRESSION = "linear_regression"
    PROPENSITY_SCORE_MATCHING = "propensity_score_matching"
    PROPENSITY_SCORE_WEIGHTING = "propensity_score_weighting"
    PROPENSITY_SCORE_STRATIFICATION = "propensity_score_stratification"
    DOUBLY_ROBUST = "doubly_robust"
    TARGETED_MAXIMUM_LIKELIHOOD = "tmle"
    CAUSAL_FOREST = "causal_forest"
    DOUBLE_ML = "double_ml"
    TWO_STAGE_LS = "2sls"
    DEEP_IV = "deepiv"
    ORTHO_IV = "orthoiv"
    T_LEARNER = "t_learner"
    S_LEARNER = "s_learner"
    X_LEARNER = "x_learner"
    R_LEARNER = "r_learner"
    DR_LEARNER = "dr_learner"
    CAUSAL_FOREST_CATE = "causal_forest_cate"
    METALearners = "metalearners"

class RefutationMethod(str, Enum):
    PLACEBO_TREATMENT = "placebo_treatment"
    PLACEBO_OUTCOME = "placebo_outcome"
    RANDOM_COMMON_CAUSE = "random_common_cause"
    DATA_SUBSET = "data_subset"
    SIMULATED_CONFOUNDER = "simulated_confounder"
    ADD_UNOBSERVED_CONFOUNDER = "add_unobserved_confounder"
```
