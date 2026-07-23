# API: Analysis

## SensitivityAnalyzer

```python
class SensitivityAnalyzer:
    def __init__(self, causal_model: Any = None)

    def rosenbaum_bounds(
        self,
        estimate: CausalEstimate,
        gamma_range: Tuple[float, float] = (1.0, 3.0),
        n_points: int = 50,
        alpha: float = 0.05,
    ) -> SensitivityResult: ...

    def cinelli_hazlett(
        self,
        estimate: CausalEstimate,
        benchmark_covariate: str = None,
        r2_yz_dx: float = None,
        r2_zd_x: float = None,
        k_yz: int = 1,
        k_zd: int = 1,
        alpha: float = 0.05,
    ) -> SensitivityResult: ...

    def e_value(
        self,
        estimate: CausalEstimate,
        true_effect: float = 0.0,
        alpha: float = 0.05,
    ) -> SensitivityResult: ...

    def tip_curve(
        self,
        estimate: CausalEstimate,
        r2_yz_range: Tuple[float, float] = (0, 0.5),
        r2_zd_range: Tuple[float, float] = (0, 0.5),
        n_points: int = 30,
    ) -> Dict[str, Any]: ...

    def summarize(self) -> str: ...

    def _compute_benchmark_r2(self, covariate: str) -> Tuple[float, float]: ...

# Factory
def run_sensitivity_suite(
    estimate: CausalEstimate,
    model: Any = None,
    benchmark_covariates: List[str] = None,
) -> SensitivityAnalyzer: ...
```

## SensitivityResult

```python
@dataclass
class SensitivityResult:
    method: str
    gamma: Optional[float] = None
    robustness_value: Optional[float] = None
    r2_yz_dx: Optional[float] = None
    r2_zd_x: Optional[float] = None
    e_value: Optional[float] = None
    e_value_ci: Optional[float] = None
    benchmark_covariate: Optional[str] = None
    benchmark_r2_yz: Optional[float] = None
    benchmark_r2_zd: Optional[float] = None
    conclusion_reversed: bool = False
    details: Dict[str, Any] = field(default_factory=dict)
```

## ABTestAnalyzer

```python
class ABTestAnalyzer:
    def __init__(self, confidence_level: float = 0.95)

    # Frequentist
    def proportion_ztest(
        self,
        data: ABTestData,
        alternative: Alternative = Alternative.TWO_SIDED,
    ) -> ABTestResult: ...

    def ttest(
        self,
        data: ABTestData,
        alternative: Alternative = Alternative.TWO_SIDED,
        equal_var: bool = False,
    ) -> ABTestResult: ...

    def mann_whitney(
        self,
        data_a: np.ndarray,
        data_b: np.ndarray,
        alternative: Alternative = Alternative.TWO_SIDED,
    ) -> ABTestResult: ...

    # Sequential
    def sprt(
        self,
        data: ABTestData,
        mde: float = 0.05,
        alpha: float = 0.05,
        beta: float = 0.2,
    ) -> Dict[str, Any]: ...

    def msprt(
        self,
        data: ABTestData,
        mde: float = 0.05,
        alpha: float = 0.05,
        prior_strength: float = 1.0,
    ) -> Dict[str, Any]: ...

    # Bayesian
    def bayesian_proportion(
        self,
        data: ABTestData,
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0,
        rope_width: float = 0.01,
    ) -> ABTestResult: ...

    def bayesian_normal(
        self,
        data: ABTestData,
        prior_mu: float = 0.0,
        prior_sigma: float = 10.0,
        rope_width: float = 0.01,
    ) -> ABTestResult: ...

    # Power & Sample Size
    def power_analysis(
        self,
        baseline_rate: float,
        mde: float,
        alpha: float = 0.05,
        power: float = 0.8,
        ratio: float = 1.0,
    ) -> Dict[str, float]: ...

    def mde_calculation(
        self,
        baseline_rate: float,
        n_per_variant: int,
        alpha: float = 0.05,
        power: float = 0.8,
    ) -> float: ...

    # Multiple Testing
    def bonferroni_correction(
        self,
        p_values: List[float],
        alpha: float = 0.05,
    ) -> Dict: ...

    def benjamini_hochberg(
        self,
        p_values: List[float],
        alpha: float = 0.05,
    ) -> Dict: ...

    # High-level
    def analyze(
        self,
        data: ABTestData,
        test_type: TestType = TestType.PROPORTION,
        method: str = "frequentist",
        **kwargs,
    ) -> ABTestResult: ...

    def from_dataframe(
        self,
        df: pd.DataFrame,
        variant_col: str,
        outcome_col: str,
        variant_a: str,
        variant_b: str,
        test_type: TestType = TestType.PROPORTION,
    ) -> ABTestData: ...
```

## ABTestResult

```python
@dataclass
class ABTestResult:
    test_type: TestType
    alternative: Alternative
    estimate_a: float
    estimate_b: float
    difference: float
    relative_difference: float
    statistic: float
    p_value: float
    ci_lower: float
    ci_upper: float
    confidence_level: float
    n_a: int
    n_b: int
    power: Optional[float] = None
    mde: Optional[float] = None
    prob_b_better: Optional[float] = None
    rope_probability: Optional[float] = None
    expected_loss_a: Optional[float] = None
    expected_loss_b: Optional[float] = None
```

## Enums

```python
class TestType(str, Enum):
    PROPORTION = "proportion"
    MEAN = "mean"
    REVENUE = "revenue"
    RATIO = "ratio"

class Alternative(str, Enum):
    TWO_SIDED = "two-sided"
    GREATER = "greater"
    LESS = "less"
```

## Uplift Evaluation

```python
def evaluate_uplift(
    uplift: np.ndarray,
    treatment: np.ndarray,
    outcome: np.ndarray,
    n_bins: int = 10,
) -> Dict[str, float]:
    """
    Returns:
    {
        "qini": float,
        "auuc": float,
        "gain_at_10pct": float,
        "gain_at_50pct": float,
        "uplift_mean": float,
        "uplift_std": float,
    }
    """
```
