# API: Wrappers

## DoWhyWrapper

```python
class DoWhyWrapper:
    def __init__(self, causal_model: CausalModel)

    def identify(
        self,
        strategy: IdentificationStrategy = IdentificationStrategy.BACKDOOR,
        **kwargs
    ) -> CausalEstimand: ...

    def estimate(
        self,
        estimator: EstimatorType,
        **estimator_kwargs
    ) -> CausalEstimate: ...

    def refute(
        self,
        methods: List[RefutationMethod] = None,
        **kwargs
    ) -> List[RefutationResult]: ...

    def sensitivity_analysis(
        self,
        method: str = "cinelli_hazlett",
        **kwargs
    ) -> Any: ...

    # DoWhy-specific
    def _build_dowhy_model(self) -> None: ...
    def _build_graph_string(self) -> str: ...
    def _map_strategy(self, strategy: IdentificationStrategy) -> str: ...
    def _map_estimator(self, estimator: EstimatorType) -> str: ...
    def _map_refutation(self, method: RefutationMethod) -> str: ...

# Factory
def create_dowhy_model(causal_model: CausalModel) -> DoWhyWrapper: ...
```

## EconMLWrapper

```python
class EconMLWrapper:
    def __init__(
        self,
        data: pd.DataFrame,
        treatment: str,
        outcome: str,
        covariates: List[str],
        effect_modifiers: List[str] = None
    )

    def estimate_cate(
        self,
        estimator: EstimatorType,
        **estimator_kwargs
    ) -> CausalEstimate: ...

    def estimate_ate(
        self,
        estimator: EstimatorType,
        **estimator_kwargs
    ) -> CausalEstimate: ...

    def _get_estimator(self, estimator: EstimatorType, **kwargs) -> BaseEstimator: ...

    # Metalearner factories
    def _t_learner(self, **kwargs) -> TLearner: ...
    def _s_learner(self, **kwargs) -> SLearner: ...
    def _x_learner(self, **kwargs) -> XLearner: ...
    def _r_learner(self, **kwargs) -> RLearner: ...
    def _dr_learner(self, **kwargs) -> DRLearner: ...
    def _causal_forest(self, **kwargs) -> CausalForest: ...
    def _metalearner(self, **kwargs) -> Metalearner: ...

    def _compute_ci(self, model: Any, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]: ...

# Factory
def create_econml_wrapper(
    data: pd.DataFrame,
    treatment: str,
    outcome: str,
    covariates: List[str],
    effect_modifiers: List[str] = None
) -> EconMLWrapper: ...
```

## UpliftModeler

```python
class UpliftModeler:
    def __init__(
        self,
        data: pd.DataFrame,
        treatment: str,
        outcome: str,
        covariates: List[str]
    )

    def fit(
        self,
        method: str = "causal_forest",
        **kwargs
    ) -> "UpliftModeler": ...

    def predict_uplift(self, X: np.ndarray = None) -> np.ndarray: ...

    def evaluate(
        self,
        X_test: np.ndarray,
        T_test: np.ndarray,
        Y_test: np.ndarray
    ) -> Dict[str, float]: ...

    # Internal model fitting
    def _fit_causal_forest(self, **kwargs) -> CausalForest: ...
    def _fit_two_model(self, **kwargs) -> TLearner: ...
    def _fit_class_transformation(self, **kwargs) -> TransformedOutcome: ...
    def _fit_dr_learner(self, **kwargs) -> DRLearner: ...

    # Visualization
    def plot_qini(
        self,
        X_test: np.ndarray,
        T_test: np.ndarray,
        Y_test: np.ndarray
    ) -> plt.Figure: ...

    def plot_gain(
        self,
        X_test: np.ndarray,
        T_test: np.ndarray,
        Y_test: np.ndarray
    ) -> plt.Figure: ...
```
