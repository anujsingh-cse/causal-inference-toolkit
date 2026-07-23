# Estimation

Methods for estimating causal effects after identification.

## Backdoor Estimators

| Estimator | Type | Best For | Key Parameters |
|-----------|------|----------|----------------|
| `LINEAR_REGRESSION` | Parametric | Simple, interpretable | None |
| `PROPENSITY_SCORE_MATCHING` | Matching | Few covariates, balance needed | `n_neighbors`, `caliper` |
| `PROPENSITY_SCORE_WEIGHTING` | Weighting | Many covariates | `stabilized`, `trim_quantiles` |
| `DOUBLY_ROBUST` | Augmented IPW | Default recommendation | `propensity_model`, `outcome_model` |
| `TARGETED_MAXIMUM_LIKELIHOOD` | TMLE | Efficient, asymptotic | `initial_estimator` |
| `CAUSAL_FOREST` | ML / CATE | Heterogeneous effects | `n_estimators`, `min_samples_leaf` |
| `DOUBLE_ML` | Orthogonal ML | High-dim confounders | `ml_g`, `ml_m` |

## Choosing an Estimator

```
Is treatment effect homogeneous?
├── Yes → linear_regression, doubly_robust (ATE)
└── No (need CATE) → causal_forest, double_ml, or EconML metalearners

Are covariates high-dimensional?
├── Yes → double_ml, causal_forest
└── No → any

Need inference (CI, p-values)?
├── Yes → doubly_robust, tmle, causal_forest (with honest forest)
└── Must be simple → linear_regression
```

## EconML CATE Estimators

| Metalearner | Description | When to Use |
|-------------|-------------|-------------|
| `T_LEARNER` | Separate models for T=0, T=1 | Simple, sparse data |
| `S_LEARNER` | Single model with T as feature | Strong overlap, interactions |
| `X_LEARNER` | Learns treatment/control effects separately | Many control, few treated |
| `R_LEARNER` | Residual-on-residual (Robinson) | General purpose, orthogonality |
| `DR_LEARNER` | Doubly robust residual | Best overall properties |

```python
from causal_toolkit.wrappers import EconMLWrapper
from causal_toolkit.core.base import EstimatorType

econml = EconMLWrapper(
    data=df,
    treatment="treatment",
    outcome="outcome",
    covariates=["X1", "X2", "X3"],
    effect_modifiers=["X1", "X2"]  # CATE varies by these
)

# Get CATE
cate = econml.estimate_cate(EstimatorType.CAUSAL_FOREST_CATE)

# Get ATE (average of CATE)
ate = econml.estimate_ate(EstimatorType.CAUSAL_FOREST_CATE)
```

## Instrumental Variables

| Estimator | Method |
|-----------|--------|
| `TWO_STAGE_LS` | 2SLS, linear first/second stage |
| `DEEP_IV` | Deep learning for complex IV |
| `ORTHO_IV` | Orthogonal IV (Chernozhukov) |

## Estimator Parameters

```yaml
estimation:
  method: "doubly_robust"
  params:
    propensity_model: "logistic"        # or sklearn estimator
    outcome_model: "linear_regression"  # or sklearn estimator
    cv: 5                               # cross-validation folds
```

For causal_forest:
```yaml
estimation:
  method: "causal_forest"
  params:
    n_estimators: 1000
    min_samples_leaf: 10
    max_depth: 10
    honest: true        # Use honest forest for valid inference
    random_state: 42
```

## Diagnostics

```python
estimate = dowhy.estimate(EstimatorType.DOUBLY_ROBUST)
print(estimate.diagnostics)
# {
#   "propensity_model_score": 0.72,
#   "outcome_model_score": 0.65,
#   "effective_sample_size": 847,
#   "weight_summary": {"mean": 1.0, "std": 0.3, "max": 5.2}
# }
```
