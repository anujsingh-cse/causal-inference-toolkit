# EconML Wrapper

Unified interface to EconML's CATE/heterogeneous effect estimators.

## Quick Start

```python
from causal_toolkit.wrappers import EconMLWrapper
from causal_toolkit.core.base import EstimatorType

econml = EconMLWrapper(
    data=df,
    treatment="treatment",
    outcome="outcome",
    covariates=["X1", "X2", "X3", "X4"],
    effect_modifiers=["X1", "X2"]  # CATE varies by these
)

# CATE (Conditional Average Treatment Effect)
cate_estimate = econml.estimate_cate(EstimatorType.CAUSAL_FOREST_CATE)

# ATE (Average Treatment Effect)
ate_estimate = econml.estimate_ate(EstimatorType.CAUSAL_FOREST_CATE)
```

## Metalearners

| Type | Class | Best For |
|------|-------|----------|
| `T_LEARNER` | Two separate models | Simple, few covariates |
| `S_LEARNER` | Single model with T as feature | Strong overlap, many interactions |
| `X_LEARNER` | Learn effects in each group | Imbalanced treatment/control |
| `R_LEARNER` | Robinson transformation | General purpose, orthogonality |
| `DR_LEARNER` | Doubly robust residual | **Recommended default** |

```python
# T-Learner
cate = econml.estimate_cate(EstimatorType.T_LEARNER)

# S-Learner  
cate = econml.estimate_cate(EstimatorType.S_LEARNER)

# X-Learner
cate = econml.estimate_cate(EstimatorType.X_LEARNER)

# R-Learner
cate = econml.estimate_cate(EstimatorType.R_LEARNER)

# DR-Learner (doubly robust)
cate = econml.estimate_cate(EstimatorType.DR_LEARNER)
```

## Causal Forest

Non-parametric, provides honest inference.

```python
cate = econml.estimate_cate(
    EstimatorType.CAUSAL_FOREST_CATE,
    n_estimators=1000,
    min_samples_leaf=20,
    max_depth=10,
    honest=True,            # Use honest forest for valid CI
    random_state=42
)

# Built-in confidence intervals (if honest=True)
print(cate.ci_lower)  # Per-unit CIs
print(cate.ci_upper)
```

## Custom Models

```python
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Lasso

# For metalearners
cate = econml.estimate_cate(
    EstimatorType.R_LEARNER,
    models=GradientBoostingRegressor(n_estimators=200),
    propensity_model=GradientBoostingClassifier(n_estimators=100)
)

# For DR-Learner
cate = econml.estimate_cate(
    EstimatorType.DR_LEARNER,
    model_regression=GradientBoostingRegressor(n_estimators=200),
    model_propensity=GradientBoostingClassifier(n_estimators=100)
)
```

## UpliftModeler (High-Level)

Dedicated uplift modeling interface with evaluation.

```python
from causal_toolkit.wrappers import UpliftModeler

uplift = UpliftModeler(df, "treatment", "outcome", covariates=["X1", "X2"])

# Fit
uplift.fit("causal_forest", n_estimators=500)

# Predict individual uplift
pred_uplift = uplift.predict_uplift()  # On training data
pred_uplift = uplift.predict_uplift(X_test)  # On new data

# Evaluate
metrics = uplift.evaluate(X_test, T_test, Y_test)
# {"qini": 0.12, "auuc": 0.08, "gain_at_10pct": 0.15, ...}

# Plot Qini curve
fig = uplift.plot_qini(X_test, T_test, Y_test)
fig.savefig("qini.png")
```

## Available Uplift Methods

| Method | Class | Description |
|--------|-------|-------------|
| `causal_forest` | `CausalForest` | Non-parametric, honest CIs |
| `two_model` | `TLearner` | Separate models per arm |
| `class_transformation` | `TransformedOutcome` | Single model on transformed Y |
| `dr_learner` | `DRLearner` | Doubly robust metalearner |

## Effect Modifiers vs Covariates

```python
# Covariates: used for all estimation (confounding adjustment)
# Effect modifiers: where CATE is allowed to vary

econml = EconMLWrapper(
    data=df,
    treatment="T",
    outcome="Y",
    covariates=["X1", "X2", "X3", "X4", "X5"],  # All confounders
    effect_modifiers=["X1", "X2"]              # Only heterogeneity here
)

# CATE estimated conditional on X1, X2
# Confounding adjusted by X1..X5
```

## Diagnostics

```python
cate_estimate = econml.estimate_cate(EstimatorType.CAUSAL_FOREST_CATE)

print(cate_estimate.diagnostics)
# {
#   "model": CausalForest object,
#   "n_treated": 450,
#   "n_control": 550,
#   "n_estimators": 500,
#   "honest": true
# }
```
