# DoWhy Wrapper

Unified interface to DoWhy's causal engine with our type system.

## Quick Start

```python
from causal_toolkit import CausalModel
from causal_toolkit.wrappers import DoWhyWrapper
from causal_toolkit.core.base import EstimatorType, IdentificationStrategy

model = CausalModel(df, "treatment", "outcome", common_causes=["X1", "X2"])
dowhy = DoWhyWrapper(model)

# Full pipeline
estimand = dowhy.identify(IdentificationStrategy.BACKDOOR)
estimate = dowhy.estimate(EstimatorType.DOUBLY_ROBUST)
refutations = dowhy.refute()

print(estimand)
print(estimate)
for r in refutations:
    print(r)
```

## Identification

```python
# Backdoor (default)
estimand = dowhy.identify(IdentificationStrategy.BACKDOOR)

# Frontdoor
estimand = dowhy.identify(
    IdentificationStrategy.FRONTDOOR,
    mediator="compliance"
)

# Instrumental Variable
estimand = dowhy.identify(
    IdentificationStrategy.INSTRUMENTAL_VARIABLE,
    instrumental_variables=["distance"]
)

# Mediation
estimand = dowhy.identify(
    IdentificationStrategy.MEDIATION,
    mediators=["M1", "M2"]
)
```

## Estimators

```python
# ATE estimators
estimate = dowhy.estimate(EstimatorType.LINEAR_REGRESSION)
estimate = dowhy.estimate(EstimatorType.PROPENSITY_SCORE_MATCHING, n_neighbors=5)
estimate = dowhy.estimate(EstimatorType.PROPENSITY_SCORE_WEIGHTING, stabilized=True)
estimate = dowhy.estimate(EstimatorType.DOUBLY_ROBUST)  # Recommended default
estimate = dowhy.estimate(EstimatorType.TARGETED_MAXIMUM_LIKELIHOOD)

# IV
estimate = dowhy.estimate(EstimatorType.TWO_STAGE_LS)

# Access diagnostics
print(estimate.diagnostics)
```

## Refutation

```python
# All default
results = dowhy.refute()

# Specific methods
results = dowhy.refute([
    RefutationMethod.PLACEBO_TREATMENT,
    RefutationMethod.RANDOM_COMMON_CAUSE,
    RefutationMethod.DATA_SUBSET,
    RefutationMethod.SIMULATED_CONFOUNDER,
])

# With parameters
results = dowhy.refute(
    methods=[RefutationMethod.DATA_SUBSET],
    subset_fraction=0.8,
    num_simulations=30
)
```

## Sensitivity Analysis

```python
result = dowhy.sensitivity_analysis(method="cinelli_hazlett")
# Returns SensitivityResult

# Or use SensitivityAnalyzer directly
from causal_toolkit.analysis import SensitivityAnalyzer
analyzer = SensitivityAnalyzer(model)
analyzer.cinelli_hazlett(estimate)
```

## Graph Operations

```python
# Get DoWhy's graph
graph = dowhy._dowhy_model

# Identify backdoor variables
backdoor = graph.get_backdoor_variables()

# Get adjustment sets
adj_sets = graph.get_all_adjustment_sets()

# Check if identified
print(estimand.identification_method)
```

## Advanced: Custom Estimators

```python
from sklearn.ensemble import RandomForestRegressor

estimate = dowhy.estimate(
    EstimatorType.DOUBLY_ROBUST,
    propensity_model=RandomForestRegressor(n_estimators=100),
    outcome_model=RandomForestRegressor(n_estimators=100),
)
```

## Estimand & Estimate Objects

```python
# CausalEstimand
estimand.expression          # E[E[Y|T=t, X] - E[Y|T=t', X]]
estimand.estimand_type       # "ATE" / "ATT" / "ATC"
estimand.adjustment_set      # ["X1", "X2"]
estimand.assumptions         # Assumptions object
estimand.identification_method # IdentificationStrategy.BACKDOOR

# CausalEstimate
estimate.value               # Point estimate (float or np.ndarray)
estimate.ci_lower            # Lower CI bound
estimate.ci_upper            # Upper CI bound
estimate.standard_error      # Standard error
estimate.p_value             # p-value
estimate.n_samples           # Effective sample size
estimate.diagnostics         # Method-specific diagnostics
estimate.is_significant      # p < 0.05 (or CI excludes 0)
```
