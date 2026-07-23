# CausalModel

Central class for specifying and running causal analyses.

## Overview

`CausalModel` is the main entry point. It holds:
- Data
- Variable roles (treatment, outcome, confounders, instruments, mediators)
- Causal graph (optional)
- Identification assumptions
- Results cache (estimand, estimate, refutations)

## Constructor

```python
from causal_toolkit import CausalModel

model = CausalModel(
    data=df,                    # pd.DataFrame
    treatment="treatment",      # Treatment column name (str)
    outcome="outcome",          # Outcome column name (str)
    graph=None,                 # Optional: networkx.DiGraph or DOT string
    common_causes=["X1", "X2"], # Confounders (list[str])
    instruments=["Z"],          # Instrumental variables (list[str])
    effect_modifiers=["W"],     # Effect modifiers / heterogeneity vars
    assumptions=Assumptions()   # Identification assumptions
)
```

## Variable Roles

| Role | Purpose | Required For |
|------|---------|--------------|
| `treatment` | The intervention variable | All estimators |
| `outcome` | The response variable | All estimators |
| `common_causes` | Confounders affecting both T and Y | Backdoor identification |
| `instruments` | Affect T but not Y (except via T) | IV identification |
| `effect_modifiers` | Variables for CATE heterogeneity | EconML CATE estimators |

## Properties

```python
model.treatment          # str
model.outcome            # str
model.common_causes      # list[str]
model.instruments        # list[str]
model.effect_modifiers   # list[str]
model.assumptions        # Assumptions dataclass
model.data               # pd.DataFrame (copy)
```

## Results Cache

```python
model.estimand   # CausalEstimand or None
model.estimate   # CausalEstimate or None
model.refutations # list[RefutationResult]
```

## Summary

```python
print(model.summary())
# CausalModel: treatment -> outcome
#   Data: 1000 rows, 15 cols
#   Common causes: ['age', 'income', 'education']
#   Instruments: []
#   Effect modifiers: []
#   Estimand: ATE: E[E[Y|T=t, X] - E[Y|T=t', X]]
#   Estimate: 2.34 [1.89, 2.79]
#   Refutations: 3 tests
```

## Assumptions

```python
from causal_toolkit.core.base import Assumptions

assumptions = Assumptions(
    unconfoundedness=True,        # No unmeasured confounders
    positivity=True,              # 0 < P(T|X) < 1 for all X
    consistency=True,             # Well-defined interventions
    sutva=True,                   # Stable Unit Treatment Value Assumption
    no_interference=True,         # No interference between units
    correct_model_specification=False  # Model correctly specified
)

violations = assumptions.validate()
# Returns list of violated assumption descriptions
```

---

## Usage with Wrappers

```python
# DoWhy (identification, ATE, refutation)
from causal_toolkit.wrappers import DoWhyWrapper
dowhy = DoWhyWrapper(model)
estimand = dowhy.identify(strategy=IdentificationStrategy.BACKDOOR)
estimate = dowhy.estimate(EstimatorType.DOUBLY_ROBUST)
refutations = dowhy.refute()

# EconML (CATE, heterogeneous effects)
from causal_toolkit.wrappers import EconMLWrapper, UpliftModeler
econml = EconMLWrapper(df, "treatment", "outcome", covariates=["X1", "X2"])
cate = econml.estimate_cate(EstimatorType.CAUSAL_FOREST_CATE)
ate = econml.estimate_ate(EstimatorType.CAUSAL_FOREST_CATE)

# Uplift Modeling
uplift = UpliftModeler(df, "treatment", "outcome", covariates=["X1", "X2"])
uplift.fit(method="causal_forest")
uplift_predictions = uplift.predict_uplift()
metrics = uplift.evaluate(X_test, T_test, Y_test)
```
