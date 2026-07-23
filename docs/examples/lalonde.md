# Lalonde NSW/PSID Demo

Classic propensity score matching benchmark: National Supported Work (NSW) demonstration vs PSID population.

## Dataset

- **185 treated** (NSW job training)
- **15,992 controls** (PSID-1/PSID-3, synthetic "observational")
- **Outcome**: 1978 earnings (`re78`)
- **Covariates**: age, education, race, marital status, degree, prior earnings (`re74`, `re75`)
- **Confounding**: Treated group disadvantaged on almost all covariates

```python
from causal_toolkit.utils import load_dataset
df = load_dataset("lalonde")
print(df.treatment.value_counts())
# 0    15992
# 1      185
```

## Key Challenge: Selection Bias

```python
# Baseline comparison (naive)
treated = df[df.treatment == 1].re78.mean()   # ~$6,350
control = df[df.treatment == 0].re78.mean()   # ~$7,200
print(f"Naive diff: {treated - control:.0f}")  # Negative! -$850
```

Training program *appears* harmful due to selection bias.

## Matching / Weighting

```python
from causal_toolkit import CausalModel
from causal_toolkit.wrappers import DoWhyWrapper
from causal_toolkit.core.base import EstimatorType, IdentificationStrategy
from causal_toolkit.visualization import LovePlot
from causal_toolkit.utils import compute_all_smds

model = CausalModel(
    df, "treatment", "re78",
    common_causes=["age", "education", "black", "hispanic", "married", "nodegree", "re74", "re75"]
)

# Balance before
before = compute_all_smds(df, "treatment", model.common_causes)
print("Before matching:", {k: v["smd"] for k, v in before.items()})
# age: 0.15, education: 0.11, black: 0.62, re74: 0.28, ...

# Matching
dowhy = DoWhyWrapper(model)
estimate = dowhy.estimate(EstimatorType.PROPENSITY_SCORE_MATCHING, n_neighbors=1)

# Balance after (approximate - DoWhy doesn't expose matched sample easily)
# Would need manual matching for full Love plot
```

## Multiple Estimators Comparison

```python
estimates = []
for est in [
    EstimatorType.LINEAR_REGRESSION,
    EstimatorType.PROPENSITY_SCORE_MATCHING,
    EstimatorType.PROPENSITY_SCORE_WEIGHTING,
    EstimatorType.DOUBLY_ROBUST,
]:
    est_result = dowhy.estimate(est)
    estimates.append({
        "label": est.value,
        "estimate": est_result.value,
        "ci_lower": est_result.ci_lower,
        "ci_upper": est_result.ci_upper,
    })

# Forest plot
from causal_toolkit.visualization import ForestPlot
fig = ForestPlot().plot(estimates, title="Lalonde: ATE by Estimator")
fig.savefig("lalonde_forest.png")
```

## Sensitivity Analysis

```python
from causal_toolkit.analysis import SensitivityAnalyzer

sa = SensitivityAnalyzer(model)
sa.cinelli_hazlett(estimate, benchmark_covariate="re74")
sa.rosenbaum_bounds(estimate)
print(sa.summarize())
```

## Expected Results

| Estimator | ATE ($) | Notes |
|-----------|---------|-------|
| Naive | -850 | Confounded |
| Linear regression | +886 | Model-dependent |
| PS Matching | +1,794 | LaLonde 1986 benchmark |
| PS Weighting | +1,948 | IPTW |
| Doubly robust | +1,821 | Recommended |

True effect (from RCT): **~$1,800**

## Key Lessons

1. **Naive comparison fails** — selection bias masks true effect
2. **Matching/weighting needed** — adjust for confounders
3. **Doubly robust** — best of both worlds (model + propensity)
4. **Overlap matters** — poor overlap for Black participants drives variance
5. **Sensitivity** — results moderately robust to unmeasured confounding
