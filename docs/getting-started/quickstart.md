# Quickstart

Run a complete causal analysis in 5 minutes.

## 1. Get Data

```bash
# Download IHDP dataset (built-in)
causal-toolkit demo --dataset ihdp --out ./data
```

Or load your own CSV:
```python
import pandas as pd
df = pd.read_csv("your_data.csv")
```

## 2. Define Causal Model

```python
from causal_toolkit import CausalModel
from causal_toolkit.wrappers import DoWhyWrapper
from causal_toolkit.core.base import EstimatorType, IdentificationStrategy

# Specify treatment, outcome, and confounders
model = CausalModel(
    data=df,
    treatment="treatment",      # Column name
    outcome="outcome",          # Column name
    common_causes=["age", "income", "education", "sex"]  # Confounders
)
```

## 3. Identify Causal Effect

```python
dowhy = DoWhyWrapper(model)
estimand = dowhy.identify(IdentificationStrategy.BACKDOOR)
print(estimand)
# ATE: E[E[Y|T=t, X] - E[Y|T=t', X]]
```

## 4. Estimate Treatment Effect

```python
# Choose estimator
estimate = dowhy.estimate(EstimatorType.DOUBLY_ROBUST)
print(f"ATE: {estimate.value:.4f} [{estimate.ci_lower:.4f}, {estimate.ci_upper:.4f}]")
```

**Available estimators:**
- `LINEAR_REGRESSION` — Simple OLS adjustment
- `PROPENSITY_SCORE_MATCHING` — Nearest-neighbor matching
- `PROPENSITY_SCORE_WEIGHTING` — IPTW
- `DOUBLY_ROBUST` — AIPW (recommended default)
- `CAUSAL_FOREST` — Heterogeneous effects (EconML)
- `TWO_STAGE_LS` — Instrumental variables

## 5. Refute (Sensitivity Checks)

```python
refutations = dowhy.refute()
for r in refutations:
    print(r)
```

**Methods tested:**
- Placebo treatment (random treatment assignment)
- Random common cause (add noise covariate)
- Data subset (remove random fraction)

## 6. Sensitivity Analysis

```python
from causal_toolkit.analysis import SensitivityAnalyzer

analyzer = SensitivityAnalyzer(model)
analyzer.rosenbaum_bounds(estimate)      # Γ-sensitivity
analyzer.cinelli_hazlett(estimate)      # Robustness value
analyzer.e_value(estimate)              # E-value
print(analyzer.summarize())
```

**Key outputs:**
- **Rosenbaum Γ** — How strong hidden confounding must be to flip conclusion
- **Robustness Value (RV)** — Minimum confounding strength to change significance
- **E-value** — Minimum risk ratio of unmeasured confounder

## 7. Config-Driven Pipeline (YAML)

```yaml
# config.yaml
pipeline:
  identification:
    strategy: backdoor
    adjustment_set: ["age", "income", "education"]
  estimation:
    method: doubly_robust
    params:
      n_estimators: 100
  refutation:
    - placebo_treatment
    - random_common_cause
    - data_subset
  sensitivity:
    method: cinelli_hazlett
    benchmark_covariates: ["age", "income"]
```

```bash
causal-toolkit estimate --config config.yaml --data data.csv --out results/
```

---

## What's Next?

- [A/B Testing](../user-guide/ab-testing.md) — Frequentist & Bayesian experiment analysis
- [Uplift Modeling](../user-guide/uplift.md) — Heterogeneous treatment effects for targeting
- [Visualization](../user-guide/visualization.md) — DAGs, forest plots, love plots, Qini curves
- [Configuration](configuration.md) — Full YAML schema reference
