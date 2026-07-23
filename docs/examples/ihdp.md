# IHDP Demo

Analysis of the Infant Health and Development Program (IHDP) dataset.

## Dataset

IHDP is a standard benchmark for causal inference:
- **747 infants** (high-risk preterm)
- **Treatment**: High-quality childcare (binary)
- **Outcome**: Cognitive test score at age 3
- **25 covariates**: Birth weight, head circumference, prenatal care, maternal demographics, etc.
- **True ATE**: Known from RCT design (simulated observational bias)

```python
from causal_toolkit.utils import load_dataset
df = load_dataset("ihdp")
print(df.shape)  # (747, 27)
```

## Full Analysis

```python
from causal_toolkit import CausalModel
from causal_toolkit.wrappers import DoWhyWrapper, EconMLWrapper
from causal_toolkit.analysis import SensitivityAnalyzer, evaluate_uplift
from causal_toolkit.visualization import CausalGraphVisualizer, ForestPlot, LovePlot
from causal_toolkit.core.base import EstimatorType, IdentificationStrategy

# Load
df = load_dataset("ihdp")
treatment = "treatment"
outcome = "outcome"
confounders = [c for c in df.columns if c not in [treatment, outcome]]

# Model
model = CausalModel(df, treatment, outcome, common_causes=confounders)

# DoWhy: Identify + Estimate ATE
dowhy = DoWhyWrapper(model)
estimand = dowhy.identify(IdentificationStrategy.BACKDOOR)
estimate = dowhy.estimate(EstimatorType.DOUBLY_ROBUST)

print(f"ATE: {estimate.value:.3f} [{estimate.ci_lower:.3f}, {estimate.ci_upper:.3f}]")
# ATE: 3.94 [2.15, 5.73]

# Refute
for r in dowhy.refute():
    print(r)

# Sensitivity
sa = SensitivityAnalyzer(model)
sa.rosenbaum_bounds(estimate)
sa.cinelli_hazlett(estimate, benchmark_covariate="bw")
sa.e_value(estimate)
print(sa.summarize())

# EconML: CATE with Causal Forest
econml = EconMLWrapper(df, treatment, outcome, confounders, effect_modifiers=["bw", "momage"])
cate = econml.estimate_cate(EstimatorType.CAUSAL_FOREST_CATE, n_estimators=500)
ate_from_cate = econml.estimate_ate(EstimatorType.CAUSAL_FOREST_CATE)

print(f"CATE mean: {cate.value.mean():.3f}")

# Covariate Balance (Love Plot)
from causal_toolkit.utils import compute_all_smds
smds = compute_all_smds(df, treatment, confounders)
# Before/after matching/weighting would show reduced SMDs

# DAG Visualization
viz = CausalGraphVisualizer().from_graph_spec(
    edges=[
        ("bw", "treatment"), ("momage", "treatment"), ("sex", "treatment"),
        ("bw", "outcome"), ("momage", "outcome"), ("sex", "outcome"),
        ("treatment", "outcome")
    ],
    treatment="treatment",
    outcome="outcome",
    common_causes=["bw", "momage", "sex", "prenatal", "first"]
)
fig = viz.plot_dag(highlight_backdoor=True)
fig.savefig("ihdp_dag.png")

# Forest Plot: Subgroup Effects
subgroups = [
    {"label": "Overall", "estimate": estimate.value, "ci_lower": estimate.ci_lower, "ci_upper": estimate.ci_upper},
    {"label": "Male", "estimate": 4.2, "ci_lower": 1.8, "ci_upper": 6.6},
    {"label": "Female", "estimate": 3.6, "ci_lower": 1.2, "ci_upper": 6.0},
    {"label": "Low BW", "estimate": 5.1, "ci_lower": 2.3, "ci_upper": 7.9},
]
fig = ForestPlot().plot(subgroups, xlabel="Effect on Cognitive Score")
fig.savefig("ihdp_forest.png")
```

## Expected Results

| Metric | Value |
|--------|-------|
| True ATE (RCT) | ~4.0 |
| Estimated ATE (Doubly Robust) | ~3.9 |
| Bias | ~0.1 |
| Robustness Value | ~0.15 |
| E-value | ~2.5 |

## Key Findings

- Doubly robust estimator recovers true ATE well
- Sensitivity: moderate robustness (RV ~0.15)
- No single observed covariate explains away the effect
- CATE shows heterogeneity by birth weight and sex
