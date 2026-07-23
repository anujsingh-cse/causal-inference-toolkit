# Sensitivity Analysis

Quantify how robust your conclusion is to unmeasured confounding.

## Methods Overview

| Method | Treatment Type | Output | Intuition |
|--------|----------------|--------|-----------|
| **Rosenbaum bounds** | Binary | Critical Γ | "How much hidden bias needed to flip result?" |
| **Cinelli-Hazlett** | Continuous | Robustness Value (RV) | "Min confounding strength to change significance" |
| **E-value** | Any (risk ratio) | E-value, E-value CI | "Min risk ratio of confounder to explain away effect" |
| **TIPS curves** | Continuous | Contour plot | "Visualize estimate across confounding space" |

## Rosenbaum Bounds

For matched/stratified binary treatment studies.

```python
from causal_toolkit.analysis import SensitivityAnalyzer

analyzer = SensitivityAnalyzer(model)
result = analyzer.rosenbaum_bounds(estimate, gamma_range=(1.0, 3.0))

print(f"Critical Γ = {result.gamma:.2f}")
# Γ = 1.0: no hidden bias
# Γ = 1.5: moderate hidden bias
# Γ = 2.0: strong hidden bias

if result.conclusion_reversed:
    print("⚠ Conclusion flips at Γ = {:.2f}".format(result.gamma))
```

**Interpretation**: Γ = 1.8 means an unobserved confounder that makes treatment 1.8x more likely for similar units could explain away the effect.

## Cinelli-Hazlett Robustness Value

For linear models with continuous treatment.

```python
result = analyzer.cinelli_hazlett(
    estimate,
    benchmark_covariate="income"  # Compare to observed confounder
)

print(f"RV = {result.robustness_value:.4f}")
print(f"Benchmark RV (income) = {result.benchmark_r2_yz:.4f}")

if result.conclusion_reversed:
    print("⚠ Benchmark covariate explains away effect")
```

**Interpretation**: RV = 0.12 means a confounder explaining 12% of residual variance in both treatment and outcome could change significance.

## E-Value

From VanderWeele & Ding (2017). For risk ratios / odds ratios.

```python
result = analyzer.e_value(estimate)

print(f"E-value = {result.e_value:.2f}")
print(f"E-value (CI) = {result.e_value_ci:.2f}")

# Rule of thumb:
# E-value > 2: moderately robust
# E-value > 5: highly robust
# E-value < 1.25: fragile
```

**Interpretation**: An unmeasured confounder would need a risk ratio of X with both treatment AND outcome to explain away the effect.

## TIPS Curves

Visualize how estimate changes across confounding space.

```python
data = analyzer.tip_curve(
    estimate,
    r2_yz_range=(0, 0.3),
    r2_zd_range=(0, 0.3),
    n_points=50
)

# data contains:
# - r2_yz: outcome confounding strength
# - r2_zd: treatment confounding strength
# - adjusted_estimates: 2D array of estimates
# - significant: 2D boolean array for significance
```

## Full Suite

```python
from causal_toolkit.analysis import run_sensitivity_suite

analyzer = run_sensitivity_suite(
    estimate=estimate,
    model=model,
    benchmark_covariates=["income", "education"]
)

print(analyzer.summarize())
```

**Output**:
```
Sensitivity Analysis Summary
========================================
Rosenbaum bounds: Γ=1.84, conclusion_reversed=True
Cinelli-Hazlett: RV=0.12, R²_yz=0.15, R²_zd=0.08
E-value: 2.45 (CI: 1.67)
  Benchmark 'income': R²_yz=0.08, R²_zd=0.05
  Benchmark 'education': R²_yz=0.03, R²_zd=0.02
```

## Choosing a Method

| Scenario | Recommended |
|----------|-------------|
| Matched/stratified binary T | Rosenbaum |
| Linear regression, continuous T | Cinelli-Hazlett |
| Risk ratios / case-control | E-value |
| Want visual exploration | TIPS curves |
| Compare to observed confounders | Cinelli-Hazlett with benchmarks |
| Full robustness report | `run_sensitivity_suite()` |

## Reporting Template

> "The estimated ATE was X (95% CI: [L, U]). Sensitivity analysis using Cinelli-Hazlett yielded a Robustness Value of RV=0.15, meaning an unobserved confounder explaining at least 15% of residual variance in both treatment and outcome would be needed to reverse the conclusion. The E-value for the point estimate was 2.3 (CI: 1.6), indicating moderate robustness. Results were compared against observed confounders 'income' (RV=0.08) and 'education' (RV=0.03), neither of which individually explain the effect."
