# A/B Testing

Complete A/B test analysis: frequentist, Bayesian, sequential, power, multiple testing.

## Quick Start

```python
from causal_toolkit.analysis import ABTestAnalyzer, ABTestData, TestType

analyzer = ABTestAnalyzer(confidence_level=0.95)

# From counts (conversion test)
data = ABTestData(
    n_a=10000, successes_a=1200,
    n_b=10000, successes_b=1350
)
result = analyzer.proportion_ztest(data)
print(result)

# From DataFrame
data = analyzer.from_dataframe(df, variant_col="group", outcome_col="converted",
                                variant_a="control", variant_b="treatment",
                                test_type=TestType.PROPORTION)
result = analyzer.analyze(data, TestType.PROPORTION, method="bayesian")
```

## Frequentist Tests

```python
# Proportion Z-test (conversions)
result = analyzer.proportion_ztest(data, Alternative.TWO_SIDED)

# t-test (continuous metrics: revenue, time, etc.)
data = ABTestData(
    n_a=100, sum_a=50000, sum_sq_a=300000000,
    n_b=100, sum_b=55000, sum_sq_b=350000000
)
result = analyzer.ttest(data)

# Mann-Whitney (non-parametric)
result = analyzer.mann_whitney(control_outcomes, treatment_outcomes)
```

## Bayesian Tests

```python
# Beta-Binomial (proportions)
result = analyzer.bayesian_proportion(
    data,
    prior_alpha=1.0,      # Prior pseudo-counts
    prior_beta=1.0,
    rope_width=0.01       # Region of Practical Equivalence (±1%)
)

# Interpreting Bayesian output:
print(f"P(B > A) = {result.prob_b_better:.2%}")      # Probability B is better
print(f"ROPE = {result.rope_probability:.2%}")       # Probability difference is negligible
print(f"Expected loss if choose A = {result.expected_loss_a:.4f}")
print(f"Expected loss if choose B = {result.expected_loss_b:.4f}")
```

**Decision Rules**:
- `prob_b_better > 0.95` → Ship B
- `rope_probability > 0.9` → No meaningful difference
- `expected_loss_b < 0.001` → Safe to ship B

## Sequential Testing

```python
# SPRT (Wald's Sequential Probability Ratio Test)
sprt = analyzer.sprt(
    data,
    mde=0.05,      # 5% relative MDE
    alpha=0.05,
    beta=0.2       # 80% power
)
# Decision: "accept_h1" / "accept_h0" / "continue"

# mSPRT (Mixture SPRT - Bayesian, optional stopping)
msprt = analyzer.msprt(
    data,
    mde=0.05,
    alpha=0.05,
    prior_strength=1.0
)
# Decision: "accept_h1" / "accept_h0" / "continue"
# Bayes factor threshold = (1-α)/α
```

## Power Analysis

```python
# Sample size for given MDE
power = analyzer.power_analysis(
    baseline_rate=0.10,    # 10% baseline conversion
    mde=0.05,              # 5% relative lift (→ 10.5%)
    alpha=0.05,
    power=0.8,
    ratio=1.0              # Equal allocation
)
# Returns: sample_size_per_variant, total_sample_size, effect_size, achieved_power

# MDE for given sample size
mde = analyzer.mde_calculation(
    baseline_rate=0.10,
    n_per_variant=10000,
    alpha=0.05,
    power=0.8
)
```

## Multiple Testing Correction

```python
p_values = [0.01, 0.03, 0.001, 0.5, 0.02]

# Bonferroni (FWER control)
bonf = analyzer.bonferroni_correction(p_values, alpha=0.05)
# adjusted_alpha = 0.01
# rejected: indices where p < 0.01

# Benjamini-Hochberg (FDR control)
bh = analyzer.benjamini_hochberg(p_values, alpha=0.05)
# More powerful than Bonferroni
```

## End-to-End Analysis

```python
result = analyzer.analyze(
    data=data,
    test_type=TestType.PROPORTION,
    method="bayesian",      # "frequentist", "bayesian", "sequential"
    rope_width=0.01         # For Bayesian
)

# Unified output in ABTestResult
print(f"p-value: {result.p_value}")
print(f"Prob B better: {result.prob_b_better}")
print(f"CI: [{result.ci_lower:.4f}, {result.ci_upper:.4f}]")
```

## Uplift Evaluation

```python
from causal_toolkit.analysis import evaluate_uplift

# Predicted individual uplift (CATE)
uplift = model.predict_uplift(X_test)

metrics = evaluate_uplift(uplift, T_test, Y_test)
# {
#   "qini": 0.142,                 # Qini coefficient
#   "auuc": 0.098,                 # Area Under Uplift Curve
#   "gain_at_10pct": 0.18,         # Gain in top 10%
#   "gain_at_50pct": 0.12,
#   "uplift_mean": 0.05,
#   "uplift_std": 0.03
# }
```

## CLI

```bash
# Proportion test
causal-toolkit ab_test --data exp.csv --variant group --outcome converted \
    --control A --treatment B --method bayesian

# Continuous outcome
causal-toolkit ab_test --data exp.csv --variant group --outcome revenue \
    --control A --treatment B --type mean --method frequentist

# Power calculation
causal-toolkit power --baseline 0.1 --mde 0.05 --alpha 0.05 --power 0.8
```
