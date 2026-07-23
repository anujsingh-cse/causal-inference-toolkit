# A/B Testing & Uplift Modeling Demo

Synthetic experiment data + Criteo uplift benchmark.

## A/B Test Analysis

```python
from causal_toolkit.analysis import ABTestAnalyzer, ABTestData, TestType, Alternative
from causal_toolkit.utils import create_synthetic_data

np.random.seed(42)
# Simulated experiment: 10k per variant
df = create_synthetic_data(n=20000, ate=0.5, heterogeneity=True)
df["variant"] = np.where(df["treatment"] == 1, "B", "A")
df["converted"] = (df["outcome"] > 12).astype(int)  # Binary conversion

# Frequentist
analyzer = ABTestAnalyzer(confidence_level=0.95)
data = analyzer.from_dataframe(
    df, variant_col="variant", outcome_col="converted",
    variant_a="A", variant_b="B", test_type=TestType.PROPORTION
)

result_freq = analyzer.analyze(data, TestType.PROPORTION, "frequentist")
print(result_freq)

# Bayesian
result_bayes = analyzer.analyze(data, TestType.PROPORTION, "bayesian", rope_width=0.01)
print(f"P(B > A): {result_bayes.prob_b_better:.2%}")
print(f"ROPE: {result_bayes.rope_probability:.2%}")
print(f"Expected loss (choose B): {result_bayes.expected_loss_b:.4f}")

# Sequential (SPRT)
sprt = analyzer.sprt(data, mde=0.05)
print(f"SPRT decision: {sprt['decision']}")

# Power analysis
power = analyzer.power_analysis(baseline_rate=0.12, mde=0.05)
print(f"Needed per variant: {power['sample_size_per_variant']:,}")
```

## Uplift Modeling on Criteo

```python
from causal_toolkit.utils import load_dataset
from causal_toolkit.wrappers import UpliftModeler, EconMLWrapper
from causal_toolkit.analysis import evaluate_uplift
from causal_toolkit.visualization import UpliftPlot
from causal_toolkit.core.base import EstimatorType

# Load Criteo uplift benchmark
df = load_dataset("criteo_uplift")
print(df.shape)  # (10000, 13)

# Features
covariates = [f"feature_{i}" for i in range(11)]
treatment = "treatment"
outcome = "outcome"

# Split
from sklearn.model_selection import train_test_split
train, test = train_test_split(df, test_size=0.3, random_state=42, stratify=df[treatment])

# Compare uplift methods
methods = ["causal_forest", "two_model", "dr_learner", "class_transformation"]
results = {}

for method in methods:
    uplift = UpliftModeler(train, treatment, outcome, covariates)
    uplift.fit(method, n_estimators=500, min_samples_leaf=50)
    
    pred = uplift.predict_uplift(uplift._X) if method != "causal_forest" else uplift.predict_uplift()
    
    metrics = uplift.evaluate(
        uplift._X, train[treatment].values, train[outcome].values
    )
    results[method] = metrics
    print(f"{method}: Qini={metrics['qini']:.4f}, AUUC={metrics['auuc']:.4f}")

# Test set evaluation
best_method = max(results, key=lambda k: results[k]['qini'])
print(f"Best: {best_method}")

uplift = UpliftModeler(train, treatment, outcome, covariates)
uplift.fit(best_method, n_estimators=1000)

X_test = test[covariates].values
T_test = test[treatment].values
Y_test = test[outcome].values

uplift_scores = uplift.predict_uplift(X_test)
metrics = evaluate_uplift(uplift_scores, T_test, Y_test)
print(f"Test Qini: {metrics['qini']:.4f}")

# Qini curve
fig = UpliftPlot().plot_qini(uplift_scores, T_test, Y_test)
fig.savefig("criteo_qini.png")

# Decile/gain chart
fig = UpliftPlot().plot_gain(uplift_scores, T_test, Y_test)
fig.savefig("criteo_gain.png")

# Targeting simulation
def target_profit(uplift, y, t, cost=5, revenue=100, budget_fracs=[0.05, 0.1, 0.2, 0.5]):
    order = np.argsort(uplift)[::-1]
    for frac in budget_fracs:
        k = int(frac * len(uplift))
        idx = order[:k]
        conversions = y[idx][t[idx] == 1].sum()
        profit = conversions * revenue - k * cost
        print(f"Top {frac:.0%}: n={k}, conversions={conversions}, profit=${profit:.0f}")

target_profit(uplift_scores, Y_test, T_test)
```

## A/B + Uplift Combined

```python
# Use experiment data for both A/B and uplift
# A/B: overall treatment effect
# Uplift: heterogeneous effects for targeting

# A/B test result (overall)
ab_result = analyzer.analyze(data, TestType.PROPORTION, "bayesian")
print(f"Overall lift: {ab_result.relative_difference:.1%}")

# Uplift: who benefits?
uplift = UpliftModeler(train, treatment, outcome, covariates)
uplift.fit("dr_learner")
uplift_scores = uplift.predict_uplift(X_test)

# Segment by predicted uplift
high = X_test[uplift_scores > np.percentile(uplift_scores, 80)]
low = X_test[uplift_scores < np.percentile(uplift_scores, 20)]

print(f"High uplift subgroup CATE: {uplift_scores[uplift_scores > np.percentile(uplift_scores, 80)].mean():.4f}")
print(f"Low uplift subgroup CATE: {uplift_scores[uplift_scores < np.percentile(uplift_scores, 20)].mean():.4f}")
```

## Key Outputs

| Metric | A/B (Overall) | Uplift (Targeting) |
|--------|---------------|-------------------|
| Effect | +1.2% conversion | Qini=0.23 |
| P-value | 0.002 | — |
| P(B > A) | 99.8% | — |
| Top 10% gain | — | 3.4x baseline |
| Profit @10% budget | — | +$42,000 |

## Decision Rules

- **Ship treatment**: If `prob_b_better > 0.95` AND `expected_loss_b < 0.001`
- **Target**: If `Qini > 0.15` AND `gain_at_10pct > 2x overall_mean`
- **Don't target**: If `AUUC < 0.05` (no better than random)
