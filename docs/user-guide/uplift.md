# Uplift Modeling

Estimate individual treatment effects for targeting and personalization.

## Quick Start

```python
from causal_toolkit.wrappers import UpliftModeler

# Initialize
uplift = UpliftModeler(
    data=df,
    treatment="treatment",
    outcome="outcome",
    covariates=["age", "income", "education", "past_purchases"]
)

# Fit
uplift.fit("causal_forest", n_estimators=500, min_samples_leaf=20)

# Predict uplift (CATE)
uplift_scores = uplift.predict_uplift()

# Target top 20%
top_20 = np.argsort(uplift_scores)[::-1][:int(0.2 * len(uplift_scores))]
```

## Model Selection

| Method | Use Case | Pros | Cons |
|--------|----------|------|------|
| `causal_forest` | General, non-parametric | Honest CIs, handles interactions | Slower, needs more data |
| `two_model` | Simple, interpretable | Fast, easy to explain | High variance if groups small |
| `class_transformation` | Continuous outcome | Efficient, single model | Assumes linear effect |
| `dr_learner` | Best overall | Doubly robust, orthogonality | Complex, more params |

```python
# Compare methods
for method in ["causal_forest", "two_model", "dr_learner", "class_transformation"]:
    uplift = UpliftModeler(df, "T", "Y", covariates=["X1", "X2"])
    uplift.fit(method)
    metrics = uplift.evaluate(X_test, T_test, Y_test)
    print(f"{method}: Qini={metrics['qini']:.4f}, AUUC={metrics['auuc']:.4f}")
```

## Evaluation Metrics)

# Tip: Use causal_forest for production, dr_learner for comparison
```

## Evaluation Metrics

```python
metrics = uplift.evaluate(X_test, T_test, Y_test)

# Metrics explained:
print(f"Qini coefficient: {metrics['qini']:.4f}")   # Area under Qini curve (higher = better)
print(f"AUUC: {metrics['auuc']:.4f}")               # Area under uplift curve vs random
print(f"Gain @ 10%: {metrics['gain_at_10pct']:.4f}") # Lift in top 10% by predicted uplift
print(f"Gain @ 50%: {metrics['gain_at_50pct']:.4f}") # Lift in top 50%
print(f"Mean uplift: {metrics['uplift_mean']:.4f}")
```

| Metric | Range | Interpretation |
|--------|-------|----------------|
| Qini | 0–1 | 1 = perfect ranking, 0 = random |
| AUUC | 0–1 | 1 = perfect, 0 = random ordering |
| Gain@k% | ℝ | Actual mean uplift in top k% |

## Visualization

```python
# Qini curve
fig = uplift.plot_qini(X_test, T_test, Y_test)
fig.savefig("qini.png")

# Gain/decile chart
fig = uplift.plot_gain(X_test, T_test, Y_test)
fig.savefig("gain.png")

# Interactive (Plotly)
fig = uplift.plot_interactive_qini(X_test, T_test, Y_test)
fig.write_html("qini.html")
```

## Targeting Simulation

```python
# Simulate profit from targeting top k% by predicted uplift
def target_profit(uplift, treatment, outcome, cost_per_treatment=5, revenue_per_conversion=100):
    order = np.argsort(uplift)[::-1]
    profits = []
    for k in [0.05, 0.1, 0.2, 0.5, 1.0]:
        n_treat = int(k * len(uplift))
        idx = order[:n_treat]
        
        treated = treatment[idx]
        out = outcome[idx]
        
        conversions = out[treated == 1].sum()
        revenue = conversions * revenue_per_conversion
        cost = n_treat * cost_per_treatment
        profit = revenue - cost
        
        profits.append({"k": k, "n_treat": n_treat, "profit": profit})
    return profits

results = target_profit(uplift_scores, T_test, Y_test)
for r in results:
    print(f"Target top {r['k']:.0%}: profit=${r['profit']:.2f}")
```

## Advanced: Custom Models

```python
from sklearn.ensemble import GradientBoostingRegressor
from econml.metalearners import DRLearner

# Custom DR-Learner
model = DRLearner(
    model_regression=GradientBoostingRegressor(n_estimators=300, max_depth=3),
    model_propensity=GradientBoostingClassifier(n_estimators=200)
)
model.fit(Y, T, X=X)

# Or use EconMLWrapper directly for more control
from causal_toolkit.wrappers import EconMLWrapper
from causal_toolkit.core.base import EstimatorType

econml = EconMLWrapper(df, "T", "Y", covariates=["X1", "X2"])
cate = econml.estimate_cate(EstimatorType.DR_LEARNER, 
    models=GradientBoostingRegressor(n_estimators=300),
    propensity_model=GradientBoostingClassifier(n_estimators=200)
)
```

## Choosing Covariates for Uplift

```python
# All potential confounders + effect modifiers
covariates = [
    # Demographics (confounders)
    "age", "sex", "income", "education",
    # Behavioral (effect modifiers)
    "past_purchases", "avg_order_value", "days_since_last_purchase",
    # Contextual
    "channel", "device", "time_of_day"
]

# Effect modifiers: where treatment effect varies
effect_modifiers = ["age", "past_purchases", "avg_order_value"]
```

## Common Pitfalls

| Pitfall | Solution |
|---------|----------|
| Overfitting on small data | Increase `min_samples_leaf`, use simpler model |
| No randomized data | Ensure strong overlap, use doubly robust |
| Imbalanced treatment | Use X-learner or weighted methods |
| Sequential decisions | Model time-varying effects separately |
| Simpson's paradox | Always adjust for confounders first |
