# Difference-in-Differences (DiD)

The **Difference-in-Differences (DiD)** estimator compares pre/post treatment changes between treatment and control groups to isolate causal treatment effects.

---

## 2x2 DiD Estimation

```python
import pandas as pd
from causal_toolkit.analysis import DifferenceInDifferences

df = pd.read_csv("experiment_panel.csv")

did = DifferenceInDifferences()
result = did.estimate_2x2(
    data=df,
    outcome_col="conversion",
    treatment_col="is_treatment",
    post_col="is_post"
)

print(result.summary())
```

---

## Panel Two-Way Fixed Effects (TWFE) & Parallel Trends Test

```python
result_panel = did.fit_panel(
    data=df,
    unit_col="unit_id",
    time_col="timestamp",
    outcome_col="revenue",
    treatment_col="is_treated",
    treatment_start_time=2022
)

print(f"ATT: {result_panel.att:.4f}")
print(f"Parallel Trends p-value: {result_panel.parallel_trends_pvalue:.4f}")
```
