# Synthetic Control Method

The **Synthetic Control Method (SCM)** estimates causal effects for a single treated unit (e.g. state, company, country) compared against a weighted combination of control donor units.

---

## Quick Example

```python
import pandas as pd
from causal_toolkit.analysis import SyntheticControl

# Load panel data
df = pd.read_csv("panel_data.csv")

# Fit synthetic control
sc = SyntheticControl(random_state=42)
res = sc.fit_predict(
    data=df,
    unit_col="state",
    time_col="year",
    outcome_col="gdp",
    treated_unit="California",
    treatment_time=2015,
    run_placebos=True
)

print(f"ATT Estimate: {res.att:.4f}")
print(f"Pre-RMSPE: {res.pre_rmspe:.4f}")
print(f"Permutation p-value: {res.p_value:.4f}")
```

---

## CLI Usage

```bash
causal-toolkit synthetic-control \
  --data panel.csv \
  --unit state \
  --time year \
  --outcome gdp \
  --treated-unit California \
  --treatment-time 2015 \
  --out ./results/
```
