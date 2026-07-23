# Refutation

Refutation tests falsify the causal claim. If assumptions hold, these should **not reject** the null.

## Available Methods

| Method | Description | Null Hypothesis |
|--------|-------------|-----------------|
| `placebo_treatment` | Replace treatment with random noise | Effect = 0 |
| `placebo_outcome` | Replace outcome with random noise | Effect = 0 |
| `random_common_cause` | Add random covariate to graph | Effect unchanged |
| `data_subset` | Re-estimate on random subset | Effect unchanged |
| `simulated_confounder` | Add synthetic unobserved confounder | Effect robust |
| `add_unobserved_confounder` | Rosenbaum bounds grid search | Find critical Γ |

## Running Refutations

```python
from causal_toolkit.wrappers import DoWhyWrapper
from causal_toolkit.core.base import RefutationMethod

dowhy = DoWhyWrapper(model)
dowhy.identify(IdentificationStrategy.BACKDOOR)
dowhy.estimate(EstimatorType.DOUBLY_ROBUST)

# All default tests
refutations = dowhy.refute()

# Custom selection
refutations = dowhy.refute([
    RefutationMethod.PLACEBO_TREATMENT,
    RefutationMethod.RANDOM_COMMON_CAUSE,
    RefutationMethod.DATA_SUBSET,
])

for r in refutations:
    status = "✗ REJECTED" if r.rejected else "✓ NOT REJECTED"
    print(f"{r.method.value}: {status} (p={r.p_value:.4f})")
```

## Interpretation

| Result | Meaning |
|--------|---------|
| **NOT REJECTED** (p > 0.05) | Good - robust to this challenge |
| **REJECTED** (p < 0.05) | Warning - assumption may be violated |

### Specific Warnings

| Rejected Test | Possible Issue |
|---------------|----------------|
| `placebo_treatment` | Bug in estimation, or treatment not well-defined |
| `placebo_outcome` | Outcome definition issue |
| `random_common_cause` | Sensitive to covariate inclusion |
| `data_subset` | Outliers or influential observations |
| `simulated_confounder` | Moderate unobserved confounding could explain effect |

## Sensitivity as Refutation

```python
from causal_toolkit.analysis import SensitivityAnalyzer

analyzer = SensitivityAnalyzer(model)
result = analyzer.cinelli_hazlett(estimate, benchmark_covariate="income")

if result.conclusion_reversed:
    print("⚠ Confounding as strong as 'income' would reverse conclusion")
    print(f"Robustness Value = {result.robustness_value:.4f}")
```

## Custom Parameters

```python
refutations = dowhy.refute(
    methods=[RefutationMethod.DATA_SUBSET],
    subset_fraction=0.8,      # Keep 80% of data
    num_simulations=50
)

refutations = dowhy.refute(
    methods=[RefutationMethod.SIMULATED_CONFOUNDER],
    confounder_strength=0.3,  # Correlation with T and Y
    num_simulations=20
)
```

## RefutationResult Structure

```python
@dataclass
class RefutationResult:
    method: RefutationMethod          # Which test
    null_hypothesis: str              # What was tested
    test_statistic: float             # Test stat value
    p_value: float                    # p-value
    rejected: bool                    # p < 0.05
    details: Dict[str, Any]           # Full test output
```
