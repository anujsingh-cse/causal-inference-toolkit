# Identification

How causal effects are identified from observed data.

## Strategies

| Strategy | Assumptions | Use Case |
|----------|-------------|----------|
| **Backdoor** | No unobserved confounders | Most common |
| **Frontdoor** | Mediator blocks all paths, no confounder of M→Y | When backdoor blocked |
| **Instrumental Variable** | Z affects T, Z⊥Y given T,X | Endogeneity / RCT non-compliance |
| **Mediation** | No mediator-outcome confounders | Mechanism analysis |
| **Regression Discontinuity** | Cutoff assignment | Sharp/fuzzy RD |

## Backdoor Criterion

A set of variables Z satisfies the backdoor criterion relative to (T, Y) if:
1. No node in Z is a descendant of T
2. Z blocks all backdoor paths between T and Y

```python
from causal_toolkit.visualization import CausalGraphVisualizer

viz = CausalGraphVisualizer().from_graph_spec(
    edges=[("X", "T"), ("X", "Y"), ("T", "Y"), ("U", "T"), ("U", "Y")],
    treatment="T",
    outcome="Y",
    common_causes=["X"]
)

# Find valid adjustment sets
adjustment_sets = viz.find_adjustment_sets()
print(adjustment_sets)
# [['X']]  # X blocks backdoor path T <- U -> Y? No - U unobserved!
# Actually: backdoor path is T <- X -> Y (blocked by X)
# AND T <- U -> Y (unblocked - needs U observed)
```

If unobserved confounders exist → backdoor fails → need IV, frontdoor, or sensitivity analysis.

## Frontdoor Criterion

When direct T→Y path is confounded but there's mediator M:
- T causes M
- M causes Y
- All T→Y paths go through M
- No confounder of M→Y (or observed)

```yaml
pipeline:
  identification:
    strategy: frontdoor
    mediators: ["compliance"]
    treatment: "assignment"
    outcome: "outcome"
```

## Instrumental Variable

Z → T → Y, with Z ⊥ Y | T, X
- Relevance: Z correlates with T
- Exclusion: Z affects Y only through T
- Exchangeability: Z independent of confounders

```python
# Two-stage least squares
from causal_toolkit.core.base import EstimatorType
estimate = dowhy.estimate(EstimatorType.TWO_STAGE_LS)
```

## Checking Identification

```python
# From CausalModel
model = CausalModel(...)
dowhy = DoWhyWrapper(model)
estimand = dowhy.identify(IdentificationStrategy.BACKDOOR)

# Check assumptions
print(estimand.assumptions.validate())
# [] = no violations, or list of warnings

# Graphical check
from causal_toolkit.visualization.graphs import CausalGraphVisualizer
viz = CausalGraphVisualizer().from_graph_spec(...)
print(viz.compute_do_calculus_steps())
# Shows step-by-step do-calculus derivation
```
