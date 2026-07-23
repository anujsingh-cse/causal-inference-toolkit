# Visualization

Plotting utilities for causal inference results.

## Causal Graphs

```python
from causal_toolkit.visualization import CausalGraphVisualizer

# From edge list
viz = CausalGraphVisualizer().from_graph_spec(
    edges=[
        ("age", "treatment"), ("income", "treatment"),
        ("age", "outcome"), ("income", "outcome"),
        ("treatment", "outcome")
    ],
    treatment="treatment",
    outcome="outcome",
    common_causes=["age", "income"]
)

# Static plot
fig = viz.plot_dag(highlight_backdoor=True, highlight_adjustment=["age", "income"])
fig.savefig("dag.png", dpi=300)

# Interactive (Plotly)
fig = viz.plot_interactive(highlight_backdoor=True)
fig.write_html("dag.html")

# Graphviz export
dot = viz.to_graphviz()
dot.render("dag", format="png")

# Do-calculus steps
for step in viz.compute_do_calculus_steps():
    print(step)
```

## Forest Plots

```python
from causal_toolkit.visualization import ForestPlot

estimates = [
    {"label": "Overall", "estimate": 2.5, "ci_lower": 1.2, "ci_upper": 3.8, "is_overall": True},
    {"label": "Subgroup A", "estimate": 3.1, "ci_lower": 1.5, "ci_upper": 4.7},
    {"label": "Subgroup B", "estimate": 1.8, "ci_lower": 0.2, "ci_upper": 3.4},
    {"label": "Subgroup C", "estimate": 2.9, "ci_lower": 1.1, "ci_upper": 4.7},
]

fig = ForestPlot().plot(estimates, xlabel="Treatment Effect", title="Subgroup Effects")
fig.savefig("forest.png")

# Interactive
fig = ForestPlot().plot_interactive(estimates)
fig.write_html("forest.html")
```

## Love Plots (Covariate Balance)

```python
from causal_toolkit.visualization import LovePlot
from causal_toolkit.utils import compute_all_smds

# Before/after matching/weighting
smds = compute_all_smds(df, "treatment", covariates=["age", "income", "education"])
# Returns: {"age": {"smd": 0.35, "balanced": False}, ...}

# Or manual dict
standardized_diffs = {
    "age": (0.45, 0.08),
    "income": (0.32, 0.05),
    "education": (0.12, 0.03),
}

fig = LovePlot().plot(standardized_diffs, threshold=0.1, title="Balance: Before vs After Matching")
fig.savefig("love.png")
```

## Sensitivity Curves

```python
from causal_toolkit.visualization import SensitivityPlot
from causal_toolkit.analysis import SensitivityAnalyzer

analyzer = SensitivityAnalyzer(model)
result = analyzer.cinelli_hazlett(estimate, benchmark_covariate="income")
tips = analyzer.tip_curve(estimate)

fig = SensitivityPlot().plot_cinelli_hazlett(
    r2_yz=tips["r2_yz"],
    r2_zd=tips["r2_zd"],
    adjusted_estimates=tips["adjusted_estimates"],
    significant=tips["significant"],
    rv=result.robustness_value
)
fig.savefig("sensitivity.png")
```

## Uplift / Qini Curves

```python
from causal_toolkit.visualization import UpliftPlot

# Qini curve
fig = UpliftPlot().plot_qini(uplift_scores, T_test, Y_test)
fig.savefig("qini.png")

# Gain/decile chart
fig = UpliftPlot().plot_gain(uplift_scores, T_test, Y_test)
fig.savefig("gain.png")

# Interactive
fig = UpliftPlot().plot_interactive_qini(uplift_scores, T_test, Y_test)
fig.write_html("qini.html")
```

## Counterfactual Distributions

```python
from causal_toolkit.visualization import CounterfactualPlot

# Y(0) vs Y(1) for individual unit
fig = CounterfactualPlot().plot_distributions(
    y0_samples=counterfactual_y0,  # Potential outcome under control
    y1_samples=counterfactual_y1,  # Potential outcome under treatment
    unit_id=42
)
fig.savefig("counterfactual_42.png")

# ITE distribution
fig = CounterfactualPlot().plot_ite_distribution(ite_samples)
fig.savefig("ite_dist.png")
```

## Saving Figures

All plot methods return `matplotlib.figure.Figure` or `plotly.graph_objects.Figure`.

```python
# Matplotlib
fig.savefig("plot.png", dpi=300, bbox_inches="tight")
fig.savefig("plot.pdf", bbox_inches="tight")

# Plotly
fig.write_html("plot.html")
fig.write_image("plot.png", width=800, height=600)
```

## Styling

```python
from causal_toolkit.visualization.plots import CausalGraphVisualizer, ForestPlot, LovePlot
import matplotlib.pyplot as plt

# Global style
plt.style.use('seaborn-v0_8-whitegrid')

# Or per-plot
fig = ForestPlot(figsize=(12, 8)).plot(estimates)
# Use kwargs:
# fig, ax = plt.subplots(figsize=(12, 8))
# ForestPlot().plot(estimates, ax=ax)
```

## Gallery

See [Examples Gallery](../examples/ihdp.md) for full analysis plots.
