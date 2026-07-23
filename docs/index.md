# Causal Inference Toolkit

**Production-ready causal inference toolkit** wrapping DoWhy/EconML with sensitivity analysis, A/B testing, uplift modeling, and counterfactual estimation.

[![PyPI version](https://badge.fury.io/py/causal-toolkit.svg)](https://pypi.org/project/causal-toolkit/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://github.com/yourusername/causal-toolkit/actions/workflows/ci.yml/badge.svg)](https://github.com/yourusername/causal-toolkit/actions)
[![Documentation](https://img.shields.io/badge/docs-material-blue.svg)](https://yourusername.github.io/causal-toolkit/)

---

## Why Causal Toolkit?

Most data scientists learn correlation-based ML. But **real decisions need causation**:

| Question | Correlation Approach | Causal Approach |
|----------|---------------------|-----------------|
| "Will this feature increase conversions?" | Train classifier, predict | A/B test + uplift modeling |
| "What's the effect of price change?" | Regression on observational data | IV / diff-in-diff / RCT |
| "Which customers to target?" | High propensity score | Individual treatment effects (CATE) |
| "Is the effect real?" | p-value | Sensitivity analysis + refutation |

This toolkit bridges the gap with **unified APIs**, **visual diagnostics**, and **production-ready workflows**.

---

## Key Features

### 🔬 Causal Identification & Estimation
- **DoWhy integration**: Backdoor, frontdoor, IV, mediation, DiD, synthetic control
- **EconML integration**: T/S/X/R/DR-learners, Causal Forest, Metalearners, DeepIV, OrthoIV
- **Config-driven**: YAML → full pipeline (identification → estimation → refutation)

### 📊 Sensitivity Analysis (Unique!)
- **Canneli-Hazlett**: Robustness Value (RV), R² decompositions, benchmark covariates
- **Rosenbaum bounds**: Γ-sensitivity for matched/stratified studies
- **E-values**: Minimum confounding to explain away results
- **TIPS curves**: Visualize estimate stability

### 🧪 A/B Testing (Frequentist + Bayesian)
- Proportion z-test, t-test, Mann-Whitney, revenue metrics
- **Sequential testing**: SPRT, mSPRT for continuous monitoring
- **Bayesian**: Beta-Binomial, Normal-Normal with ROPE, expected loss
- **Power analysis**: Sample size, MDE, minimum duration

### 🎯 Uplift Modeling (CATE + Targeting)
- T-learner, S-learner, X-learner, R-learner, DR-learner, Causal Forest
- **Evaluation**: Qini, AUUC, gain/decile charts
- **Deployment-ready**: sklearn-compatible API

### 📈 Visualization Suite
- **Causal graphs**: DAGs with backdoor path highlighting, do-calculus steps
- **Forest plots**: Meta-analysis style, subgroup effects
- **Love plots**: Covariate balance (SMD) before/after adjustment
- **Sensitivity contours**: RV, Γ, E-value visualizations
- **Uplift curves**: Qini, AUUC, gain charts

### ⚙️ CLI for Production Workflows
```bash
causal-toolkit estimate --config config.yaml --data data.csv
causal-toolkit sensitivity --data data.csv --treatment T --outcome Y --method all
causal-toolkit ab_test --data data.csv --variant variant --outcome conv --control A --treatment B
causal-toolkit uplift --data data.csv --treatment T --outcome Y --covariates X --model causal_forest
causal-toolkit graph --config graph.yaml --render plotly --backdoor
```

---

## Quick Install

```bash
pip install causal-toolkit
# Or with extras
pip install "causal-toolkit[dev,viz,notebooks]"
```

---

## 30-Second Demo

```python
import pandas as pd
from causal_toolkit import CausalModel, DoWhyWrapper, SensitivityAnalyzer
from causal_toolkit.core.base import EstimatorType, IdentificationStrategy

# Load data
df = pd.read_csv("your_data.csv")

# Build causal model
model = CausalModel(
    data=df,
    treatment="treatment",
    outcome="outcome",
    common_causes=["age", "income", "education"]
)

# Identify & estimate
dowhy = DoWhyWrapper(model)
estimand = dowhy.identify(strategy=IdentificationStrategy.BACKDOOR)
ate = dowhy.estimate(EstimatorType.LINEAR_REGRESSION)

print(f"ATE: {ate.value:.3f} [{ate.ci_lower:.3f}, {ate.ci_upper:.3f}]")

# Sensitivity analysis
analyzer = SensitivityAnalyzer(model)
analyzer.cinelli_hazlett(ate)      # Robustness Value
analyzer.rosenbaum_bounds(ate)      # Γ-bounds
analyzer.e_value(ate)               # E-value
print(analyzer.summarize())

# Refutation tests
refutations = dowhy.refute()
for r in refutations:
    print(r)
```

---

## Documentation

| Section | Description |
|---------|-------------|
| [Installation](getting-started/installation.md) | Setup, dependencies, virtual environments |
| [Quickstart](getting-started/quickstart.md) | End-to-end example in 5 minutes |
| [Configuration](getting-started/configuration.md) | YAML schemas, environment variables |
| [Core Concepts](concepts/causal-model.md) | Causal models, identification, estimation |
| [DoWhy Wrapper](user-guide/dowhy.md) | Identification, estimation, refutation |
| [EconML Wrapper](user-guide/econml.md) | CATE, uplift, heterogeneous effects |
| [A/B Testing](user-guide/ab-testing.md) | Frequentist, Bayesian, sequential |
| [Uplift Modeling](user-guide/uplift.md) | Targeting, evaluation, deployment |
| [Visualization](user-guide/visualization.md) | Graphs, plots, interactive charts |
| [API Reference](api/core.md) | Complete API documentation |

---

## Example Notebooks

| Notebook | Dataset | Focus |
|----------|---------|-------|
| [IHDP](examples/ihdp.md) | Infant Health (747 obs) | Full pipeline + CATE benchmarks |
| [Lalonde](examples/lalonde.md) | NSW/PSID (16K obs) | Selection bias, propensity, forest plots |
| [A/B + Uplift](examples/ab-uplift.md) | Synthetic/Criteo | Testing, sequential, targeting |

Run them:
```bash
jupyter lab examples/notebooks/
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    causal-toolkit                            │
├──────────────┬──────────────┬──────────────┬────────────────┤
│    Core      │  Wrappers    │   Analysis   │  Visualization │
├──────────────┼──────────────┼──────────────┼────────────────┤
│ CausalModel  │ DoWhyWrapper │ Sensitivity  │ CausalGraph    │
│ CausalEstim  │ EconMLWrapper│ ABTest       │ ForestPlot     │
│ Assumptions  │ UpliftModeler│ Uplift       │ LovePlot       │
│ Refutation   │              │ Mediation    │ SensitivityPlot│
└──────────────┴──────────────┴──────────────┴────────────────┘
         ▲              ▲              ▲              ▲
         └──────────────┴──────────────┴──────────────┘
                            │
                     ┌──────▼──────┐
                     │    CLI      │
                     │  estimate   │
                     │ sensitivity │
                     │   ab_test   │
                     │   uplift    │
                     │    graph    │
                     └─────────────┘
```

---

## Contributing

We welcome contributions! See [Contributing Guide](contributing.md).

```bash
# Development setup
git clone https://github.com/yourusername/causal-toolkit
cd causal-toolkit
pip install -e ".[dev]"
pre-commit install
pytest
ruff check .
mypy src/causal_toolkit
```

---

## Citation

```bibtex
@software{causal_toolkit,
  author = {Anuj Singh},
  title = {Causal Inference Toolkit},
  year = {2025},
  url = {https://github.com/yourusername/causal-toolkit}
}
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

**Built with** ❤️ **for the causal inference community**

* [DoWhy](https://github.com/py-why/dowhy) — Microsoft
* [EconML](https://github.com/py-why/EconML) — Microsoft
* [Cinelli & Hazlett (2020)](https://doi.org/10.1080/01621459.2020.1718279) — Sensitivity analysis
* [VanderWeele & Ding (2017)](https://doi.org/10.1093/aje/kwx365) — E-values