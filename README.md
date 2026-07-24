# Causal Inference Toolkit

[![PyPI version](https://img.shields.io/pypi/v/causal-toolkit.svg)](https://pypi.org/project/causal-toolkit/)
[![Python versions](https://img.shields.io/pypi/pyversions/causal-toolkit.svg)](https://pypi.org/project/causal-toolkit/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Build Status](https://github.com/anujsingh-cse/causal-inference-toolkit/workflows/CI/badge.svg)](https://github.com/anujsingh-cse/causal-inference-toolkit/actions)
[![Documentation](https://img.shields.io/badge/docs-latest-blue.svg)](https://anujsingh-cse.github.io/causal-inference-toolkit/)

**Production-ready causal inference toolkit** wrapping DoWhy/EconML with sensitivity analysis, A/B testing, uplift modeling, and counterfactual estimation — built for data scientists and AI engineers.

---

## Features

### Core Causal Inference
- **Unified interface** for DoWhy (identification/refutation) and EconML (heterogeneous effects)
- **Config-driven pipelines**: YAML → identification → estimation → refutation → sensitivity
- **Multiple estimators**: IPW, matching, doubly robust, double ML, causal forest, IV methods

### Sensitivity Analysis
- **Rosenbaum bounds** for binary treatment with matched data
- **Cinelli-Hazlett** robustness values for linear models
- **E-values** (VanderWeele & Ding) for unmeasured confounding
- **TIPS curves** — visualize treatment-effect sensitivity contours

### A/B Testing
- **Frequentist**: t-tests, proportion z-tests, sequential testing (SPRT, mSPRT)
- **Bayesian**: Beta-Binomial, Normal-Normal with ROPE & expected loss
- **Power analysis** & sample size calculation
- **Multiple testing**: Bonferroni, Benjamini-Hochberg FDR

### Uplift Modeling
- **Metalearners**: T/S/X/R/DR-learners, Causal Forest
- **Evaluation**: Qini coefficient, AUUC, gain/decile charts
- **Targeting simulation**: personalize treatment decisions

### Visualization
- **Causal DAGs** with backdoor path highlighting & adjustment sets
- **Forest plots** for meta-analysis/subgroup effects
- **Love plots** for covariate balance diagnostics
- **Sensitivity contours**, **Qini curves**, **counterfactual distributions**

### Counterfactuals
- G-computation, TMLE for individual effect estimation
- Uncertainty quantification via bootstrap

---

## Installation

```bash
pip install causal-toolkit
# Or with optional dependencies
pip install "causal-toolkit[dev,viz,notebooks]"
```

---

## Quick Start

### Python API

```python
from causal_toolkit import CausalModel, DoWhyWrapper, SensitivityAnalyzer
from causal_toolkit.core.base import EstimatorType, IdentificationStrategy

# Load your data
df = pd.read_csv("your_data.csv")

# Define causal model
model = CausalModel(
    data=df,
    treatment="treatment",
    outcome="outcome",
    common_causes=["age", "income", "education"]
)

# Identify and estimate
dowhy = DoWhyWrapper(model)
estimand = dowhy.identify(IdentificationStrategy.BACKDOOR)
estimate = dowhy.estimate(EstimatorType.DOUBLY_ROBUST)

print(f"ATE: {estimate.value:.4f} [{estimate.ci_lower:.4f}, {estimate.ci_upper:.4f}]")

# Sensitivity analysis
analyzer = SensitivityAnalyzer(model)
analyzer.rosenbaum_bounds(estimate)
analyzer.cinelli_hazlett(estimate)
analyzer.e_value(estimate)
print(analyzer.summarize())
```

### CLI

```bash
# Estimate causal effect from config
causal-toolkit estimate --config config.yaml --data data.csv

# Sensitivity analysis
causal-toolkit sensitivity --data data.csv --treatment T --outcome Y --confounders X1 X2

# A/B test
causal-toolkit ab_test --data exp.csv --variant group --outcome converted --control A --treatment B

# Uplift modeling
causal-toolkit uplift --data data.csv --treatment T --outcome Y --covariates X1 X2 --model causal_forest

# Visualize causal graph
causal-toolkit graph --config dag.yaml --render plotly
```

### Configuration (YAML)

```yaml
pipeline:
  identification:
    strategy: backdoor
    adjustment_set: ["age", "income", "education"]
    graph: dag.dot
  estimation:
    method: causal_forest
    params:
      n_estimators: 500
      min_samples_leaf: 10
  refutation:
    - placebo_treatment
    - random_common_cause
    - data_subset
    - simulated_confounder
  sensitivity:
    method: cinelli_hazlett
    benchmark_covariates: ["age", "income"]
```

---

## Demo Notebooks

| Notebook | Dataset | Focus |
|----------|---------|-------|
| `01_ihdp_demo.ipynb` | IHDP (747 obs) | Full pipeline: identification, estimation, refutation, sensitivity, CATE |
| `02_lalonde_psid_demo.ipynb` | Lalonde/PSID (16K obs) | Propensity methods, selection bias, forest plots, covariate balance |
| `03_ab_testing_uplift_demo.ipynb` | Synthetic/Criteo | A/B testing (freq/Bayes/sequential), power, uplift models, targeting |

```bash
jupyter lab examples/notebooks/01_ihdp_demo.ipynb
```

---

## Documentation

Full documentation at: **https://anujsingh-cse.github.io/causal-inference-toolkit/**

### Key Pages
- [Installation & Setup](https://anujsingh-cse.github.io/causal-inference-toolkit/getting-started/installation/)
- [Quick Start](https://anujsingh-cse.github.io/causal-inference-toolkit/getting-started/quickstart/)
- [Core Concepts](https://anujsingh-cse.github.io/causal-inference-toolkit/concepts/causal-model/)
- [API Reference](https://anujsingh-cse.github.io/causal-inference-toolkit/api/core/)

---

## Architecture

```
src/causal_toolkit/
├── core/           # Base types: CausalModel, CausalEstimand, CausalEstimate
├── wrappers/       # DoWhyWrapper, EconMLWrapper, UpliftModeler
├── analysis/       # SensitivityAnalyzer, ABTestAnalyzer
├── visualization/  # ForestPlot, LovePlot, SensitivityPlot, UpliftPlot, CausalGraphVisualizer
├── cli/            # Typer-based command line interface
└── utils/          # Data loading, preprocessing, synthetic data
```

---

## Contributing

1. Fork the repo
2. Create feature branch: `git checkout -b feature/amazing-thing`
3. Install dev deps: `pip install -e ".[dev]"`
4. Run tests: `pytest`
5. Run lint: `ruff check . && mypy src/causal_toolkit`
6. Submit PR

---

## Citation

If you use this toolkit in research, please cite:

```bibtex
@software{causal_toolkit,
  author = {Anuj Singh},
  title = {Causal Inference Toolkit},
  year = {2025},
  url = {https://github.com/anujsingh-cse/causal-inference-toolkit}
}
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Acknowledgments

- **DoWhy** (Microsoft) — causal identification & refutation
- **EconML** (Microsoft) — heterogeneous treatment effects
- **Cinelli & Hazlett (2020)** — sensitivity analysis framework
- **VanderWeele & Ding (2017)** — E-values
- **Rosenbaum (2002)** — sensitivity bounds for matched studies