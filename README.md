# Causal Inference Toolkit

[![PyPI version](https://img.shields.io/pypi/v/causal-toolkit.svg)](https://pypi.org/project/causal-toolkit/)
[![Python versions](https://img.shields.io/pypi/pyversions/causal-toolkit.svg)](https://pypi.org/project/causal-toolkit/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Build Status](https://github.com/anujsingh-cse/causal-inference-toolkit/workflows/CI/badge.svg)](https://github.com/anujsingh-cse/causal-inference-toolkit/actions)
[![Documentation](https://img.shields.io/badge/docs-latest-blue.svg)](https://anujsingh-cse.github.io/causal-inference-toolkit/)

**Production-ready causal inference toolkit** wrapping DoWhy/EconML with sensitivity analysis, quasi-experiments (Synthetic Control & DiD), A/B testing, uplift modeling, interactive web UI, and executive HTML report generation — built for data scientists and AI engineers.

---

## Features

### Core Causal Inference
- **Unified interface** for DoWhy (identification/refutation) and EconML (heterogeneous effects)
- **Config-driven pipelines**: YAML → identification → estimation → refutation → sensitivity
- **Multiple estimators**: IPW, matching, doubly robust, double ML, causal forest, IV methods

### Quasi-Experiments & Panel Data
- **Synthetic Control Method (SCM)**: SLSQP weight optimization & placebo permutation tests
- **Difference-in-Differences (DiD)**: 2x2 OLS, Two-Way Fixed Effects (TWFE) panel models, parallel trends testing, and dynamic event study trajectories

### Sensitivity Analysis
- **Rosenbaum bounds** for binary treatment with matched data
- **Cinelli-Hazlett** robustness values for linear models
- **E-values** (VanderWeele & Ding) for unmeasured confounding
- **TIPS curves** — visualize treatment-effect sensitivity contours

### Interactive Web Dashboard & Executive Reporting
- **Streamlit Web UI**: Zero-code interactive dashboard (`causal-toolkit app`)
- **Executive HTML Reports**: Standalone report generation (`causal-toolkit report` / `CausalReportGenerator`)

### A/B Testing & Uplift Modeling
- **Frequentist & Bayesian A/B**: t-tests, z-tests, SPRT, mSPRT, Beta-Binomial, Normal-Normal
- **Uplift Metalearners**: T/S/X/R/DR-learners, Causal Forest, Qini & AUUC metrics

---

## Installation

```bash
pip install causal-toolkit

# Or with web dashboard optional dependencies
pip install "causal-toolkit[app,dev,viz]"
```

---

## Interactive Web Dashboard

Launch the zero-code interactive dashboard with a single command:

```bash
causal-toolkit app
```

*(Or run directly via Streamlit: `streamlit run src/causal_toolkit/demo/app.py`)*

The dashboard provides 5 interactive tabs:
1. 📊 **Data Explorer**: Dataset selection (`ihdp`, `lalonde`, `synthetic`) or CSV upload.
2. 🎯 **Causal Estimation**: Interactive backdoor identification & estimation.
3. 🔍 **Sensitivity Analysis**: Robustness Value & E-value computations.
4. 📈 **Quasi-Experiments**: Synthetic Control & Difference-in-Differences panel analysis.
5. 📄 **Executive Report Export**: One-click standalone HTML report download.

---

## Quick Start

### Python API

```python
from causal_toolkit import CausalModel, DoWhyWrapper, SensitivityAnalyzer, DifferenceInDifferences, CausalReportGenerator
from causal_toolkit.core.base import EstimatorType, IdentificationStrategy

# 1. Core Estimation
df = pd.read_csv("your_data.csv")
model = CausalModel(data=df, treatment="treatment", outcome="outcome", common_causes=["age", "income"])
dowhy = DoWhyWrapper(model)
estimand = dowhy.identify(IdentificationStrategy.BACKDOOR)
estimate = dowhy.estimate(EstimatorType.DOUBLY_ROBUST)
print(f"ATE: {estimate.value:.4f}")

# 2. Quasi-Experiments (Difference-in-Differences)
did = DifferenceInDifferences()
did_res = did.estimate_2x2(df, outcome_col="y", treatment_col="t", post_col="post")
print(did_res.summary())

# 3. Sensitivity Analysis
analyzer = SensitivityAnalyzer(model)
analyzer.cinelli_hazlett(estimate)
analyzer.e_value(estimate)
print(analyzer.summarize())

# 4. Generate Executive HTML Report
gen = CausalReportGenerator(title="Executive Causal Analysis")
gen.save_report("report.html", estimate_summary={"value": estimate.value, "ci_lower": estimate.ci_lower, "ci_upper": estimate.ci_upper})
```

### Command-Line Interface (CLI)

```bash
# Launch Streamlit web app
causal-toolkit app

# Estimate causal effect from config
causal-toolkit estimate --config config.yaml --data data.csv

# Run Synthetic Control analysis
causal-toolkit synthetic-control --data panel.csv --unit region --time year --outcome sales --treated-unit "West" --treatment-time 2020

# Run Difference-in-Differences estimation
causal-toolkit did --data panel.csv --outcome sales --treatment treated --post post_period

# Generate Executive HTML report
causal-toolkit report --data data.csv --treatment T --outcome Y --out executive_report.html
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

---

## Architecture

```
src/causal_toolkit/
├── core/           # Base types: CausalModel, CausalEstimand, CausalEstimate
├── wrappers/       # DoWhyWrapper, EconMLWrapper, UpliftModeler
├── analysis/       # SensitivityAnalyzer, ABTestAnalyzer, SyntheticControl, DifferenceInDifferences
├── reports/        # CausalReportGenerator (HTML executive reporting)
├── demo/           # Streamlit interactive web application dashboard
├── visualization/  # ForestPlot, LovePlot, SensitivityPlot, UpliftPlot, CausalGraphVisualizer
├── cli/            # Typer-based command line interface
└── utils/          # Data loading, preprocessing, synthetic data
```

---

## Contributing

1. Fork the repo
2. Create feature branch: `git checkout -b feature/amazing-thing`
3. Install dev deps: `pip install -e ".[app,dev]"`
4. Run tests: `pytest`
5. Run lint: `ruff check . && mypy src/`
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