# Causal Inference Toolkit — SPEC

## Project Overview

**Name:** causal-toolkit  
**Type:** Python package (open-source, MIT)  
**Target:** Data scientists, ML engineers, researchers building causal ML systems  
**Goal:** Production-ready causal inference toolkit wrapping DoWhy/EconML with viz, sensitivity analysis, A/B test analysis, uplift modeling, counterfactuals — demo on public dataset with technical writeup

---

## Core Modules

### 1. `causal_toolkit.core` — Foundations
- **CausalModel** base class: unified interface for identification, estimation, refutation
- **Assumptions** dataclass: unconfoundedness, positivity, consistency, SUTVA
- **IdentificationStrategy** enum: backdoor, frontdoor, IV, mediation
- **EstimatorRegistry**: pluggable estimators (backdoor: linear, ML, matching; IV: 2SLR, DeepIV; etc.)
- **RefutationSuite**: placebo, data subset, random common cause, simulated confounder

### 2. `causal_toolkit.wrappers` — DoWhy/EconML Adapters
- **DoWhyWrapper**: causal graph → identification → estimation → refutation pipeline
- **EconMLWrapper**: CATE estimators (T-learner, S-learner, X-learner, R-learner, DR-learner, CausalForest, Metalearners)
- **UpliftModel**: unified interface for uplift modeling (TwoModel, ClassTransformation, DR)
- **IVEstimator**: 2SLS, DeepIV, OrthoIV
- **Config-driven**: YAML/JSON spec → instantiated pipeline

### 3. `causal_toolkit.analysis` — Analytical Tools
- **SensitivityAnalysis**: 
  - Rosenbaum bounds (binary treatment)
  - Cinelli-Hazlett (continuous treatment)  
  - Robustness Value (RV), R² decompositions
  - E-value computation
- **ABTestAnalyzer**: 
  - Frequentist (t-test, proportion z-test, sequential testing)
  - Bayesian (Beta-Binomial, Normal-Normal, ROPE)
  - Power analysis, sample size calculator
  - Multiple testing correction (Bonferroni, BH, sequential)
- **MediationAnalysis**: natural direct/indirect effects, sequential g-estimation
- **CounterfactualEngine**: individual-level counterfactuals via G-computation, TMLE

### 4. `causal_toolkit.visualization` — Plotting
- **CausalGraphViz**: DAG rendering (graphviz/pygraphviz), do-calculus steps highlighted
- **ForestPlot**: meta-analysis style, subgroup effects, confidence intervals
- **LovePlot**: covariate balance (SMD) pre/post matching/weighting
- **SensitivityCurve**: RV/E-value contours, tornado plots
- **UpliftCurve**: Qini, AUUC, gain/decile plots
- **CounterfactualDist**: individual outcome distributions under treatment/control

### 5. `causal_toolkit.demo` — Portfolio Demos
- **DemoRunner**: CLI + notebooks for each use case
- **Datasets**: 
  - `ihdp` (Infant Health Development Program) — RCT + observational benchmark
  - `lalonde` (NSW/PSID) — classic propensity score demo
  - `criteo-uplift` — large-scale uplift benchmark
  - `online-ab-test` — synthetic A/B test with sequential monitoring
- **Notebooks**: end-to-end analysis with markdown narrative

---

## CLI Interface

```
causal-toolkit estimate --config config.yaml --data data.csv --out results/
causal-toolkit sensitivity --method rosenbaum --gamma 1.5 --data data.csv
causal-toolkit ab-test --variant A --variant B --metric conversion --sequential
causal-toolkit uplift --model causal-forest --data data.csv --plot qini
causal-toolkit graph --dag dag.dot --render png --highlight-backdoor
causal-toolkit counterfactual --unit-id 42 --treatment 1 --data data.csv
```

---

## Configuration Schema (YAML)

```yaml
pipeline:
  identification:
    strategy: backdoor
    adjustment_set: ["age", "income", "education"]
    graph: "dag.dot"
  estimation:
    method: "causal_forest"
    params:
      n_estimators: 500
      min_samples_leaf: 10
  refutation:
    - placebo_treatment
    - random_common_cause
    - data_subset
    - simulated_confounder
  sensitivity:
    method: "cinelli_hazlett"
    benchmark_covariates: ["age", "income"]
```

---

## Acceptance Criteria

| Module | Criteria |
|--------|----------|
| Core | CausalModel instantiates from config; runs identification→estimation→refutation |
| Wrappers | DoWhy/EconML estimators callable via unified `estimate()`; params passthrough |
| Sensitivity | Rosenbaum, Cinelli-Hazlett, E-value compute + plot; benchmarks documented |
| A/B Test | Frequentist + Bayesian; sequential; power calc; multiple comparison correction |
| Uplift | CATE estimators + Qini/AUUC; sklearn-compatible API |
| Counterfactual | Individual predictions with uncertainty intervals |
| Viz | All plots render to matplotlib/plotly; export PNG/HTML/SVG |
| Demo | 3+ notebooks run end-to-end on public data; HTML export for portfolio |
| Tests | >80% coverage; unit + integration; CI passes |
| Docs | mkdocs-material site; API reference; Quickstart; Gallery |

---

## Tech Stack

- **Core**: Python 3.10+, numpy, pandas, scipy, networkx
- **Causal**: dowhy, econml, causalinference
- **Viz**: matplotlib, seaborn, plotly, graphviz, dowhy.plotter
- **Config**: pydantic-settings, yaml
- **CLI**: typer, rich
- **Testing**: pytest, hypothesis, pytest-cov
- **Docs**: mkdocs-material, mkdocstrings
- **CI**: GitHub Actions (lint, type-check, test, build, deploy docs)

---

## Repository Structure

```
causal-toolkit/
├── SPEC.md
├── pyproject.toml
├── README.md
├── LICENSE
├── .github/workflows/ci.yml
├── src/causal_toolkit/
│   ├── __init__.py
│   ├── core/
│   ├── wrappers/
│   ├── analysis/
│   ├── visualization/
│   └── demo/
├── tests/
├── docs/
├── examples/
└── configs/
```

---

## Milestones

1. **M1** — Scaffold + core types + CLI skeleton (Week 1)
2. **M2** — DoWhy/EconML wrappers + config-driven pipeline (Week 2)
3. **M3** — Sensitivity analysis + A/B test analyzer (Week 3)
4. **M4** — Uplift modeling + counterfactuals (Week 4)
5. **M5** — Visualization suite + demo notebooks (Week 5)
6. **M6** — Docs, CI, PyPI publish, portfolio writeup (Week 6)