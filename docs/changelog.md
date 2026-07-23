# Changelog

All notable changes to this project.

## [0.1.0] - 2025-01-XX

### Added
- Core causal inference types: `CausalModel`, `CausalEstimand`, `CausalEstimate`, `Assumptions`
- DoWhy wrapper: identification (backdoor, frontdoor, IV, mediation), estimation, refutation
- EconML wrapper: T/S/X/R/DR-learners, Causal Forest, CATE/ATE estimation
- UpliftModeler: causal_forest, two_model, class_transformation, dr_learner methods
- Sensitivity analysis: Rosenbaum bounds, Cinelli-Hazlett RV, E-values, TIPS curves
- A/B testing: proportion z-test, t-test, SPRT/mSPRT, Bayesian Beta-Binomial, power analysis, multiple testing
- Visualization: causal DAGs, forest plots, love plots, sensitivity contours, Qini curves, counterfactual distributions
- CLI: estimate, sensitivity, ab_test, uplift, graph, counterfactual, power, demo commands
- Data utilities: IHDP, Lalonde, Criteo uplift datasets, synthetic data generation, propensity scoring, SMD, IPW, bootstrap CI
- Configuration-driven pipelines via YAML
- MkDocs Material documentation site
- GitHub Actions CI pipeline (lint, type-check, test, build, deploy docs)

### Changed
- N/A

### Deprecated
- N/A

### Removed
- N/A

### Fixed
- N/A

### Security
- N/A
