# Configuration

YAML-driven pipelines for reproducible causal analysis.

## Schema

```yaml
pipeline:
  identification:
    strategy: backdoor          # backdoor, frontdoor, iv, mediation
    treatment: "treatment"       # Required: treatment column
    outcome: "outcome"           # Required: outcome column
    adjustment_set:             # Required for backdoor
      - "age"
      - "income"
      - "education"
    graph: "dag.dot"            # Optional: DOT graph file
    instrumental_variables:     # For IV strategy
      - "distance_to_hospital"
    mediators:                  # For mediation
      - "compliance"

  estimation:
    method: "doubly_robust"     # linear_regression, propensity_score_matching,
                                # propensity_score_weighting, doubly_robust,
                                # causal_forest, tmle, 2sls, deepiv
    params:                     # Estimator-specific parameters
      n_estimators: 500
      min_samples_leaf: 10
      max_depth: 5

  refutation:                   # List of refutation tests
    - placebo_treatment
    - placebo_outcome
    - random_common_cause
    - data_subset
    - simulated_confounder
    - add_unobserved_confounder

  sensitivity:
    method: "cinelli_hazlett"   # rosenbaum, cinelli_hazlett, evalue, tips
    benchmark_covariates:       # For Cinelli-Hazlett benchmarking
      - "age"
      - "income"
    gamma_range: [1.0, 3.0]     # For Rosenbaum bounds
```

## Full Example

```yaml
pipeline:
  identification:
    strategy: backdoor
    treatment: "treatment"
    outcome: "outcome"
    adjustment_set:
      - "age"
      - "sex"
      - "education"
      - "income"
      - "pre_existing_condition"
    graph: "graphs/medical_dag.dot"

  estimation:
    method: "causal_forest"
    params:
      n_estimators: 1000
      min_samples_leaf: 20
      max_depth: 10
      random_state: 42

  refutation:
    - placebo_treatment
    - random_common_cause
    - data_subset
    - simulated_confounder

  sensitivity:
    method: "cinelli_hazlett"
    benchmark_covariates:
      - "income"
      - "education"
```

## CLI Usage

```bash
# Run full pipeline
causal-toolkit estimate --config config.yaml --data data.csv --out results/

# Override specific params
causal-toolkit estimate --config config.yaml --data data.csv \
    --treatment treatment --outcome outcome --confounders age income education \
    --estimator causal_forest
```

## Estimator Parameters

| Estimator | Key Parameters |
|-----------|----------------|
| `linear_regression` | None |
| `propensity_score_matching` | `n_neighbors`, `caliper`, `replace` |
| `propensity_score_weighting` | `stabilized`, `trim_quantiles` |
| `doubly_robust` | `propensity_model`, `outcome_model` |
| `causal_forest` | `n_estimators`, `min_samples_leaf`, `max_depth`, `honest` |
| `tmle` | `initial_estimator`, `fluctuation_param` |
| `2sls` | None |
| `t_learner` / `s_learner` | `models` (list of sklearn estimators) |
| `x_learner` / `r_learner` / `dr_learner` | `models`, `propensity_model` |
