# CLI: Estimate

```bash
causal-toolkit estimate --config CONFIG.yaml --data DATA.csv [OPTIONS]
```

## Options

| Option | Short | Description |
|--------|-------|-------------|
| `--config` | `-c` | YAML config file |
| `--data` | `-d` | Input CSV |
| `--out` | `-o` | Output directory (default: `./results`) |
| `--treatment` | `-t` | Override treatment column |
| `--outcome` | `-y` | Override outcome column |
| `--confounders` | | Override confounder columns |
| `--estimator` | `-e` | Estimator type (default: `linear_regression`) |

## Estimators

`linear_regression`, `propensity_score_matching`, `propensity_score_weighting`, `propensity_score_stratification`, `doubly_robust`, `targeted_maximum_likelihood`, `causal_forest`, `double_ml`, `two_stage_ls`, `deepiv`, `orthoiv`

## Example

```bash
causal-toolkit estimate -c config.yaml -d data.csv -o results/
```

## Output

```
Loading config from config.yaml...
Loading data from data.csv...
Treatment: treatment, Outcome: outcome, Confounders: ['age', 'income', 'education']
Identified estimand: ATE: E[E[Y|T=t, X] - E[Y|T=t', X]]
Estimate: 2.3456 [1.8923, 2.7989]
Refutation: placebo_treatment: NOT REJECTED (p=0.6234)
Refutation: random_common_cause: NOT REJECTED (p=0.4123)
Refutation: data_subset: NOT REJECTED (p=0.7845)
[green]Results saved to results/results.json[/green]
```
