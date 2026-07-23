# CLI: Uplift

```bash
causal-toolkit uplift --data DATA.csv --treatment T --outcome Y --covariates X1 X2 [OPTIONS]
```

## Options

| Option | Short | Required | Description |
|--------|-------|----------|-------------|
| `--data` | `-d` | Yes | Input CSV |
| `--treatment` | `-t` | Yes | Treatment column |
| `--outcome` | `-y` | Yes | Outcome column |
| `--covariates` | `-c` | Yes | Covariate columns (space-separated) |
| `--model` | `-m` | No | `causal_forest`, `two_model`, `dr_learner`, `class_transformation` (default: `causal_forest`) |
| `--plot` | `-p` | No | `qini`, `gain` (default: `qini`) |
| `--out` | `-o` | No | Output directory (default: `./uplift`) |

## Examples

```bash
# Causal Forest (recommended)
causal-toolkit uplift -d data.csv -t treatment -y outcome -c age income education \
    -m causal_forest

# DR-Learner
causal-toolkit uplift -d data.csv -t treatment -y outcome -c age income education \
    -m dr_learner --plot gain

# With custom params (via config)
causal-toolkit uplift -d data.csv -t treatment -y outcome -c age income education \
    -m causal_forest --n_estimators 1000 --min_samples_leaf 20
```

## Output

```
Uplift Metrics: {'qini': 0.342, 'auuc': 0.187, 'gain_at_10pct': 2.45, 'gain_at_50pct': 1.12}
```

Files:
- `uplift/uplift_metrics.json`
- `uplift/qini_curve.png` or `gain_curve.png`
