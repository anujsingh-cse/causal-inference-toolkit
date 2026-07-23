# CLI: A/B Test

```bash
causal-toolkit ab_test --data DATA.csv --variant VAR --outcome OUT --control A --treatment B [OPTIONS]
```

## Options

| Option | Short | Required | Description |
|--------|-------|----------|-------------|
| `--data` | `-d` | Yes | Input CSV |
| `--variant` | `-v` | Yes | Variant column name |
| `--outcome` | `-y` | Yes | Outcome column name |
| `--control` | | Yes | Control variant value |
| `--treatment` | | Yes | Treatment variant value |
| `--method` | `-m` | No | `frequentist`, `bayesian`, `sequential` (default: `frequentist`) |
| `--type` | | No | `proportion`, `mean` (default: `proportion`) |
| `--alpha` | | No | Significance level (default: `0.05`) |
| `--out` | `-o` | No | Output directory (default: `./ab_test`) |

## Examples

```bash
# Frequentist proportion test
causal-toolkit ab_test -d exp.csv -v group -y converted --control A --treatment B

# Bayesian with ROPE
causal-toolkit ab_test -d exp.csv -v group -y converted --control A --treatment B \
    --method bayesian --rope-width 0.01

# Sequential testing (SPRT)
causal-toolkit ab_test -d exp.csv -v group -y converted --control A --treatment B \
    --method sequential --mde 0.05

# Continuous outcome (t-test)
causal-toolkit ab_test -d exp.csv -v group -y revenue --control A --treatment B \
    --type mean
```

## Output

```json
{
  "variant_a": "A",
  "variant_b": "B",
  "method": "bayesian",
  "test_type": "proportion",
  "estimate_a": 0.125,
  "estimate_b": 0.142,
  "difference": 0.017,
  "relative_difference": 0.136,
  "p_value": 0.0234,
  "ci_lower": 0.002,
  "ci_upper": 0.032,
  "confidence_level": 0.95,
  "n_a": 10000,
  "n_b": 10000,
  "prob_b_better": 0.943,
  "rope_probability": 0.087,
  "expected_loss_a": 0.0012,
  "expected_loss_b": 0.0003
}
```
