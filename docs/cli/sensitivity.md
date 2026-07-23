# CLI: Sensitivity

```bash
causal-toolkit sensitivity --data DATA.csv --treatment T --outcome Y [OPTIONS]
```

## Options

| Option | Short | Description |
|--------|-------|-------------|
| `--data` | `-d` | Input CSV |
| `--treatment` | `-t` | Treatment column |
| `--outcome` | `-y` | Outcome column |
| `--confounders` | `-c` | Confounder columns |
| `--method` | `-m` | `rosenbaum`, `cinelli_hazlett`, `evalue`, `all` (default: `all`) |
| `--estimate` | `-e` | Point estimate (optional, runs estimation if omitted) |
| `--se` | | Standard error (optional) |
| `--out` | `-o` | Output directory (default: `./sensitivity`) |

## Examples

```bash
# Full suite (runs estimation first if --estimate/--se not provided)
causal-toolkit sensitivity -d data.csv -t treatment -y outcome -c age income education

# Just E-value with known estimate
causal-toolkit sensitivity -d data.csv -t treatment -y outcome \
    --estimate 2.5 --se 0.45 -m evalue
```

## Output

```
Rosenbaum bounds: Γ=1.84, conclusion_reversed=True
Cinelli-Hazlett: RV=0.12, R²_yz=0.15, R²_zd=0.08
E-value: 2.45 (CI: 1.67)
  Benchmark 'income': R²_yz=0.08, R²_zd=0.05
  Benchmark 'education': R²_yz=0.03, R²_zd=0.02
[green]Sensitivity results saved to sensitivity/sensitivity.json[/green]
```
