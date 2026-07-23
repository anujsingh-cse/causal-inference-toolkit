# CLI: Graph

```bash
causal-toolkit graph --config DAG.yaml [OPTIONS]
```

## Config Format (dag.yaml)

```yaml
edges:
  - ["age", "treatment"]
  - ["income", "treatment"]
  - ["age", "outcome"]
  - ["income", "outcome"]
  - ["treatment", "outcome"]
treatment: "treatment"
outcome: "outcome"
common_causes: ["age", "income"]
instruments: []
mediators: []
```

## Options

| Option | Short | Description |
|--------|-------|-------------|
| `--render` | `-r` | `matplotlib`, `plotly`, `graphviz` (default: `matplotlib`) |
| `--backdoor/--no-backdoor` | | Highlight backdoor paths (default: true) |
| `--out` | `-o` | Output file (without extension) |

## Examples

```bash
# Matplotlib PNG
causal-toolkit graph -c dag.yaml -r matplotlib -o mydag

# Interactive HTML
causal-toolkit graph -c dag.yaml -r plotly -o mydag

# Graphviz (high quality)
causal-toolkit graph -c dag.yaml -r graphviz -o mydag
```

## Output

```
Found 2 backdoor path(s):
['treatment', 'age', 'outcome']
['treatment', 'income', 'outcome']

Minimal adjustment sets:
[['age', 'income']]

Do-calculus steps:
Target: P(Y|do(T))
Step 1: Backdoor criterion satisfied with adjustment sets:
  Set 1: {age, income}
Step 2: Apply backdoor adjustment: P(Y|do(T)) = Σ_{C∈Adj} P(Y|T,C)P(C)
```
