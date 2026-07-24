# Installation

## Quick Install

```bash
pip install causal-toolkit
```

## Optional Dependencies

```bash
# Development tools (pytest, ruff, mypy, mkdocs, etc.)
pip install "causal-toolkit[dev]"

# Visualization extras (pygraphviz for hierarchical layouts)
pip install "causal-toolkit[viz]"

# Notebook support (jupyter, nbconvert, ipykernel)
pip install "causal-toolkit[notebooks]"

# All extras
pip install "causal-toolkit[dev,viz,notebooks]"
```

## Development Install

```bash
git clone https://github.com/anujsingh-cse/causal-inference-toolkit.git
cd causal-inference-toolkit
pip install -e ".[dev,viz,notebooks]"
```

## Requirements

- Python 3.10+
- numpy >= 1.24
- pandas >= 2.0
- scipy >= 1.10
- networkx >= 3.1
- dowhy >= 0.9
- econml >= 0.15
- causalinference >= 0.1
- matplotlib >= 3.7
- seaborn >= 0.12
- plotly >= 5.15
- graphviz >= 0.20
- pydantic >= 2.5
- pydantic-settings >= 2.1
- pyyaml >= 6.0
- typer >= 0.9
- rich >= 13.5
- scikit-learn >= 1.3
- statsmodels >= 0.14

## System Dependencies

### Graphviz (for DAG rendering)

```bash
# Ubuntu/Debian
sudo apt-get install graphviz graphviz-dev

# macOS
brew install graphviz

# Windows
# Download from https://graphviz.org/download/
# Add to PATH
```

### PyGraphviz (optional, for hierarchical layouts)

```bash
# Ubuntu/Debian
sudo apt-get install graphviz libgraphviz-dev pkg-config
pip install pygraphviz

# macOS
brew install graphviz
pip install pygraphviz

# Windows
# May require Microsoft Visual C++ Build Tools
pip install pygraphviz
```

## Verify Installation

```python
import causal_toolkit as ct
print(ct.__version__)  # 0.1.0

from causal_toolkit import CausalModel, DoWhyWrapper, SensitivityAnalyzer
print("Core imports work!")
```
