# Contributing

Thank you for contributing!

## Development Setup

```bash
git clone https://github.com/anujsingh-cse/causal-inference-toolkit.git
cd causal-inference-toolkit
pip install -e ".[dev,viz,notebooks]"
pre-commit install
```

## Running Tests

```bash
# All tests
pytest

# With coverage
pytest --cov=causal_toolkit --cov-report=html

# Specific test file
pytest tests/test_core.py -v
```

## Code Style

```bash
# Format
ruff format src tests

# Lint
ruff check src tests

# Type check
mypy src/causal_toolkit
```

## Pull Request Process

1. Fork and create feature branch: `git checkout -b feature/amazing-thing`
2. Write code + tests
3. Run full check: `ruff check . && mypy src/causal_toolkit && pytest`
4. Submit PR

## Reporting Bugs

- Use GitHub Issues
- Include minimal reproducible example
- Specify Python version, OS, package version

## Feature Requests

- Open Issue with `enhancement` label
- Describe use case and expected API
