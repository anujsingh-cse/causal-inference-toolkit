"""
CLI tests for Causal Inference Toolkit.

Tests CLI commands via typer.testing.CliRunner without spawning subprocesses.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from typer.testing import CliRunner

from causal_toolkit.cli import app

runner = CliRunner()


@pytest.fixture
def tmp_dir(tmp_path):
    """Provide temp directory for test outputs."""
    return tmp_path


@pytest.fixture
def synthetic_csv(tmp_dir):
    """Create a simple synthetic CSV for CLI tests."""
    np.random.seed(42)
    n = 200
    X = np.random.randn(n, 3)
    logit_p = X[:, 0] + 0.5 * X[:, 1]
    treatment = np.random.binomial(1, 1 / (1 + np.exp(-logit_p)))
    outcome = 10 + X[:, 0] + 2 * treatment + np.random.randn(n)

    df = pd.DataFrame(
        {
            "x0": X[:, 0],
            "x1": X[:, 1],
            "x2": X[:, 2],
            "treatment": treatment,
            "outcome": outcome,
        }
    )
    path = tmp_dir / "data.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def config_yaml(tmp_dir):
    """Create minimal config YAML."""
    cfg = {
        "pipeline": {
            "identification": {
                "strategy": "backdoor",
                "treatment": "treatment",
                "outcome": "outcome",
                "adjustment_set": ["x0", "x1", "x2"],
            },
            "estimation": {"method": "linear_regression"},
        }
    }
    path = tmp_dir / "config.yaml"
    with open(path, "w") as f:
        yaml.dump(cfg, f)
    return path


@pytest.fixture
def ab_csv(tmp_dir):
    """Create A/B test CSV."""
    np.random.seed(42)
    n = 500
    variant = np.random.choice(["control", "treatment"], n)
    conversion = np.where(
        variant == "treatment",
        np.random.binomial(1, 0.12, n),
        np.random.binomial(1, 0.10, n),
    )
    df = pd.DataFrame({"variant": variant, "conversion": conversion})
    path = tmp_dir / "ab_data.csv"
    df.to_csv(path, index=False)
    return path


class TestPowerCommand:
    """Test power analysis CLI command."""

    def test_power_basic(self):
        result = runner.invoke(
            app,
            ["power", "--baseline", "0.10", "--mde", "0.05", "--alpha", "0.05", "--power", "0.8"],
        )
        assert result.exit_code == 0
        assert "Required sample per variant" in result.output

    def test_power_custom_ratio(self):
        result = runner.invoke(
            app,
            [
                "power",
                "--baseline", "0.15",
                "--mde", "0.10",
                "--alpha", "0.01",
                "--power", "0.9",
                "--ratio", "2.0",
            ],
        )
        assert result.exit_code == 0
        assert "Total sample size" in result.output


class TestDemoCommand:
    """Test demo CLI command."""

    def test_demo_synthetic(self, tmp_dir):
        result = runner.invoke(app, ["demo", "--dataset", "synthetic", "--out", str(tmp_dir / "demo")])
        assert result.exit_code == 0
        assert "Demo completed" in result.output
        assert (tmp_dir / "demo" / "synthetic.csv").exists()

    def test_demo_ihdp(self, tmp_dir):
        result = runner.invoke(app, ["demo", "--dataset", "ihdp", "--out", str(tmp_dir / "demo_ihdp")])
        assert result.exit_code == 0
        assert "Demo completed" in result.output


class TestCounterfactualCommand:
    """Test counterfactual command (should exit with message)."""

    def test_counterfactual_not_implemented(self, synthetic_csv, tmp_dir):
        result = runner.invoke(
            app,
            [
                "counterfactual",
                "--data", str(synthetic_csv),
                "--unit", "0",
                "--treatment", "1.0",
                "--outcome", "outcome",
                "--treatment-col", "treatment",
                "--covariates", "x0",
                "--covariates", "x1",
                "--out", str(tmp_dir / "cf"),
            ],
        )
        assert result.exit_code == 1
        assert "Not yet implemented" in result.output


class TestGraphCommand:
    """Test graph CLI command."""

    def test_graph_missing_fields(self, tmp_dir):
        """Config missing treatment/outcome should error."""
        cfg = {"edges": []}
        cfg_path = tmp_dir / "bad_graph.yaml"
        with open(cfg_path, "w") as f:
            yaml.dump(cfg, f)

        result = runner.invoke(
            app,
            ["graph", "--config", str(cfg_path), "--out", str(tmp_dir / "graph_out")],
        )
        assert result.exit_code == 1


class TestCLIHelp:
    """Test CLI help output."""

    def test_main_help(self):
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "causal-toolkit" in result.output.lower() or "estimate" in result.output.lower()

    def test_estimate_help(self):
        result = runner.invoke(app, ["estimate", "--help"])
        assert result.exit_code == 0
        assert "config" in result.output.lower()

    def test_sensitivity_help(self):
        result = runner.invoke(app, ["sensitivity", "--help"])
        assert result.exit_code == 0

    def test_ab_test_help(self):
        result = runner.invoke(app, ["ab-test", "--help"])
        assert result.exit_code == 0

    def test_uplift_help(self):
        result = runner.invoke(app, ["uplift", "--help"])
        assert result.exit_code == 0

    def test_power_help(self):
        result = runner.invoke(app, ["power", "--help"])
        assert result.exit_code == 0

    def test_demo_help(self):
        result = runner.invoke(app, ["demo", "--help"])
        assert result.exit_code == 0
