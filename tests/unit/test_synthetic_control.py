"""
Unit tests for Synthetic Control method.
"""

import numpy as np
import pandas as pd
import pytest

from causal_toolkit.analysis.synthetic_control import SyntheticControl, SyntheticControlResult


@pytest.fixture
def sample_panel_data():
    """Generate synthetic panel data with 1 treated unit and 4 donor units over 10 time periods."""
    np.random.seed(42)
    units = ["A", "B", "C", "D", "E"]
    times = list(range(1, 11))

    rows = []
    for u in units:
        base_trend = np.array(times) * 2.0
        unit_offset = {"A": 5, "B": 2, "C": 8, "D": 4, "E": 6}[u]
        noise = np.random.normal(0, 0.5, len(times))

        y = base_trend + unit_offset + noise
        # Add treatment effect of +10 to unit 'A' from time 6 onwards
        if u == "A":
            y[5:] += 10.0

        for t, val in zip(times, y, strict=False):
            rows.append({"unit": u, "time": t, "outcome": val})

    return pd.DataFrame(rows)


def test_synthetic_control_fit_predict(sample_panel_data):
    sc = SyntheticControl(random_state=42)
    res = sc.fit_predict(
        data=sample_panel_data,
        unit_col="unit",
        time_col="time",
        outcome_col="outcome",
        treated_unit="A",
        treatment_time=6,
        run_placebos=True,
    )

    assert isinstance(res, SyntheticControlResult)
    assert res.treated_unit == "A"
    assert len(res.weights) == 4
    assert pytest.approx(sum(res.weights.values()), 1e-4) == 1.0
    assert res.att > 5.0  # True effect was +10
    assert res.p_value is not None
    assert "B" in res.placebo_results
    assert repr(res).startswith("<SyntheticControlResult")


def test_synthetic_control_missing_unit_raises(sample_panel_data):
    sc = SyntheticControl()
    with pytest.raises(ValueError, match="not in data"):
        sc.fit_predict(
            data=sample_panel_data,
            unit_col="unit",
            time_col="time",
            outcome_col="outcome",
            treated_unit="NONEXISTENT",
            treatment_time=6,
        )
