"""
Unit tests for Difference-in-Differences (DiD) estimator.
"""

import numpy as np
import pandas as pd
import pytest

from causal_toolkit.analysis.did import DiDResult, DifferenceInDifferences


@pytest.fixture
def sample_did_data():
    """Generate 2x2 DiD data with known true ATT = 3.0."""
    np.random.seed(42)
    n = 400
    treatment = np.random.binomial(1, 0.5, size=n)
    post = np.random.binomial(1, 0.5, size=n)
    true_att = 3.0

    noise = np.random.normal(0, 1.0, n)
    y = 10.0 + 2.0 * treatment + 1.5 * post + true_att * (treatment * post) + noise
    return pd.DataFrame({"outcome": y, "treatment": treatment, "post": post})


def test_did_2x2_estimation(sample_did_data):
    did = DifferenceInDifferences()
    res = did.estimate_2x2(
        data=sample_did_data,
        outcome_col="outcome",
        treatment_col="treatment",
        post_col="post",
    )

    assert isinstance(res, DiDResult)
    assert res.att == pytest.approx(3.0, abs=0.5)
    assert res.p_value < 0.05
    assert res.ci_lower < res.att < res.ci_upper
    assert "Difference-in-Differences" in res.summary()
    assert repr(res).startswith("<DiDResult")


def test_did_fit_panel(sample_did_data):
    # Add dummy unit and time columns
    units = ["unit1", "unit2", "unit3"]
    sample_did_data["unit"] = np.random.choice(units, size=len(sample_did_data))
    sample_did_data["time"] = np.where(sample_did_data["post"] == 1, 2021, 2020)

    did = DifferenceInDifferences()
    res = did.fit_panel(
        data=sample_did_data,
        unit_col="unit",
        time_col="time",
        outcome_col="outcome",
        treatment_col="treatment",
        treatment_start_time=2021,
    )

    assert isinstance(res, DiDResult)
    assert res.att == pytest.approx(3.0, abs=0.5)
    assert res.parallel_trends_pvalue is not None

