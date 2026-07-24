"""
Causal Toolkit - Analysis Module

Sensitivity analysis, A/B testing, uplift modeling.
"""

from causal_toolkit.analysis.ab_test import (
    ABTestAnalyzer,
    ABTestData,
    ABTestResult,
    ABTestType,
    Alternative,
    TestType,
    evaluate_uplift,
)
from causal_toolkit.analysis.did import DifferenceInDifferences, DiDResult
from causal_toolkit.analysis.sensitivity import (
    SensitivityAnalyzer,
    SensitivityResult,
    run_sensitivity_suite,
)
from causal_toolkit.analysis.synthetic_control import SyntheticControl, SyntheticControlResult

__all__ = [
    "ABTestAnalyzer",
    "ABTestData",
    "ABTestResult",
    "ABTestType",
    "Alternative",
    "DifferenceInDifferences",
    "DiDResult",
    "SensitivityAnalyzer",
    "SensitivityResult",
    "SyntheticControl",
    "SyntheticControlResult",
    "TestType",
    "evaluate_uplift",
    "run_sensitivity_suite",
]

