"""
Causal Inference Toolkit
========================

Open-source Python package for causal inference with DoWhy/EconML wrappers,
sensitivity analysis, causal graphs, A/B test analysis, uplift modeling,
and counterfactual estimation.

Target: Data Scientist / AI Engineer portfolio project.
"""

from causal_toolkit.analysis.ab_test import ABTestAnalyzer
from causal_toolkit.analysis.did import DifferenceInDifferences
from causal_toolkit.analysis.sensitivity import SensitivityAnalyzer
from causal_toolkit.analysis.synthetic_control import SyntheticControl
from causal_toolkit.core.base import CausalEstimand, CausalEstimate, CausalModel
from causal_toolkit.reports.generator import CausalReportGenerator
from causal_toolkit.visualization.graphs import CausalGraphVisualizer
from causal_toolkit.visualization.plots import ForestPlot, SensitivityPlot
from causal_toolkit.wrappers import DoWhyWrapper, EconMLWrapper, UpliftModeler

__version__ = "0.1.0"
__author__ = "Anuj Singh"
__license__ = "MIT"

__all__ = [
    "ABTestAnalyzer",
    "CausalEstimand",
    "CausalEstimate",
    "CausalGraphVisualizer",
    "CausalModel",
    "CausalReportGenerator",
    "DifferenceInDifferences",
    "DoWhyWrapper",
    "EconMLWrapper",
    "ForestPlot",
    "SensitivityAnalyzer",
    "SensitivityPlot",
    "SyntheticControl",
    "UpliftModeler",
]
