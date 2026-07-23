"""
Causal Inference Toolkit
========================

Open-source Python package for causal inference with DoWhy/EconML wrappers,
sensitivity analysis, causal graphs, A/B test analysis, uplift modeling,
and counterfactual estimation.

Target: Data Scientist / AI Engineer portfolio project.
"""

from causal_toolkit.core.base import CausalModel, CausalEstimand, CausalEstimate
from causal_toolkit.wrappers import DoWhyWrapper, EconMLWrapper, UpliftModeler
from causal_toolkit.analysis.sensitivity import SensitivityAnalyzer
from causal_toolkit.analysis.ab_test import ABTestAnalyzer
from causal_toolkit.visualization.graphs import CausalGraphVisualizer
from causal_toolkit.visualization.plots import ForestPlot, SensitivityPlot

__version__ = "0.1.0"
__author__ = "Anuj Singh"
__license__ = "MIT"

__all__ = [
    "CausalModel",
    "CausalEstimand",
    "CausalEstimate",
    "DoWhyWrapper",
    "EconMLWrapper",
    "UpliftModeler",
    "SensitivityAnalyzer",
    "ABTestAnalyzer",
    "CausalGraphVisualizer",
    "ForestPlot",
    "SensitivityPlot",
]