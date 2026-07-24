"""
Unit tests for CausalReportGenerator.
"""

import tempfile
from pathlib import Path

from causal_toolkit.reports.generator import CausalReportGenerator


def test_causal_report_generator_html():
    gen = CausalReportGenerator(title="Test Causal Report")
    estimate_summary = {
        "value": 2.5,
        "se": 0.4,
        "ci_lower": 1.7,
        "ci_upper": 3.3,
        "p_value": 0.001,
        "method": "Linear Regression",
    }
    sensitivity_results = {"robustness_value": 0.18, "e_value": 2.1}

    html = gen.generate_html(
        estimate_summary=estimate_summary,
        sensitivity_results=sensitivity_results,
        metadata={"dataset": "IHDP", "model_name": "DoWhyWrapper"},
    )

    assert "<title>Test Causal Report</title>" in html
    assert "2.5000" in html
    assert "1.7000" in html
    assert "18.00%" in html
    assert "Significant" in html


def test_causal_report_generator_save_file():
    gen = CausalReportGenerator()
    estimate_summary = {"value": 1.2, "se": 0.1, "p_value": 0.02}

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "report.html"
        saved_path = gen.save_report(str(out_path), estimate_summary=estimate_summary)

        p = Path(saved_path)
        assert p.exists()
        content = p.read_text(encoding="utf-8")
        assert "1.2000" in content

