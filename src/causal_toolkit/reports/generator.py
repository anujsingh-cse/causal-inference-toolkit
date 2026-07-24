"""
Automated Executive Causal Report Generator.

Generates standalone HTML analytical reports compiling causal identification,
estimation, sensitivity analyses, and refutation test results.
"""

from pathlib import Path
from typing import Any


class CausalReportGenerator:
    """
    Generates standalone HTML report summaries for causal inference analyses.
    """

    def __init__(self, title: str = "Executive Causal Analysis Report"):
        self.title = title

    def generate_html(
        self,
        estimate_summary: dict[str, Any],
        sensitivity_results: dict[str, Any] | None = None,
        refutation_results: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """
        Build standalone HTML string with styled cards and key metrics.
        """
        meta = metadata or {}
        dataset_name = meta.get("dataset", "Custom Dataset")
        author = meta.get("author", "causal-toolkit")

        # Extract main estimate
        ate = estimate_summary.get("value", estimate_summary.get("att", 0.0))
        se = estimate_summary.get("se", 0.0)
        p_val = estimate_summary.get("p_value", 0.05)
        ci_low = estimate_summary.get("ci_lower", ate - 1.96 * se)
        ci_high = estimate_summary.get("ci_upper", ate + 1.96 * se)
        method = estimate_summary.get("method", "Backdoor Estimation")

        sig_style = "background:#28a745;color:white;padding:3px 8px;border-radius:4px;font-size:12px;"
        not_sig_style = (
            "background:#dc3545;color:white;padding:3px 8px;border-radius:4px;font-size:12px;"
        )
        p_badge = (
            f'<span style="{sig_style}">Significant</span>'
            if p_val < 0.05
            else f'<span style="{not_sig_style}">Not Significant</span>'
        )

        # Refutations block
        ref_html = ""
        if refutation_results:
            ref_rows = ""
            for test_name, res in refutation_results.items():
                status = "PASSED" if getattr(res, "passed", True) else "FAILED"
                color = "#28a745" if status == "PASSED" else "#dc3545"
                new_val = getattr(res, "new_effect", "N/A")
                if isinstance(new_val, float):
                    new_val = f"{new_val:.4f}"
                ref_rows += f"""
                <tr>
                    <td style="padding:8px;border-bottom:1px solid #eee;">
                        <strong>{test_name}</strong>
                    </td>
                    <td style="padding:8px;border-bottom:1px solid #eee;">{new_val}</td>
                    <td style="padding:8px;border-bottom:1px solid #eee;">
                        <strong style="color:{color};">{status}</strong>
                    </td>
                </tr>
                """
            ref_html = f"""
            <div style="background:white;padding:20px;border-radius:8px;margin-bottom:20px;">
                <h3 style="margin-top:0;color:#333;">Refutation Diagnostics</h3>
                <table style="width:100%;border-collapse:collapse;text-align:left;">
                    <thead>
                        <tr style="background:#f8f9fa;">
                            <th style="padding:8px;border-bottom:2px solid #dee2e6;">Test</th>
                            <th style="padding:8px;border-bottom:2px solid #dee2e6;">Effect</th>
                            <th style="padding:8px;border-bottom:2px solid #dee2e6;">Status</th>
                        </tr>
                    </thead>
                    <tbody>
                        {ref_rows}
                    </tbody>
                </table>
            </div>
            """

        # Sensitivity block
        sens_html = ""
        if sensitivity_results:
            rv = sensitivity_results.get("robustness_value", sensitivity_results.get("rv", "N/A"))
            e_val = sensitivity_results.get("e_value", "N/A")
            if isinstance(rv, float):
                rv = f"{rv * 100:.2f}%"
            if isinstance(e_val, float):
                e_val = f"{e_val:.3f}"

            sens_html = f"""
            <div style="background:white;padding:20px;border-radius:8px;margin-bottom:20px;">
                <h3 style="margin-top:0;color:#333;">Sensitivity to Unobserved Confounding</h3>
                <div style="display:flex;gap:20px;">
                    <div style="flex:1;background:#f8f9fa;padding:15px;border-radius:6px;">
                        <span style="font-size:12px;color:#6c757d;">Robustness Value (RV)</span>
                        <h2 style="margin:5px 0;color:#007bff;">{rv}</h2>
                    </div>
                    <div style="flex:1;background:#f8f9fa;padding:15px;border-radius:6px;">
                        <span style="font-size:12px;color:#6c757d;">E-Value</span>
                        <h2 style="margin:5px 0;color:#17a2b8;">{e_val}</h2>
                    </div>
                </div>
            </div>
            """

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>{self.title}</title>
    <style>
        body {{ font-family: sans-serif; background: #f4f6f9; color: #333; padding: 40px; }}
        .container {{ max-width: 900px; margin: 0 auto; }}
        .header {{ background: #1e3c72; color: white; padding: 30px; border-radius: 8px; }}
        .card {{ background: white; padding: 25px; border-radius: 8px; margin-bottom: 20px; }}
        .grid {{ display: flex; gap: 20px; margin-top: 15px; }}
        .item {{ flex: 1; background: #f8f9fa; padding: 15px; border-radius: 6px; }}
        .val {{ font-size: 24px; font-weight: bold; color: #2c3e50; margin-top: 5px; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1 style="margin:0;">{self.title}</h1>
            <p style="margin:8px 0 0 0;">Dataset: {dataset_name} | Method: {method} | Author: {author}</p>
        </div>

        <div class="card">
            <div style="display:flex;justify-content:space-between;align-items:center;">
                <h3 style="margin:0;">Causal Effect Estimate</h3>
                {p_badge}
            </div>
            <div class="grid">
                <div class="item">
                    <span style="font-size:12px;color:#6c757d;">Point Estimate</span>
                    <div class="val">{ate:.4f}</div>
                </div>
                <div class="item">
                    <span style="font-size:12px;color:#6c757d;">Standard Error</span>
                    <div class="val">{se:.4f}</div>
                </div>
                <div class="item">
                    <span style="font-size:12px;color:#6c757d;">95% CI</span>
                    <div class="val">[{ci_low:.4f}, {ci_high:.4f}]</div>
                </div>
                <div class="item">
                    <span style="font-size:12px;color:#6c757d;">p-value</span>
                    <div class="val">{p_val:.4f}</div>
                </div>
            </div>
        </div>

        {ref_html}
        {sens_html}
    </div>
</body>
</html>
"""
        return html

    def save_report(
        self,
        output_path: str,
        estimate_summary: dict[str, Any],
        sensitivity_results: dict[str, Any] | None = None,
        refutation_results: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Generate and write report to specified file path."""
        html_content = self.generate_html(
            estimate_summary=estimate_summary,
            sensitivity_results=sensitivity_results,
            refutation_results=refutation_results,
            metadata=metadata,
        )
        p = Path(output_path).resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(html_content, encoding="utf-8")
        return str(p)

