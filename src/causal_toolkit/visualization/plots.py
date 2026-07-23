"""
Visualization Module

Provides plotting utilities for causal inference:
- Causal graphs (DAGs)
- Forest plots for meta-analysis
- Love plots for covariate balance
- Sensitivity curves
- Uplift/Qini curves
- Counterfactual distributions
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
matplotlib.rcParams['figure.dpi'] = 100


class CausalGraphVisualizer:
    """Visualize causal graphs (DAGs) with DoWhy integration."""

    def __init__(self, figsize: Tuple[int, int] = (10, 8)):
        self.figsize = figsize

    def plot_dag(
        self,
        graph: Any,  # networkx.DiGraph or DoWhy graph
        treatment: str,
        outcome: str,
        highlight_backdoor: bool = True,
        highlight_iv: bool = False,
        layout: str = "spring",
        **kwargs,
    ) -> plt.Figure:
        """
        Plot DAG with treatment/outcome highlighted.

        Args:
            graph: networkx.DiGraph or DoWhy CausalModel graph
            treatment: Treatment variable name
            outcome: Outcome variable name
            highlight_backdoor: Highlight backdoor path variables
            highlight_iv: Highlight instrumental variables
            layout: 'spring', 'circular', 'kamada_kawai', 'planar'
        """
        import networkx as nx

        if hasattr(graph, 'graph'):
            G = graph.graph
        else:
            G = graph

        fig, ax = plt.subplots(figsize=self.figsize)

        # Layout
        if layout == "spring":
            pos = nx.spring_layout(G, k=2, iterations=50)
        elif layout == "circular":
            pos = nx.circular_layout(G)
        elif layout == "kamada_kawai":
            pos = nx.kamada_kawai_layout(G)
        elif layout == "planar":
            try:
                pos = nx.planar_layout(G)
            except:
                pos = nx.spring_layout(G)
        else:
            pos = nx.spring_layout(G)

        # Node colors
        node_colors = []
        for node in G.nodes():
            if node == treatment:
                node_colors.append('#e74c3c')  # Red for treatment
            elif node == outcome:
                node_colors.append('#2ecc71')  # Green for outcome
            elif highlight_backdoor and node in self._get_backdoor_nodes(G, treatment, outcome):
                node_colors.append('#f39c12')  # Orange for confounders
            elif highlight_iv and node in self._get_iv_nodes(G, treatment):
                node_colors.append('#9b59b6')  # Purple for IVs
            else:
                node_colors.append('#3498db')  # Blue for others

        # Draw
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=800, ax=ax)
        nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True, arrowsize=20, ax=ax)
        nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold', ax=ax)

        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#e74c3c', label='Treatment'),
            Patch(facecolor='#2ecc71', label='Outcome'),
        ]
        if highlight_backdoor:
            legend_elements.append(Patch(facecolor='#f39c12', label='Confounder (backdoor)'))
        if highlight_iv:
            legend_elements.append(Patch(facecolor='#9b59b6', label='Instrumental Variable'))
        legend_elements.append(Patch(facecolor='#3498db', label='Other'))
        ax.legend(handles=legend_elements, loc='upper right')

        ax.set_title(f"Causal DAG: {treatment} → {outcome}")
        ax.axis('off')
        plt.tight_layout()
        return fig

    def _get_backdoor_nodes(self, G, treatment, outcome):
        """Get nodes on backdoor paths."""
        import networkx as nx
        try:
            # All ancestors of treatment that are also ancestors of outcome
            treatment_ancestors = set(nx.ancestors(G, treatment))
            outcome_ancestors = set(nx.ancestors(G, outcome))
            return treatment_ancestors & outcome_ancestors
        except:
            return set()

    def _get_iv_nodes(self, G, treatment):
        """Get instrumental variable candidates."""
        import networkx as nx
        try:
            # Nodes that affect treatment but not outcome (except through treatment)
            treatment_ancestors = set(nx.ancestors(G, treatment))
            outcome_ancestors = set(nx.ancestors(G, treatment)) | {treatment}
            return treatment_ancestors - outcome_ancestors
        except:
            return set()

    def plot_do_calculus_steps(
        self,
        steps: List[Dict],
        treatment: str,
        outcome: str,
    ) -> plt.Figure:
        """Visualize do-calculus identification steps."""
        n = len(steps)
        fig, axes = plt.subplots(1, n, figsize=(6*n, 5))
        if n == 1:
            axes = [axes]

        for i, step in enumerate(steps):
            G = step.get('graph')
            rule = step.get('rule', f'Step {i+1}')
            self._plot_step_on_ax(axes[i], G, treatment, outcome, rule)

        plt.tight_layout()
        return fig

    def _plot_step_on_ax(self, ax, G, treatment, outcome, title):
        import networkx as nx
        pos = nx.spring_layout(G, k=1.5)
        nx.draw(G, pos, ax=ax, with_labels=True, node_color='lightblue',
                node_size=500, font_size=8, arrows=True)
        ax.set_title(title)


class ForestPlot:
    """Forest plot for meta-analysis and subgroup effects."""

    def __init__(self, figsize: Tuple[int, int] = (10, 6)):
        self.figsize = figsize

    def plot(
        self,
        estimates: List[Dict],
        xlabel: str = "Effect Size",
        title: str = "Forest Plot",
        show_ci: bool = True,
        log_scale: bool = False,
        **kwargs,
    ) -> plt.Figure:
        """
        Plot forest plot.

        Args:
            estimates: List of dicts with keys: label, estimate, ci_lower, ci_upper, weight (optional)
            xlabel: X-axis label
            title: Plot title
            show_ci: Show confidence intervals
            log_scale: Use log scale (for odds ratios, risk ratios)
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        n = len(estimates)
        y_pos = np.arange(n)

        labels = [e['label'] for e in estimates]
        point_estimates = np.array([e['estimate'] for e in estimates])
        ci_lower = np.array([e['ci_lower'] for e in estimates])
        ci_upper = np.array([e['ci_upper'] for e in estimates])

        if log_scale:
            point_estimates = np.log(point_estimates)
            ci_lower = np.log(ci_lower)
            ci_upper = np.log(ci_upper)

        # Plot CIs
        if show_ci:
            ax.hlines(y_pos, ci_lower, ci_upper, colors='black', linewidth=1.5)

        # Plot point estimates
        ax.scatter(point_estimates, y_pos, color='red', s=50, zorder=5, label='Estimate')

        # Vertical line at null
        null_val = 0 if not log_scale else 0
        ax.axvline(null_val, color='gray', linestyle='--', linewidth=1)

        # Diamond for overall (if provided)
        if any('is_overall' in e for e in estimates):
            overall_idx = next(i for i, e in enumerate(estimates) if e.get('is_overall'))
            self._plot_diamond(ax, point_estimates[overall_idx],
                             ci_lower[overall_idx], ci_upper[overall_idx],
                             y_pos[overall_idx])

        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels)
        ax.set_xlabel(xlabel)
        ax.set_title(title)
        ax.grid(True, axis='x', alpha=0.3)

        plt.tight_layout()
        return fig

    def _plot_diamond(self, ax, center, left, right, y):
        """Plot diamond for overall estimate."""
        diamond_x = [left, center, right, center, left]
        diamond_y = [y, y + 0.15, y, y - 0.15, y]
        ax.fill(diamond_x, diamond_y, color='red', alpha=0.3)

    def plot_interactive(
        self,
        estimates: List[Dict],
        xlabel: str = "Effect Size",
        title: str = "Forest Plot",
    ) -> go.Figure:
        """Create interactive Plotly forest plot."""
        import plotly.graph_objects as go

        labels = [e['label'] for e in estimates]
        point_estimates = np.array([e['estimate'] for e in estimates])
        ci_lower = np.array([e['ci_lower'] for e in estimates])
        ci_upper = np.array([e['ci_upper'] for e in estimates])

        fig = go.Figure()

        # CIs
        fig.add_trace(go.Scatter(
            x=np.concatenate([ci_lower, ci_upper[::-1]]),
            y=np.concatenate([labels, labels[::-1]]),
            fill='toself',
            fillcolor='rgba(0,100,80,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            showlegend=False,
            hoverinfo='skip',
        ))

        # Point estimates
        fig.add_trace(go.Scatter(
            x=point_estimates,
            y=labels,
            mode='markers',
            marker=dict(size=10, color='red'),
            name='Point Estimate',
            hovertemplate='%{y}: %{x:.3f}<extra></extra>',
        ))

        # Null line
        fig.add_vline(x=0, line_dash='dash', line_color='gray')

        fig.update_layout(
            title=title,
            xaxis_title=xlabel,
            yaxis=dict(autorange='reversed'),
            height=400 + 30 * len(labels),
            showlegend=False,
        )

        return fig


class LovePlot:
    """Love plot for covariate balance diagnostics."""

    def __init__(self, figsize: Tuple[int, int] = (10, 6)):
        self.figsize = figsize

    def plot(
        self,
        standardized_diffs: Dict[str, Tuple[float, float]],
        threshold: float = 0.1,
        title: str = "Covariate Balance (Love Plot)",
        **kwargs,
    ) -> plt.Figure:
        """
        Plot standardized mean differences before/after adjustment.

        Args:
            standardized_diffs: Dict of {covariate: (before_smd, after_smd)}
            threshold: SMD threshold for balance (default 0.1)
            title: Plot title
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        covariates = list(standardized_diffs.keys())
        before = [standardized_diffs[c][0] for c in covariates]
        after = [standardized_diffs[c][1] for c in covariates]

        y_pos = np.arange(len(covariates))

        # Plot
        ax.scatter(before, y_pos, color='red', s=60, label='Before', zorder=5, marker='o')
        ax.scatter(after, y_pos, color='green', s=60, label='After', zorder=5, marker='s')

        # Connect lines
        for i in range(len(covariates)):
            ax.plot([before[i], after[i]], [y_pos[i], y_pos[i]], color='gray', alpha=0.5)

        # Threshold lines
        ax.axvline(threshold, color='red', linestyle='--', alpha=0.5, label=f'Threshold ({threshold})')
        ax.axvline(-threshold, color='red', linestyle='--', alpha=0.5)
        ax.axvline(0, color='black', linewidth=0.5)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(covariates)
        ax.set_xlabel("Standardized Mean Difference")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, axis='x', alpha=0.3)

        plt.tight_layout()
        return fig


class SensitivityPlot:
    """Sensitivity analysis visualization."""

    def __init__(self, figsize: Tuple[int, int] = (10, 6)):
        self.figsize = figsize

    def plot_rosenbaum(
        self,
        gammas: List[float],
        p_values: List[float],
        alpha: float = 0.05,
        critical_gamma: float = None,
        **kwargs,
    ) -> plt.Figure:
        """Plot Rosenbaum bounds: p-value upper bound vs Gamma."""
        fig, ax = plt.subplots(figsize=self.figsize)

        ax.plot(gammas, p_values, 'b-', linewidth=2, label='Upper bound p-value')
        ax.axhline(alpha, color='red', linestyle='--', label=f'α = {alpha}')
        if critical_gamma:
            ax.axvline(critical_gamma, color='orange', linestyle='--',
                       label=f'Critical Γ = {critical_gamma:.2f}')
        ax.set_xlabel('Sensitivity Parameter (Γ)')
        ax.set_ylabel('Upper Bound p-value')
        ax.set_title('Rosenbaum Sensitivity Analysis')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_cinelli_hazlett(
        self,
        r2_yz: List[float],
        r2_zd: List[float],
        adjusted_estimates: np.ndarray,
        significant: np.ndarray,
        rv: float,
        **kwargs,
    ) -> plt.Figure:
        """Plot TIPS contour for Cinelli-Hazlett sensitivity."""
        fig, ax = plt.subplots(figsize=self.figsize)

        R2_yz, R2_zd = np.meshgrid(r2_yz, r2_zd)

        # Contour for adjusted estimates
        cf = ax.contourf(R2_yz, R2_zd, adjusted_estimates, levels=20, cmap='RdBu_r', alpha=0.7)
        plt.colorbar(cf, ax=ax, label='Adjusted Estimate')

        # Significance boundary
        ax.contour(R2_yz, R2_zd, significant, levels=[0.5], colors='black', linewidths=2)

        # RV contour
        if rv > 0:
            rv_curve_y = (rv**2) / (R2_yz * (1 - rv**2) + rv**2)
            ax.plot(r2_yz, rv_curve_y, 'w--', linewidth=2, label=f'RV = {rv:.3f}')

        ax.set_xlabel('R²_{Y←Z|X,D} (Outcome confounding)')
        ax.set_ylabel('R²_{Z←D|X} (Treatment confounding)')
        ax.set_title('Cinelli-Hazlett Sensitivity Contours')
        ax.legend()
        plt.tight_layout()
        return fig


class UpliftPlot:
    """Uplift modeling visualization."""

    def __init__(self, figsize: Tuple[int, int] = (10, 6)):
        self.figsize = figsize

    def plot_qini(
        self,
        uplift: np.ndarray,
        treatment: np.ndarray,
        outcome: np.ndarray,
        n_bins: int = 10,
        **kwargs,
    ) -> plt.Figure:
        """Plot Qini curve (cumulative uplift vs population fraction)."""
        fig, ax = plt.subplots(figsize=self.figsize)

        # Sort by predicted uplift descending
        order = np.argsort(uplift)[::-1]
        uplift_sorted = uplift[order]
        treatment_sorted = treatment[order]
        outcome_sorted = outcome[order]

        n = len(uplift)
        bin_size = max(1, n // n_bins)

        x_vals = []
        y_vals = []
        random_y = []

        for i in range(1, n_bins + 1):
            end = min(i * bin_size, n)
            x_vals.append(end / n)

            # Model uplift in this segment
            treat = treatment_sorted[:end]
            out = outcome_sorted[:end]
            treated = out[treat == 1].mean() if treat.sum() > 0 else 0
            control = out[treat == 0].mean() if (1 - treat).sum() > 0 else 0
            y_vals.append(treated - control)

            # Random baseline
            idx = np.random.permutation(n)[:end]
            rand_treat = treatment[idx]
            rand_out = outcome[idx]
            rand_treated = rand_out[rand_treat == 1].mean() if rand_treat.sum() > 0 else 0
            rand_control = rand_out[rand_treat == 0].mean() if (1 - rand_treat).sum() > 0 else 0
            random_y.append(rand_treated - rand_control)

        ax.plot(x_vals, np.cumsum(y_vals) / np.sum(np.abs(y_vals)) if np.sum(np.abs(y_vals)) > 0 else np.zeros_like(y_vals),
                'b-', linewidth=2, label='Model')
        ax.plot(x_vals, np.cumsum(random_y) / np.sum(np.abs(random_y)) if np.sum(np.abs(random_y)) > 0 else np.zeros_like(random_y),
                'r--', linewidth=1, label='Random')
        ax.fill_between(x_vals, 0, np.cumsum(y_vals) / np.sum(np.abs(y_vals)) if np.sum(np.abs(y_vals)) > 0 else np.zeros_like(y_vals),
                        alpha=0.2, color='blue')

        ax.set_xlabel('Population Fraction')
        ax.set_ylabel('Cumulative Uplift (normalized)')
        ax.set_title('Qini Curve')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_gain(
        self,
        uplift: np.ndarray,
        treatment: np.ndarray,
        outcome: np.ndarray,
        n_bins: int = 10,
        **kwargs,
    ) -> plt.Figure:
        """Plot gain/decile chart."""
        fig, ax = plt.subplots(figsize=self.figsize)

        order = np.argsort(uplift)[::-1]
        n = len(uplift)
        bin_size = n // n_bins

        gains = []
        for i in range(n_bins):
            start = i * bin_size
            end = min((i + 1) * bin_size, n)
            idx = order[start:end]
            treat = treatment[idx]
            out = outcome[idx]
            treated = out[treat == 1].mean() if treat.sum() > 0 else 0
            control = out[treat == 0].mean() if (1 - treat).sum() > 0 else 0
            gains.append(treated - control)

        x = np.arange(1, n_bins + 1)
        ax.bar(x, gains, alpha=0.7, color='steelblue', edgecolor='black')
        ax.axhline(0, color='black', linewidth=0.5)
        ax.set_xlabel('Decile (by predicted uplift)')
        ax.set_ylabel('Average Uplift')
        ax.set_title('Uplift by Decile')
        ax.grid(True, axis='y', alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_interactive_qini(
        self,
        uplift: np.ndarray,
        treatment: np.ndarray,
        outcome: np.ndarray,
        n_bins: int = 10,
    ) -> go.Figure:
        """Interactive Qini curve with Plotly."""
        order = np.argsort(uplift)[::-1]
        n = len(uplift)
        bin_size = n // n_bins

        x_vals = []
        model_y = []
        random_y = []

        for i in range(1, n_bins + 1):
            end = min(i * bin_size, n)
            x_vals.append(end / n)

            treat = treatment[order[:end]]
            out = outcome[order[:end]]
            treated = out[treat == 1].mean() if treat.sum() > 0 else 0
            control = out[treat == 0].mean() if (1 - treat).sum() > 0 else 0
            model_y.append(treated - control)

            idx = np.random.permutation(n)[:end]
            rand_treat = treatment[idx]
            rand_out = outcome[idx]
            rand_treated = rand_out[rand_treat == 1].mean() if rand_treat.sum() > 0 else 0
            rand_control = rand_out[rand_treat == 0].mean() if (1 - rand_treat).sum() > 0 else 0
            random_y.append(rand_treated - rand_control)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x_vals, y=np.cumsum(model_y),
            mode='lines+markers', name='Model',
            line=dict(color='blue', width=3),
            marker=dict(size=6),
        ))
        fig.add_trace(go.Scatter(
            x=x_vals, y=np.cumsum(random_y),
            mode='lines', name='Random',
            line=dict(color='red', width=2, dash='dash'),
        ))

        fig.update_layout(
            title='Qini Curve (Interactive)',
            xaxis_title='Population Fraction',
            yaxis_title='Cumulative Uplift',
            hovermode='x unified',
        )

        return fig


class CounterfactualPlot:
    """Counterfactual outcome distributions visualization."""

    def __init__(self, figsize: Tuple[int, int] = (10, 6)):
        self.figsize = figsize

    def plot_distributions(
        self,
        y0_samples: np.ndarray,  # Control potential outcomes
        y1_samples: np.ndarray,  # Treatment potential outcomes
        unit_id: int = None,
        **kwargs,
    ) -> plt.Figure:
        """Plot individual counterfactual outcome distributions."""
        fig, axes = plt.subplots(1, 2, figsize=(self.figsize[0] * 1.5, self.figsize[1]))

        # Control distribution
        axes[0].hist(y0_samples, bins=30, alpha=0.7, color='blue', density=True, edgecolor='black')
        axes[0].axvline(np.mean(y0_samples), color='darkblue', linestyle='--', linewidth=2, label=f'Mean: {np.mean(y0_samples):.2f}')
        axes[0].set_title(f'Y(0) - Control Outcome{f" (Unit {unit_id})" if unit_id else ""}')
        axes[0].set_xlabel('Outcome')
        axes[0].set_ylabel('Density')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Treatment distribution
        axes[1].hist(y1_samples, bins=30, alpha=0.7, color='red', density=True, edgecolor='black')
        axes[1].axvline(np.mean(y1_samples), color='darkred', linestyle='--', linewidth=2, label=f'Mean: {np.mean(y1_samples):.2f}')
        axes[1].set_title(f'Y(1) - Treatment Outcome{f" (Unit {unit_id})" if unit_id else ""}')
        axes[1].set_xlabel('Outcome')
        axes[1].set_ylabel('Density')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # ITE distribution
        ite = y1_samples - y0_samples
        fig.suptitle(f'ITE Mean: {np.mean(ite):.2f}, Std: {np.std(ite):.2f}')

        plt.tight_layout()
        return fig

    def plot_ite_distribution(
        self,
        ite_samples: np.ndarray,
        **kwargs,
    ) -> plt.Figure:
        """Plot Individual Treatment Effect distribution."""
        fig, ax = plt.subplots(figsize=self.figsize)

        ax.hist(ite_samples, bins=50, alpha=0.7, color='purple', density=True, edgecolor='black')
        ax.axvline(np.mean(ite_samples), color='darkviolet', linestyle='--', linewidth=2,
                   label=f'Mean ITE: {np.mean(ite_samples):.2f}')
        ax.axvline(0, color='black', linewidth=0.5)

        # Add percentile annotations
        for p in [5, 25, 50, 75, 95]:
            val = np.percentile(ite_samples, p)
            ax.axvline(val, color='gray', linestyle=':', alpha=0.5)
            ax.text(val, ax.get_ylim()[1] * 0.9, f'P{p}', rotation=90, fontsize=8)

        ax.set_xlabel('Individual Treatment Effect')
        ax.set_ylabel('Density')
        ax.set_title('ITE Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig


# Convenience functions
def plot_causal_graph(*args, **kwargs) -> plt.Figure:
    return CausalGraphVisualizer().plot_dag(*args, **kwargs)


def plot_forest(*args, **kwargs) -> plt.Figure:
    return ForestPlot().plot(*args, **kwargs)


def plot_love(*args, **kwargs) -> plt.Figure:
    return LovePlot().plot(*args, **kwargs)


def plot_sensitivity_rosenbaum(*args, **kwargs) -> plt.Figure:
    return SensitivityPlot().plot_rosenbaum(*args, **kwargs)


def plot_sensitivity_ch(*args, **kwargs) -> plt.Figure:
    return SensitivityPlot().plot_cinelli_hazlett(*args, **kwargs)


def plot_qini(*args, **kwargs) -> plt.Figure:
    return UpliftPlot().plot_qini(*args, **kwargs)


def plot_counterfactual(*args, **kwargs) -> plt.Figure:
    return CounterfactualPlot().plot_distributions(*args, **kwargs)