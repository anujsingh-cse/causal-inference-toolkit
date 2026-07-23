"""
Causal Graph Visualization Module

Provides DAG rendering with do-calculus highlighting, backdoor paths analysis,
and interactive graph exploration.
"""

from typing import Any

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

try:
    import graphviz

    HAS_GRAPHVIZ = True
except ImportError:
    HAS_GRAPHVIZ = False

try:
    import plotly.graph_objects as go

    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False


class CausalGraphVisualizer:
    """
    Visualize causal DAGs with do-calculus annotations.
    """

    def __init__(
        self,
        figsize: tuple[int, int] = (12, 8),
        layout: str = "spring",
        node_size: int = 1500,
        font_size: int = 12,
    ):
        self.figsize = figsize
        self.layout = layout
        self.node_size = node_size
        self.font_size = font_size

        self._graph: nx.DiGraph | None = None
        self._pos: dict | None = None
        self._treatment: str | None = None
        self._outcome: str | None = None
        self._common_causes: list[str] = []
        self._instruments: list[str] = []
        self._mediators: list[str] = []

    def from_graph_spec(
        self,
        edges: list[tuple[str, str]],
        treatment: str,
        outcome: str,
        common_causes: list[str] = None,
        instruments: list[str] = None,
        mediators: list[str] = None,
    ) -> "CausalGraphVisualizer":
        """Build graph from edge list and variable roles."""
        self._graph = nx.DiGraph()
        self._graph.add_edges_from(edges)
        self._treatment = treatment
        self._outcome = outcome
        self._common_causes = common_causes or []
        self._instruments = instruments or []
        self._mediators = mediators or []
        self._compute_layout()
        return self

    def from_string(self, dot_string: str) -> "CausalGraphVisualizer":
        """Parse DOT format graph string."""
        self._graph = nx.nx_agraph.from_dot(dot_string) if HAS_GRAPHVIZ else nx.DiGraph()
        if not hasattr(self._graph, "nodes"):
            # Fallback: simple parser
            self._graph = self._parse_dot_simple(dot_string)
        self._compute_layout()
        return self

    def _parse_dot_simple(self, dot: str) -> nx.DiGraph:
        """Simple DOT parser fallback."""
        G = nx.DiGraph()
        lines = dot.split("\n")
        for line in lines:
            line = line.strip()
            if "->" in line and not line.startswith("//"):
                parts = line.replace("->", " ").replace(";", "").split()
                if len(parts) >= 2:
                    G.add_edge(parts[0].strip(), parts[1].strip())
        return G

    def _compute_layout(self) -> None:
        """Compute node positions."""
        if self._graph is None or len(self._graph.nodes()) == 0:
            self._pos = {}
            return

        try:
            if self.layout == "spring":
                self._pos = nx.spring_layout(self._graph, k=2, iterations=50, seed=42)
            elif self.layout == "kamada":
                self._pos = nx.kamada_kawai_layout(self._graph)
            elif self.layout == "circular":
                self._pos = nx.circular_layout(self._graph)
            elif self.layout == "hierarchical":
                self._pos = (
                    nx.nx_agraph.graphviz_layout(self._graph, prog="dot")
                    if HAS_GRAPHVIZ
                    else nx.spring_layout(self._graph)
                )
            else:
                self._pos = nx.spring_layout(self._graph, seed=42)
        except Exception:
            self._pos = nx.spring_layout(self._graph, seed=42)

    def identify_backdoor_paths(self) -> list[list[str]]:
        """Find all backdoor paths from treatment to outcome."""
        if self._graph is None or self._treatment is None or self._outcome is None:
            return []

        # Convert to undirected for path finding
        undirected = self._graph.to_undirected()

        all_paths = list(nx.all_simple_paths(undirected, self._treatment, self._outcome))
        backdoor_paths = []

        for path in all_paths:
            # Check if it's a backdoor path: starts with edge INTO treatment
            if len(path) >= 2 and self._graph.has_edge(path[1], path[0]):
                backdoor_paths.append(path)

        return backdoor_paths

    def find_adjustment_sets(self) -> list[list[str]]:
        """Find minimal valid adjustment sets (backdoor criterion)."""
        backdoor_paths = self.identify_backdoor_paths()
        if not backdoor_paths:
            return [[]]

        # Find nodes that block all backdoor paths (excluding treatment/outcome)
        candidates = set(self._graph.nodes()) - {self._treatment, self._outcome}

        valid_sets = []
        for r in range(len(candidates) + 1):
            from itertools import combinations

            for combo in combinations(candidates, r):
                if self._blocks_all_paths(set(combo), backdoor_paths):
                    valid_sets.append(list(combo))

        # Return minimal sets only
        minimal = []
        for s in valid_sets:
            if not any(set(other).issubset(set(s)) and len(other) < len(s) for other in valid_sets):
                minimal.append(s)

        return minimal

    def _blocks_all_paths(self, adjustment: set[str], paths: list[list[str]]) -> bool:
        """Check if adjustment set blocks all paths."""
        for path in paths:
            if not self._blocks_path(adjustment, path):
                return False
        return True

    def _blocks_path(self, adjustment: set[str], path: list[str]) -> bool:
        """Check if adjustment set blocks a single path."""
        # Check each collider on path
        for i in range(1, len(path) - 1):
            prev, curr, nxt = path[i - 1], path[i], path[i + 1]

            # Check if curr is collider: prev -> curr <- nxt
            is_collider = self._graph.has_edge(prev, curr) and self._graph.has_edge(nxt, curr)

            if is_collider:
                # Collider blocks path UNLESS adjusted for
                if curr in adjustment or any(
                    d in adjustment for d in nx.descendants(self._graph, curr)
                ):
                    return False  # Unblocked
            else:
                # Non-collider blocks path IF adjusted for
                if curr in adjustment:
                    return True  # Blocked

        return True  # Blocked if we reach here

    def plot_dag(
        self,
        highlight_backdoor: bool = True,
        highlight_adjustment: list[str] | None = None,
        show_legend: bool = True,
        title: str = "Causal DAG",
        save_path: str | None = None,
    ) -> plt.Figure:
        """
        Plot the causal DAG with optional backdoor/adjustment highlighting.

        Args:
            highlight_backdoor: Highlight backdoor paths in red
            highlight_adjustment: List of nodes to highlight as adjustment set
            show_legend: Show legend
            title: Plot title
            save_path: Optional path to save figure
        """
        if self._graph is None:
            raise ValueError("No graph loaded. Use from_graph_spec() first.")

        fig, ax = plt.subplots(figsize=self.figsize)

        # Node colors by role
        node_colors = []
        node_labels = {}
        for node in self._graph.nodes():
            labels = []
            if node == self._treatment:
                labels.append("T")
                color = "#ff6b6b"  # Red
            elif node == self._outcome:
                labels.append("Y")
                color = "#4ecdc4"  # Teal
            else:
                color = "#95a5a6"  # Gray

            if node in self._common_causes:
                labels.append("C")
                color = "#f39c12"  # Orange
            if node in self._instruments:
                labels.append("Z")
                color = "#9b59b6"  # Purple
            if node in self._mediators:
                labels.append("M")
                color = "#2ecc71"  # Green
            if highlight_adjustment and node in highlight_adjustment:
                color = "#e74c3c"  # Bright red for adjustment

            node_colors.append(color)
            node_labels[node] = node + ("\n(" + ", ".join(labels) + ")" if labels else "")

        # Draw edges
        nx.draw_networkx_edges(
            self._graph,
            self._pos,
            edge_color="gray",
            width=1.5,
            arrowsize=20,
            arrowstyle="->",
            node_size=self.node_size,
            ax=ax,
        )

        # Highlight backdoor edges
        if highlight_backdoor:
            backdoor_paths = self.identify_backdoor_paths()
            for path in backdoor_paths:
                for i in range(len(path) - 1):
                    if self._graph.has_edge(path[i + 1], path[i]):  # Edge into treatment
                        nx.draw_networkx_edges(
                            self._graph,
                            self._pos,
                            edgelist=[(path[i + 1], path[i])],
                            edge_color="red",
                            width=3,
                            alpha=0.7,
                            arrowsize=25,
                            arrowstyle="->",
                            node_size=self.node_size,
                            ax=ax,
                        )

        # Draw nodes
        nx.draw_networkx_nodes(
            self._graph, self._pos, node_color=node_colors, node_size=self.node_size, ax=ax
        )

        # Draw labels
        nx.draw_networkx_labels(
            self._graph,
            self._pos,
            labels=node_labels,
            font_size=self.font_size,
            font_weight="bold",
            ax=ax,
        )

        # Legend
        if show_legend:
            legend_elements = [
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor="#ff6b6b",
                    markersize=15,
                    label="Treatment (T)",
                ),
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor="#4ecdc4",
                    markersize=15,
                    label="Outcome (Y)",
                ),
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor="#f39c12",
                    markersize=15,
                    label="Confounder (C)",
                ),
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor="#9b59b6",
                    markersize=15,
                    label="Instrument (Z)",
                ),
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor="#2ecc71",
                    markersize=15,
                    label="Mediator (M)",
                ),
            ]
            if highlight_adjustment:
                legend_elements.append(
                    plt.Line2D(
                        [0],
                        [0],
                        marker="o",
                        color="w",
                        markerfacecolor="#e74c3c",
                        markersize=15,
                        label="Adjustment Set",
                    )
                )
            if highlight_backdoor:
                legend_elements.append(
                    plt.Line2D([0], [0], color="red", linewidth=3, label="Backdoor Path")
                )
            ax.legend(handles=legend_elements, loc="upper left", bbox_to_anchor=(1, 1))

        ax.set_title(title, fontsize=16, fontweight="bold")
        ax.axis("off")
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def plot_interactive(
        self, highlight_backdoor: bool = True, title: str = "Interactive Causal DAG"
    ) -> Any:
        """Create interactive Plotly graph."""
        if not HAS_PLOTLY:
            raise ImportError("Plotly not installed. Install with: pip install plotly")

        if self._graph is None:
            raise ValueError("No graph loaded.")

        # Prepare nodes
        x_nodes = [self._pos[n][0] for n in self._graph.nodes()]
        y_nodes = [self._pos[n][1] for n in self._graph.nodes()]

        node_texts = []
        node_colors = []
        for n in self._graph.nodes():
            labels = []
            if n == self._treatment:
                labels.append("T")
                color = "#ff6b6b"
            elif n == self._outcome:
                labels.append("Y")
                color = "#4ecdc4"
            else:
                color = "#95a5a6"
            if n in self._common_causes:
                labels.append("C")
                color = "#f39c12"
            if n in self._instruments:
                labels.append("Z")
                color = "#9b59b6"
            if n in self._mediators:
                labels.append("M")
                color = "#2ecc71"
            node_texts.append(f"{n}<br>({' | '.join(labels)})")
            node_colors.append(color)

        # Edges
        edge_traces = []
        for u, v in self._graph.edges():
            x0, y0 = self._pos[u]
            x1, y1 = self._pos[v]
            edge_traces.append(
                go.Scatter(
                    x=[x0, x1, None],
                    y=[y0, y1, None],
                    mode="lines",
                    line=dict(color="gray", width=1.5),
                    hoverinfo="none",
                    showlegend=False,
                )
            )

        # Backdoor edges
        if highlight_backdoor:
            backdoor_paths = self.identify_backdoor_paths()
            for path in backdoor_paths:
                for i in range(len(path) - 1):
                    if self._graph.has_edge(path[i + 1], path[i]):
                        x0, y0 = self._pos[path[i + 1]]
                        x1, y1 = self._pos[path[i]]
                        edge_traces.append(
                            go.Scatter(
                                x=[x0, x1, None],
                                y=[y0, y1, None],
                                mode="lines",
                                line=dict(color="red", width=3),
                                hoverinfo="none",
                                showlegend=False,
                            )
                        )

        # Nodes
        node_trace = go.Scatter(
            x=x_nodes,
            y=y_nodes,
            mode="markers+text",
            marker=dict(size=30, color=node_colors, line=dict(width=2, color="white")),
            text=[n for n in self._graph.nodes()],
            textposition="middle center",
            hovertext=node_texts,
            hoverinfo="text",
            showlegend=False,
        )

        fig = go.Figure(data=edge_traces + [node_trace])
        fig.update_layout(
            title=title,
            showlegend=False,
            hovermode="closest",
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor="white",
        )

        return fig

    def to_graphviz(self, **kwargs) -> Any:
        """Export to Graphviz for high-quality rendering."""
        if not HAS_GRAPHVIZ:
            raise ImportError("Graphviz not installed. Install with: pip install graphviz")

        dot = graphviz.Digraph(**kwargs)

        # Add nodes with styling
        for node in self._graph.nodes():
            attrs = {"shape": "ellipse", "style": "filled"}
            if node == self._treatment:
                attrs["fillcolor"] = "#ff6b6b"
                attrs["label"] = f"{node} (T)"
            elif node == self._outcome:
                attrs["fillcolor"] = "#4ecdc4"
                attrs["label"] = f"{node} (Y)"
            elif node in self._common_causes:
                attrs["fillcolor"] = "#f39c12"
                attrs["label"] = f"{node} (C)"
            elif node in self._instruments:
                attrs["fillcolor"] = "#9b59b6"
                attrs["label"] = f"{node} (Z)"
            elif node in self._mediators:
                attrs["fillcolor"] = "#2ecc71"
                attrs["label"] = f"{node} (M)"
            else:
                attrs["fillcolor"] = "#95a5a6"
                attrs["label"] = node

            dot.node(node, **attrs)

        # Add edges
        for u, v in self._graph.edges():
            dot.edge(u, v)

        return dot

    def compute_do_calculus_steps(self, query: str = "P(Y|do(T))") -> list[str]:
        """
        Show step-by-step do-calculus derivation.

        This is a simplified implementation showing the logic.
        """
        steps = [f"Target: {query}"]

        if self._treatment is None or self._outcome is None:
            return steps + ["Error: Treatment/outcome not specified"]

        # Step 1: Check if backdoor criterion satisfied
        adjustment_sets = self.find_adjustment_sets()
        if adjustment_sets:
            steps.append("Step 1: Backdoor criterion satisfied with adjustment sets:")
            for i, s in enumerate(adjustment_sets[:3]):
                steps.append(f"  Set {i + 1}: {{{', '.join(s) if s else '∅'}}}")
            steps.append("Step 2: Apply backdoor adjustment: P(Y|do(T)) = Σ_{C∈Adj} P(Y|T,C)P(C)")
            steps.append("Step 3: Estimate using adjustment formula")
        else:
            steps.append("Step 1: Backdoor criterion NOT satisfied (no valid adjustment set)")
            # Check frontdoor
            if self._mediators:
                steps.append("Step 2: Check frontdoor criterion via mediators...")
                steps.append(f"  Mediators available: {self._mediators}")
            # Check IV
            if self._instruments:
                steps.append("Step 2: Check instrumental variables...")
                steps.append(f"  Instruments available: {self._instruments}")

        return steps


class InterventionVisualizer:
    """Visualize intervention effects: do(T=t) vs observe T=t."""

    def __init__(self, figsize: tuple[int, int] = (12, 5)):
        self.figsize = figsize

    def plot_do_vs_see(
        self,
        observational_data: dict[str, np.ndarray],
        interventional_data: dict[str, np.ndarray],
        treatment_name: str = "T",
        outcome_name: str = "Y",
        **kwargs,
    ) -> plt.Figure:
        """
        Compare observational vs interventional distributions.

        Args:
            observational_data: Dict with keys 'treatment', 'outcome'
            interventional_data: Dict with keys 'treatment', 'outcome'
        """
        fig, axes = plt.subplots(1, 2, figsize=self.figsize)

        # Observational
        ax = axes[0]
        if "outcome" in observational_data and "treatment" in observational_data:
            treat = observational_data["treatment"]
            out = observational_data["outcome"]
            for t_val in np.unique(treat):
                mask = treat == t_val
                ax.hist(
                    out[mask],
                    bins=30,
                    alpha=0.5,
                    density=True,
                    label=f"{treatment_name}={t_val}",
                    edgecolor="black",
                )
        ax.set_xlabel(outcome_name)
        ax.set_ylabel("Density")
        ax.set_title(f"Observational: P({outcome_name}|{treatment_name})")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Interventional
        ax = axes[1]
        if "outcome" in interventional_data and "treatment" in interventional_data:
            treat = interventional_data["treatment"]
            out = interventional_data["outcome"]
            for t_val in np.unique(treat):
                mask = treat == t_val
                ax.hist(
                    out[mask],
                    bins=30,
                    alpha=0.5,
                    density=True,
                    label=f"do({treatment_name}={t_val})",
                    edgecolor="black",
                )
        ax.set_xlabel(outcome_name)
        ax.set_ylabel("Density")
        ax.set_title(f"Interventional: P({outcome_name}|do({treatment_name}))")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig


# Convenience functions
def plot_causal_dag(*args, **kwargs) -> plt.Figure:
    """Quick function to plot causal DAG."""
    return CausalGraphVisualizer().plot_dag(*args, **kwargs)


def create_dag_from_spec(
    edges: list[tuple[str, str]], treatment: str, outcome: str, **kwargs
) -> CausalGraphVisualizer:
    """Factory function to create and configure visualizer."""
    return CausalGraphVisualizer().from_graph_spec(edges, treatment, outcome, **kwargs)
