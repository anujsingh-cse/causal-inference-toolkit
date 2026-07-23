# API: Visualization

## CausalGraphVisualizer

```python
class CausalGraphVisualizer:
    def __init__(
        self,
        figsize: Tuple[int, int] = (12, 8),
        layout: str = "spring",
        node_size: int = 1500,
        font_size: int = 12,
    )

    def from_graph_spec(
        self,
        edges: List[Tuple[str, str]],
        treatment: str,
        outcome: str,
        common_causes: List[str] = None,
        instruments: List[str] = None,
        mediators: List[str] = None,
    ) -> "CausalGraphVisualizer": ...

    def from_string(self, dot_string: str) -> "CausalGraphVisualizer": ...

    def identify_backdoor_paths(self) -> List[List[str]]: ...

    def find_adjustment_sets(self) -> List[List[str]]: ...

    def plot_dag(
        self,
        highlight_backdoor: bool = True,
        highlight_adjustment: Optional[List[str]] = None,
        show_legend: bool = True,
        title: str = "Causal DAG",
        save_path: Optional[str] = None,
    ) -> plt.Figure: ...

    def plot_interactive(
        self,
        highlight_backdoor: bool = True,
        title: str = "Interactive Causal DAG",
    ) -> go.Figure: ...

    def to_graphviz(self, **kwargs) -> Any: ...

    def compute_do_calculus_steps(
        self,
        query: str = "P(Y|do(T))",
    ) -> List[str]: ...

# Factory
def create_dag_from_spec(
    edges: List[Tuple[str, str]],
    treatment: str,
    outcome: str,
    **kwargs,
) -> CausalGraphVisualizer: ...

def plot_causal_dag(*args, **kwargs) -> plt.Figure: ...
```

## ForestPlot

```python
class ForestPlot:
    def __init__(self, figsize: Tuple[int, int] = (10, 6))

    def plot(
        self,
        estimates: List[Dict],
        xlabel: str = "Effect Size",
        title: str = "Forest Plot",
        show_ci: bool = True,
        log_scale: bool = False,
        **kwargs,
    ) -> plt.Figure: ...

    def plot_interactive(
        self,
        estimates: List[Dict],
        xlabel: str = "Effect Size",
        title: str = "Forest Plot",
    ) -> go.Figure: ...

    def _plot_diamond(
        self,
        ax,
        center: float,
        left: float,
        right: float,
        y: float,
    ): ...

def plot_forest(*args, **kwargs) -> plt.Figure: ...
```

## LovePlot

```python
class LovePlot:
    def __init__(self, figsize: Tuple[int, int] = (10, 6))

    def plot(
        self,
        standardized_diffs: Dict[str, Tuple[float, float]],
        threshold: float = 0.1,
        title: str = "Covariate Balance (Love Plot)",
        **kwargs,
    ) -> plt.Figure: ...

def plot_love(*args, **kwargs) -> plt.Figure: ...
```

## SensitivityPlot

```python
class SensitivityPlot:
    def __init__(self, figsize: Tuple[int, int] = (10, 6))

    def plot_rosenbaum(
        self,
        gammas: List[float],
        p_values: List[float],
        alpha: float = 0.05,
        critical_gamma: float = None,
        **kwargs,
    ) -> plt.Figure: ...

    def plot_cinelli_hazlett(
        self,
        r2_yz: List[float],
        r2_zd: List[float],
        adjusted_estimates: np.ndarray,
        significant: np.ndarray,
        rv: float,
        **kwargs,
    ) -> plt.Figure: ...

def plot_sensitivity_rosenbaum(*args, **kwargs) -> plt.Figure: ...
def plot_sensitivity_ch(*args, **kwargs) -> plt.Figure: ...
```

## UpliftPlot

```python
class UpliftPlot:
    def __init__(self, figsize: Tuple[int, int] = (10, 6))

    def plot_qini(
        self,
        uplift: np.ndarray,
        treatment: np.ndarray,
        outcome: np.ndarray,
        n_bins: int = 10,
        **kwargs,
    ) -> plt.Figure: ...

    def plot_gain(
        self,
        uplift: np.ndarray,
        treatment: np.ndarray,
        outcome: np.ndarray,
        n_bins: int = 10,
        **kwargs,
    ) -> plt.Figure: ...

    def plot_interactive_qini(
        self,
        uplift: np.ndarray,
        treatment: np.ndarray,
        outcome: np.ndarray,
        n_bins: int = 10,
    ) -> go.Figure: ...

def plot_qini(*args, **kwargs) -> plt.Figure: ...
```

## CounterfactualPlot

```python
class CounterfactualPlot:
    def __init__(self, figsize: Tuple[int, int] = (10, 6))

    def plot_distributions(
        self,
        y0_samples: np.ndarray,
        y1_samples: np.ndarray,
        unit_id: int = None,
        **kwargs,
    ) -> plt.Figure: ...

    def plot_ite_distribution(
        self,
        ite_samples: np.ndarray,
        **kwargs,
    ) -> plt.Figure: ...

def plot_counterfactual(*args, **kwargs) -> plt.Figure: ...
```

## CausalGraphVisualizer (plots.py - DoWhy integration)

```python
class CausalGraphVisualizer:
    def __init__(self, figsize: Tuple[int, int] = (10, 8))

    def plot_dag(
        self,
        graph: Any,  # networkx.DiGraph or DoWhy graph
        treatment: str,
        outcome: str,
        highlight_backdoor: bool = True,
        highlight_iv: bool = False,
        layout: str = "spring",
        **kwargs,
    ) -> plt.Figure: ...

    def _get_backdoor_nodes(self, G, treatment, outcome) -> set: ...
    def _get_iv_nodes(self, G, treatment) -> set: ...

    def plot_do_calculus_steps(
        self,
        steps: List[Dict],
        treatment: str,
        outcome: str,
    ) -> plt.Figure: ...

    def _plot_step_on_ax(self, ax, G, treatment, outcome, title): ...

def plot_causal_graph(*args, **kwargs) -> plt.Figure: ...
```
