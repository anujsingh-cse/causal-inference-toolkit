# Causal Inference Toolkit

**Production-ready causal inference for data scientists and ML engineers.**

---

## Why This Toolkit?

| Problem | Solution |
|---------|----------|
| "DoWhy is great but I need EconML's CATE estimators too" | **Unified wrapper** — one API for identification, estimation, refutation |
| "Sensitivity analysis is an afterthought" | **First-class citizens** — Rosenbaum, Cinelli-Hazlett, E-values, TIPS curves built in |
| "A/B testing needs both frequentist and Bayesian" | **Complete ABTestAnalyzer** — z-test, t-test, SPRT, mSPRT, Beta-Binomial, ROPE, expected loss |
| "Uplift modeling is scattered across libraries" | **UpliftModeler** — T/S/X/R/DR-learners + Causal Forest + Qini/AUUC evaluation |
| "I need to show my work" | **Publication-ready viz** — DAGs with backdoor highlighting, forest plots, love plots, Qini curves |

---

## Quick Links

| | | |
|---|---|---|
| 🚀 [Quickstart](getting-started/quickstart.md) | ⚙️ [Installation](getting-started/installation.md) | 📝 [Configuration](getting-started/configuration.md) |
| 🧠 [Core Concepts](concepts/causal-model.md) | 🔧 [DoWhy Wrapper](user-guide/dowhy.md) | 🌲 [EconML Wrapper](user-guide/econml.md) |
| 📊 [A/B Testing](user-guide/ab-testing.md) | 🎯 [Uplift Modeling](user-guide/uplift.md) | 📈 [Visualization](user-guide/visualization.md) |
| 🔍 [Sensitivity Analysis](concepts/sensitivity.md) | 💻 [CLI Reference](cli/estimate.md) | 📚 [API Reference](api/core.md) |

---

## Example: Full Pipeline in 10 Lines

```python
from causal_toolkit import CausalModel, DoWhyWrapper, SensitivityAnalyzer
from causal_toolkit.core.base import EstimatorType, IdentificationStrategy

# 1. Load data
df = pd.read_csv("data.csv")

# 2. Define model
model = CausalModel(df, treatment="treatment", outcome="outcome", 
                    common_causes=["age", "income", "education"])

# 3. Identify + Estimate
dowhy = DoWhyWrapper(model)
estimand = dowhy.identify(IdentificationStrategy.BACKDOOR)
estimate = dowhy.estimate(EstimatorType.DOUBLY_ROBUST)

# 4. Refute
for r in dowhy.refute():
    print(r)

# 5. Sensitivity
sa = SensitivityAnalyzer(model)
sa.cinelli_hazlett(estimate)
print(sa.summarize())
```

---

## Installation

```bash
pip install causal-toolkit
# With extras:
pip install "causal-toolkit[dev,viz,notebooks]"
```

---

## Built on Giants

| Library | Role |
|---------|------|
| **DoWhy** | Causal identification, refutation, do-calculus |
| **EconML** | Heterogeneous treatment effects (CATE), metalearners, forests |
| **CausalInference** | Propensity score methods, matching |
| **Statsmodels** | Power analysis, proportion tests |
| **NetworkX/Graphviz** | Causal graph algorithms & rendering |

---

## License

MIT License — see [LICENSE](https://github.com/anujsingh-cse/causal-inference-toolkit/blob/master/LICENSE)

## Citation

```bibtex
@software{causal_toolkit,
  author = {Anuj Singh},
  title = {Causal Inference Toolkit},
  year = {2025},
  url = {https://github.com/anujsingh-cse/causal-inference-toolkit}
}
```
