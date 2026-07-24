"""
Utility functions for Causal Inference Toolkit.
"""

from typing import Any

import numpy as np
import pandas as pd


def standardize(
    df: pd.DataFrame, columns: list[str] | None = None, method: str = "zscore"
) -> pd.DataFrame:
    """
    Standardize numeric columns.

    Args:
        df: Input DataFrame
        columns: Columns to standardize (default: all numeric)
        method: 'zscore', 'minmax', 'robust'
    """
    result = df.copy()
    cols = columns or df.select_dtypes(include=[np.number]).columns.tolist()

    for col in cols:
        if method == "zscore":
            mean, std = df[col].mean(), df[col].std()
            if std > 0:
                result[col] = (df[col] - mean) / std
        elif method == "minmax":
            min_val, max_val = df[col].min(), df[col].max()
            if max_val > min_val:
                result[col] = (df[col] - min_val) / (max_val - min_val)
        elif method == "robust":
            median = df[col].median()
            mad = (df[col] - median).abs().median()
            if mad > 0:
                result[col] = (df[col] - median) / (1.4826 * mad)

    return result


def compute_smd(treated: pd.Series, control: pd.Series, pooled: bool = True) -> float:
    """
    Compute Standardized Mean Difference (Cohen's d) between two groups.

    Args:
        treated: Values for treated group
        control: Values for control group
        pooled: Use pooled variance (Cohen's d) or control variance (Glass's delta)
    """
    n1, n2 = len(treated), len(control)
    if n1 == 0 or n2 == 0:
        return np.nan

    mean1, mean2 = treated.mean(), control.mean()
    var1, var2 = treated.var(ddof=1), control.var(ddof=1)

    if pooled:
        pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
        return (mean1 - mean2) / np.sqrt(pooled_var) if pooled_var > 0 else np.nan
    else:
        return (mean1 - mean2) / np.sqrt(var2) if var2 > 0 else np.nan


def compute_all_smds(
    df: pd.DataFrame, treatment_col: str, covariates: list[str], threshold: float = 0.1
) -> dict[str, dict[str, float]]:
    """
    Compute SMDs for all covariates between treatment groups.

    Returns dict with before/after SMDs and balance flags.
    """
    treated = df[df[treatment_col] == 1]
    control = df[df[treatment_col] == 0]

    results = {}
    for cov in covariates:
        smd = compute_smd(treated[cov], control[cov])
        results[cov] = {
            "smd": smd,
            "balanced": abs(smd) < threshold if not np.isnan(smd) else False,
            "threshold": threshold,
        }
    return results


def propensity_score(
    df: pd.DataFrame,
    treatment_col: str,
    covariates: list[str],
    model: str = "logistic",
    **kwargs: Any,
) -> np.ndarray:
    """
    Estimate propensity scores.

    Args:
        df: DataFrame
        treatment_col: Binary treatment column
        covariates: Covariate columns
        model: 'logistic', 'gbm', 'rf', 'xgboost'
    """
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression

    X = df[covariates].values
    T = df[treatment_col].values

    if model == "logistic":
        clf = LogisticRegression(max_iter=1000, **kwargs)
    elif model == "gbm":
        clf = GradientBoostingClassifier(**kwargs)
    elif model == "rf":
        clf = RandomForestClassifier(**kwargs)
    else:
        raise ValueError(f"Unknown model: {model}")

    clf.fit(X, T)
    return np.asarray(clf.predict_proba(X)[:, 1])


def inverse_probability_weighting(
    treatment: np.ndarray, propensity: np.ndarray, stabilized: bool = True
) -> np.ndarray:
    """
    Compute inverse probability weights.

    Args:
        treatment: Binary treatment (0/1)
        propensity: Propensity scores P(T=1|X)
        stabilized: Use stabilized weights
    """
    # Avoid division by zero
    propensity = np.clip(propensity, 1e-6, 1 - 1e-6)

    if stabilized:
        p_treat = treatment.mean()
        weights = np.where(treatment == 1, p_treat / propensity, (1 - p_treat) / (1 - propensity))
    else:
        weights = np.where(treatment == 1, 1 / propensity, 1 / (1 - propensity))

    return weights


def trim_weights(
    weights: np.ndarray, lower_quantile: float = 0.01, upper_quantile: float = 0.99
) -> np.ndarray:
    """Trim extreme weights at quantiles."""
    lower = np.quantile(weights, lower_quantile)
    upper = np.quantile(weights, upper_quantile)
    return np.clip(weights, lower, upper)


def effective_sample_size(weights: np.ndarray) -> float:
    """Compute effective sample size from weights."""
    return float((np.sum(weights) ** 2) / np.sum(weights**2))


def bootstrap_ci(
    data: np.ndarray,
    statistic_fn: Any,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    random_state: int = 42,
) -> tuple[float, float]:
    """
    Compute bootstrap confidence interval.

    Args:
        data: Input data array
        statistic_fn: Function that computes statistic from data
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level
        random_state: Random seed
    """
    rng = np.random.default_rng(random_state)
    n = len(data)
    stats = []

    for _ in range(n_bootstrap):
        idx = rng.choice(n, n, replace=True)
        sample = data[idx]
        stats.append(statistic_fn(sample))

    stats_arr = np.array(stats)
    alpha = 1 - confidence
    lower = np.percentile(stats_arr, 100 * alpha / 2)
    upper = np.percentile(stats_arr, 100 * (1 - alpha / 2))

    return float(lower), float(upper)


def bootstrap_ci_pairs(
    data_a: np.ndarray,
    data_b: np.ndarray,
    statistic_fn: Any,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    random_state: int = 42,
) -> tuple[float, float]:
    """Bootstrap CI for pairwise statistic."""
    rng = np.random.default_rng(random_state)
    n_a, n_b = len(data_a), len(data_b)
    stats = []

    for _ in range(n_bootstrap):
        idx_a = rng.choice(n_a, n_a, replace=True)
        idx_b = rng.choice(n_b, n_b, replace=True)
        stats.append(statistic_fn(data_a[idx_a], data_b[idx_b]))

    stats_arr = np.array(stats)
    alpha = 1 - confidence
    lower = np.percentile(stats_arr, 100 * alpha / 2)
    upper = np.percentile(stats_arr, 100 * (1 - alpha / 2))

    return float(lower), float(upper)


def load_dataset(name: str) -> pd.DataFrame:
    """
    Load built-in benchmark datasets.

    Available: 'ihdp', 'lalonde', 'criteo_uplift'
    """
    if name == "ihdp":
        return _load_ihdp()
    elif name == "lalonde":
        return _load_lalonde()
    elif name == "criteo_uplift":
        return _load_criteo_uplift()
    else:
        raise ValueError(f"Unknown dataset: {name}")


def _load_ihdp() -> pd.DataFrame:
    """Load Infant Health Development Program dataset."""
    # IHDP is a standard causal inference benchmark
    # We generate a synthetic version here
    np.random.seed(42)
    n = 747

    # Covariates (simplified version of IHDP covariates)
    data = pd.DataFrame(
        {
            "treatment": np.random.binomial(1, 0.5, n),
            "bw": np.random.normal(3000, 500, n),  # birth weight
            "b.head": np.random.normal(34, 1.5, n),  # head circumference
            "preterm": np.random.binomial(1, 0.1, n),
            "birth.o": np.random.randint(1, 5, n),
            "nnhealth": np.random.binomial(1, 0.2, n),
            "momage": np.random.normal(27, 6, n),
            "sex": np.random.binomial(1, 0.5, n),
            "twin": np.random.binomial(1, 0.02, n),
            "b.marr": np.random.binomial(1, 0.7, n),
            "mom.lths": np.random.binomial(1, 0.2, n),
            "mom.hs": np.random.binomial(1, 0.3, n),
            "mom.sc": np.random.binomial(1, 0.2, n),
            "cig": np.random.binomial(1, 0.2, n),
            "first": np.random.binomial(1, 0.4, n),
            "booze": np.random.binomial(1, 0.1, n),
            "drugs": np.random.binomial(1, 0.02, n),
            "work.dur": np.random.normal(20, 10, n),
            "prenatal": np.random.binomial(1, 0.9, n),
            "site": np.random.randint(1, 9, n),
        }
    )

    # True ATE = 4 (simulate)
    # Y = 100 + 4*T + 0.01*bw + ... + noise
    data["outcome"] = (
        100
        + 4 * data["treatment"]  # True ATE
        + 0.01 * data["bw"]
        + 0.5 * data["momage"]
        - 2 * data["mom.lths"]
        + np.random.normal(0, 10, n)
    )

    return data


def _load_lalonde() -> pd.DataFrame:
    """Load Lalonde NSW/PSID dataset (simplified synthetic version)."""
    np.random.seed(42)
    n_treated = 185
    n_control = 15992  # PSID control size

    # Generate realistic covariates
    def gen_group(n: int, treat_prob: float) -> pd.DataFrame:
        age = np.random.normal(25, 8, n)
        educ = np.random.normal(10, 3, n).clip(0, 18).astype(int)
        black = np.random.binomial(1, 0.84, n)
        hisp = np.random.binomial(1, 0.06, n)
        married = np.random.binomial(1, 0.19, n)
        nodegree = np.random.binomial(1, 0.71, n)
        re74 = np.random.exponential(5000, n)
        re75 = np.random.exponential(3000, n)

        return pd.DataFrame(
            {
                "treatment": np.random.binomial(1, treat_prob, n),
                "age": age,
                "education": educ,
                "black": black,
                "hispanic": hisp,
                "married": married,
                "nodegree": nodegree,
                "re74": re74,  # 1974 earnings
                "re75": re75,  # 1975 earnings
            }
        )

    treated = gen_group(n_treated, 1.0)
    control = gen_group(n_control, 0.0)

    # Outcome: 1978 earnings (re78)
    def gen_outcome(df: pd.DataFrame) -> Any:
        return (
            1000
            + 100 * df["age"]
            + 200 * df["education"]
            - 1500 * df["black"]
            - 1000 * df["hispanic"]
            + 500 * df["married"]
            + 300 * (1 - df["nodegree"])
            + 0.1 * df["re74"]
            + 0.15 * df["re75"]
            + (
                1500 if "treatment" in df and df["treatment"].iloc[0] == 1 else 0
            )  # Treatment effect
            + np.random.normal(0, 3000, len(df))
        )

    treated["outcome"] = gen_outcome(treated)
    control["outcome"] = gen_outcome(control)

    return pd.concat([treated, control], ignore_index=True)


def _load_criteo_uplift() -> pd.DataFrame:
    """Load synthetic Criteo uplift benchmark."""
    np.random.seed(42)
    n = 10000

    # Simplified Criteo-like features
    data = pd.DataFrame(
        {
            "treatment": np.random.binomial(1, 0.5, n),
            "feature_0": np.random.normal(0, 1, n),
            "feature_1": np.random.normal(0, 1, n),
            "feature_2": np.random.normal(0, 1, n),
            "feature_3": np.random.normal(0, 1, n),
            "feature_4": np.random.normal(0, 1, n),
            "feature_5": np.random.normal(0, 1, n),
            "feature_6": np.random.normal(0, 1, n),
            "feature_7": np.random.normal(0, 1, n),
            "feature_8": np.random.normal(0, 1, n),
            "feature_9": np.random.normal(0, 1, n),
            "feature_10": np.random.normal(0, 1, n),
            "feature_11": np.random.normal(0, 1, n),
        }
    )

    # Heterogeneous treatment effect
    cate = (
        0.5 * data["feature_0"]
        + 0.3 * data["feature_1"] ** 2
        - 0.2 * data["feature_2"]
        + np.random.normal(0, 0.1, n)
    )

    data["outcome"] = (
        0.1
        + 0.5 * data["feature_0"]
        + 0.3 * data["feature_1"]
        + 0.2 * data["feature_2"]
        + data["treatment"] * cate
        + np.random.normal(0, 0.5, n)
    )

    return data


def create_synthetic_data(
    n: int = 1000,
    n_covariates: int = 10,
    ate: float = 2.0,
    heterogeneity: bool = True,
    confounding: bool = True,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Create synthetic causal inference dataset.

    Args:
        n: Sample size
        n_covariates: Number of covariates
        ate: Average treatment effect
        heterogeneity: Add treatment effect heterogeneity
        confounding: Add confounding between treatment and covariates
        random_state: Random seed
    """
    np.random.seed(random_state)

    # Generate covariates
    X = np.random.randn(n, n_covariates)
    X_df = pd.DataFrame(X, columns=[f"x{i}" for i in range(n_covariates)])

    # Treatment assignment
    if confounding:
        # Treatment depends on covariates
        logit_p = X[:, 0] + 0.5 * X[:, 1] - 0.3 * X[:, 2]
        p_treat = 1 / (1 + np.exp(-logit_p))
    else:
        p_treat = np.full(n, 0.5)

    treatment = np.random.binomial(1, p_treat)

    # Outcome with treatment effect
    if heterogeneity:
        # CATE varies with covariates
        cate = ate + 0.5 * X[:, 0] - 0.3 * X[:, 1] + 0.2 * X[:, 2] ** 2
    else:
        cate = ate * np.ones(n)

    outcome = (
        10 + X[:, 0] + 0.5 * X[:, 1] - 0.3 * X[:, 2] + treatment * cate + np.random.randn(n) * 2
    )

    data = X_df.copy()
    data["treatment"] = treatment
    data["outcome"] = outcome
    data["true_cate"] = cate

    return data
