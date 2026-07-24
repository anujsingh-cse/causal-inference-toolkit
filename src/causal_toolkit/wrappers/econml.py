"""
EconML wrapper for CATE/heterogeneous treatment effect estimation.

Provides unified interface to EconML's estimators with our type definitions.
"""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from causal_toolkit.core.base import CausalEstimate, EstimatorType


class EconMLWrapper:
    """
    Wrapper around EconML estimators for CATE/ATE/ATT estimation.

    Supports: T-learner, S-learner, X-learner, R-learner, DR-learner,
    CausalForest, Metalearners, DeepIV, OrthoIV.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        treatment: str,
        outcome: str,
        covariates: list[str],
        effect_modifiers: list[str] | None = None,
    ):
        self.data = data.copy()
        self.treatment = treatment
        self.outcome = outcome
        self.covariates = covariates
        self.effect_modifiers = effect_modifiers or covariates

        self._X = self.data[self.covariates].values
        self._T = self.data[self.treatment].values
        self._Y = self.data[self.outcome].values
        if effect_modifiers:
            self._X_mod = self.data[self.effect_modifiers].values
        else:
            self._X_mod = self._X

        self._models: dict[str, Any] = {}

    def estimate_cate(self, estimator: EstimatorType, **estimator_kwargs: Any) -> CausalEstimate:
        """Estimate Conditional Average Treatment Effect (CATE)."""
        estimator_func = self._get_estimator(estimator, **estimator_kwargs)
        estimator_func.fit(self._Y, self._T, X=self._X, W=self._X_mod)
        cate = estimator_func.effect(self._X_mod)

        # Compute confidence intervals via bootstrap or built-in
        ci_lower, ci_upper = self._compute_ci(estimator_func, self._X_mod)

        return CausalEstimate(
            value=cate,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            confidence_level=0.95,
            estimator=estimator.value,
            n_samples=len(self.data),
            diagnostics={
                "model": estimator_func,
                "n_treated": np.sum(self._T),
                "n_control": np.sum(1 - self._T),
            },
        )

    def estimate_ate(self, estimator: EstimatorType, **estimator_kwargs: Any) -> CausalEstimate:
        """Estimate Average Treatment Effect (ATE)."""
        cate_estimate = self.estimate_cate(estimator, **estimator_kwargs)
        cate_vals = np.asarray(cate_estimate.value)
        ate = float(np.mean(cate_vals))
        ate_se = float(np.std(cate_vals) / np.sqrt(len(cate_vals)))

        from scipy import stats

        z = stats.norm.ppf(0.975)
        ci_lower = float(ate - z * ate_se)
        ci_upper = float(ate + z * ate_se)

        return CausalEstimate(
            value=ate,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            confidence_level=0.95,
            estimator=f"{estimator.value}_ate",
            standard_error=ate_se,
            n_samples=len(self.data),
            diagnostics=cate_estimate.diagnostics,
        )

    def _get_estimator(self, estimator: EstimatorType, **kwargs: Any) -> BaseEstimator:
        """Instantiate EconML estimator."""
        if estimator == EstimatorType.T_LEARNER:
            return self._t_learner(**kwargs)
        elif estimator == EstimatorType.S_LEARNER:
            return self._s_learner(**kwargs)
        elif estimator == EstimatorType.X_LEARNER:
            return self._x_learner(**kwargs)
        elif estimator == EstimatorType.R_LEARNER:
            return self._r_learner(**kwargs)
        elif estimator == EstimatorType.DR_LEARNER:
            return self._dr_learner(**kwargs)
        elif (
            estimator == EstimatorType.CAUSAL_FOREST_CATE
            or estimator == EstimatorType.CAUSAL_FOREST
        ):
            return self._causal_forest(**kwargs)
        elif estimator == EstimatorType.METALearners:
            return self._metalearner(**kwargs)
        else:
            raise ValueError(f"Unknown EconML estimator: {estimator}")

    def _t_learner(self, **kwargs: Any) -> Any:
        from econml.metalearners import TLearner
        from sklearn.ensemble import GradientBoostingRegressor

        models = kwargs.get(
            "models",
            [
                GradientBoostingRegressor(n_estimators=100, max_depth=3),
                GradientBoostingRegressor(n_estimators=100, max_depth=3),
            ],
        )
        return TLearner(models=models)

    def _s_learner(self, **kwargs: Any) -> Any:
        from econml.metalearners import SLearner
        from sklearn.ensemble import GradientBoostingRegressor

        model = kwargs.get("model", GradientBoostingRegressor(n_estimators=200, max_depth=3))
        return SLearner(overall_model=model)

    def _x_learner(self, **kwargs: Any) -> Any:
        from econml.metalearners import XLearner
        from sklearn.ensemble import GradientBoostingRegressor

        models = kwargs.get(
            "models",
            [
                GradientBoostingRegressor(n_estimators=100, max_depth=3),
                GradientBoostingRegressor(n_estimators=100, max_depth=3),
            ],
        )
        return XLearner(models=models, propensity_model=kwargs.get("propensity_model"))

    def _r_learner(self, **kwargs: Any) -> Any:
        from econml.metalearners import RLearner
        from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

        model = kwargs.get("model", GradientBoostingRegressor(n_estimators=200, max_depth=3))
        propensity = kwargs.get("propensity_model", GradientBoostingClassifier(n_estimators=100))
        return RLearner(overall_model=model, propensity_model=propensity)

    def _dr_learner(self, **kwargs: Any) -> Any:
        from econml.metalearners import DRLearner
        from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

        model = kwargs.get("model", GradientBoostingRegressor(n_estimators=200, max_depth=3))
        propensity = kwargs.get("propensity_model", GradientBoostingClassifier(n_estimators=100))
        return DRLearner(model_regression=model, model_propensity=propensity)

    def _causal_forest(self, **kwargs: Any) -> Any:
        from econml.grf import CausalForest

        return CausalForest(
            n_estimators=kwargs.get("n_estimators", 500),
            min_samples_leaf=kwargs.get("min_samples_leaf", 10),
            max_depth=kwargs.get("max_depth"),
            random_state=kwargs.get("random_state", 42),
        )

    def _metalearner(self, **kwargs: Any) -> Any:
        from econml.metalearners import Metalearner

        return Metalearner(**kwargs)

    def _compute_ci(self, model: Any, X: np.ndarray) -> tuple:
        """Compute confidence intervals for CATE predictions."""
        # Try built-in confidence intervals
        if hasattr(model, "effect_interval"):
            try:
                lower, upper = model.effect_interval(X)
                return lower, upper
            except Exception:
                pass

        # Fallback: bootstrap (simplified - in production use proper bootstrap)
        from scipy import stats

        se = np.std([m.predict(X) for m in getattr(model, "models_", [model])], axis=0)
        z = stats.norm.ppf(0.975)
        return -z * se, z * se


class UpliftModeler:
    """
    High-level uplift modeling interface.

    Wraps multiple uplift approaches:
    - Two-model approach
    - Class transformation
    - DR-learner for uplift
    - Causal Forest for uplift
    """

    def __init__(self, data: pd.DataFrame, treatment: str, outcome: str, covariates: list[str]):
        self.data = data.copy()
        self.treatment = treatment
        self.outcome = outcome
        self.covariates = covariates

        self._X = self.data[self.covariates].values
        self._T = self.data[self.treatment].values
        self._Y = self.data[self.outcome].values

        self._uplift_model: Any = None

    def fit(self, method: str = "causal_forest", **kwargs: Any) -> "UpliftModeler":
        """Fit uplift model."""
        if method == "causal_forest":
            self._uplift_model = self._fit_causal_forest(**kwargs)
        elif method == "two_model":
            self._uplift_model = self._fit_two_model(**kwargs)
        elif method == "class_transformation":
            self._uplift_model = self._fit_class_transformation(**kwargs)
        elif method == "dr_learner":
            self._uplift_model = self._fit_dr_learner(**kwargs)
        else:
            raise ValueError(f"Unknown uplift method: {method}")

        return self

    def _fit_causal_forest(self, **kwargs: Any) -> Any:
        from econml.grf import CausalForest

        model = CausalForest(
            n_estimators=kwargs.get("n_estimators", 500),
            min_samples_leaf=kwargs.get("min_samples_leaf", 10),
            max_depth=kwargs.get("max_depth", 10),
            random_state=kwargs.get("random_state", 42),
        )
        model.fit(self._Y, self._T, X=self._X)
        return model

    def _fit_two_model(self, **kwargs: Any) -> Any:
        from econml.metalearners import TLearner
        from sklearn.ensemble import GradientBoostingRegressor

        model = TLearner(
            models=[
                GradientBoostingRegressor(n_estimators=200, max_depth=3),
                GradientBoostingRegressor(n_estimators=200, max_depth=3),
            ]
        )
        model.fit(self._Y, self._T, X=self._X)
        return model

    def _fit_class_transformation(self, **kwargs: Any) -> Any:
        from econml.metalearners import TransformedOutcome
        from sklearn.ensemble import GradientBoostingRegressor

        model = TransformedOutcome(model=GradientBoostingRegressor(n_estimators=200, max_depth=3))
        model.fit(self._Y, self._T, X=self._X)
        return model

    def _fit_dr_learner(self, **kwargs: Any) -> Any:
        from econml.metalearners import DRLearner
        from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

        model = DRLearner(
            model_regression=GradientBoostingRegressor(n_estimators=200, max_depth=3),
            model_propensity=GradientBoostingClassifier(n_estimators=100),
        )
        model.fit(self._Y, self._T, X=self._X)
        return model

    def predict_uplift(self, X: np.ndarray | None = None) -> np.ndarray:
        """Predict individual uplift (CATE)."""
        if self._uplift_model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        if X is None:
            X = self._X
        return np.asarray(self._uplift_model.effect(X))

    def evaluate(
        self, X_test: np.ndarray, T_test: np.ndarray, Y_test: np.ndarray
    ) -> dict[str, float]:
        """Evaluate uplift model using Qini, AUUC, etc."""
        from causal_toolkit.analysis.uplift import evaluate_uplift

        uplift = self.predict_uplift(X_test)
        return dict(evaluate_uplift(uplift, T_test, Y_test))

    def plot_qini(self, X_test: np.ndarray, T_test: np.ndarray, Y_test: np.ndarray) -> Any:
        """Plot Qini curve."""
        from causal_toolkit.visualization.plots import UpliftPlot

        uplift = self.predict_uplift(X_test)
        return UpliftPlot().plot_qini(uplift, T_test, Y_test)


def create_econml_wrapper(
    data: pd.DataFrame,
    treatment: str,
    outcome: str,
    covariates: list[str],
    effect_modifiers: list[str] | None = None,
) -> EconMLWrapper:
    """Factory function to create EconMLWrapper."""
    return EconMLWrapper(data, treatment, outcome, covariates, effect_modifiers)
