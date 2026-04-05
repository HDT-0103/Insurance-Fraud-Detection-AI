from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Literal

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from insurance_project.features import add_age_and_tenure_groups


@dataclass(frozen=True)
class ThresholdTuningResult:
    """Result of threshold selection under a precision constraint."""

    threshold: float
    precision: float
    recall: float


class FraudPreprocessor(BaseEstimator, TransformerMixin):
    """
    Leakage-safe preprocessing for the fraud dataset.

    - Adds `age_group` and `tenure_group`
    - Ordinal-encodes `incident_severity`
    - One-hot encodes other categorical columns
    - Scales numeric columns with StandardScaler (fit only on train folds)
    - Aligns encoded columns across folds (fit captures training columns)
    """

    def __init__(
        self,
        *,
        drop_cols: Iterable[str] = ("policy_bind_date", "incident_date"),
        severity_col: str = "incident_severity",
        numeric_cols: Iterable[str] = (
            "months_as_customer",
            "age",
            "policy_annual_premium",
            "umbrella_limit",
            "capital-gains",
            "capital-loss",
            "incident_hour_of_the_day",
            "number_of_vehicles_involved",
            "bodily_injuries",
            "witnesses",
            "total_claim_amount",
            "injury_claim",
            "property_claim",
            "vehicle_claim",
            "days_to_incident",
            "policy_deductable",
        ),
        ohe_cols: Iterable[str] = (
            "policy_state",
            "policy_csl",
            "insured_sex",
            "insured_education_level",
            "insured_occupation",
            "insured_hobbies",
            "insured_relationship",
            "incident_type",
            "collision_type",
            "authorities_contacted",
            "incident_state",
            "incident_city",
            "property_damage",
            "police_report_available",
            "auto_make",
            "auto_model",
            "age_group",
            "tenure_group",
        ),
        drop_first: bool = True,
    ) -> None:
        self.drop_cols = tuple(drop_cols)
        self.severity_col = severity_col
        self.numeric_cols = tuple(numeric_cols)
        self.ohe_cols = tuple(ohe_cols)
        self.drop_first = drop_first

        self._severity_mapping = {
            "Trivial Damage": 0,
            "Minor Damage": 1,
            "Major Damage": 2,
            "Total Loss": 3,
        }
        self._encoded_columns: list[str] | None = None
        self._scaler: StandardScaler | None = None
        self._numeric_cols_present: list[str] | None = None

    def fit(self, X: pd.DataFrame, y: Any = None) -> "FraudPreprocessor":
        df = self._prepare_raw_df(X)
        df = self._encode(df, fit=True)

        numeric_cols_present = [c for c in self.numeric_cols if c in df.columns]
        self._numeric_cols_present = numeric_cols_present

        scaler = StandardScaler()
        if numeric_cols_present:
            scaler.fit(df[numeric_cols_present])
        self._scaler = scaler

        self._encoded_columns = list(df.columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self._encoded_columns is None or self._scaler is None or self._numeric_cols_present is None:
            raise ValueError("Preprocessor is not fitted. Call fit() before transform().")

        df = self._prepare_raw_df(X)
        df = self._encode(df, fit=False)
        df = df.reindex(columns=self._encoded_columns, fill_value=0)

        if self._numeric_cols_present:
            df[self._numeric_cols_present] = self._scaler.transform(df[self._numeric_cols_present])

        # Ensure everything is numeric for XGBoost/SMOTE
        return df.astype(float)

    def _prepare_raw_df(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("FraudPreprocessor expects a pandas DataFrame as input.")

        df = X.copy()
        drop = [c for c in self.drop_cols if c in df.columns]
        if drop:
            df = df.drop(columns=drop)

        df = add_age_and_tenure_groups(df)
        return df

    def _encode(self, df: pd.DataFrame, *, fit: bool) -> pd.DataFrame:
        df = df.copy()

        if self.severity_col in df.columns:
            df[self.severity_col] = df[self.severity_col].map(self._severity_mapping).astype(float)

        available_ohe = [c for c in self.ohe_cols if c in df.columns]
        df = pd.get_dummies(df, columns=available_ohe, drop_first=self.drop_first)

        bool_cols = df.select_dtypes(include=["bool"]).columns
        if len(bool_cols) > 0:
            df[bool_cols] = df[bool_cols].astype(int)

        # Any object columns left are not expected; coerce to numeric if possible.
        obj_cols = df.select_dtypes(include=["object"]).columns
        if len(obj_cols) > 0:
            for col in obj_cols:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        return df


def build_xgb_pipeline(*, random_state: int = 42) -> Pipeline:
    """
    Build an imblearn Pipeline that is safe for CV:
    preprocessing -> SMOTE -> XGBoost.
    """

    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        random_state=random_state,
        n_jobs=-1,
    )

    return Pipeline(
        steps=[
            ("prep", FraudPreprocessor()),
            ("smote", SMOTE(random_state=random_state)),
            ("model", model),
        ]
    )


def pr_auc_scorer(estimator: Any, X: pd.DataFrame, y: pd.Series) -> float:
    """Custom scorer for PR-AUC (average precision)."""
    proba = estimator.predict_proba(X)[:, 1]
    return float(average_precision_score(y, proba))


def run_xgb_random_search(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    random_state: int = 42,
    n_iter: int = 40,
    cv_splits: int = 5,
    n_jobs: int = -1,
    verbose: int = 1,
) -> RandomizedSearchCV:
    """
    RandomizedSearchCV over an XGBoost+SMOTE pipeline, optimizing PR-AUC.

    Notes:
    - SMOTE happens *inside* CV folds (no leakage).
    - Preprocessing is fit only on train folds (no leakage).
    """
    pipe = build_xgb_pipeline(random_state=random_state)
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)

    param_distributions: dict[str, list[Any]] = {
        "smote__k_neighbors": [3, 5, 7],
        "model__n_estimators": [300, 600, 900],
        "model__max_depth": [3, 4, 5, 6, 7, 8],
        "model__learning_rate": [0.01, 0.03, 0.05, 0.1],
        "model__subsample": [0.7, 0.8, 0.9, 1.0],
        "model__colsample_bytree": [0.7, 0.8, 0.9, 1.0],
        "model__min_child_weight": [1, 3, 5, 10],
        "model__gamma": [0.0, 0.5, 1.0, 5.0],
        "model__reg_alpha": [0.0, 0.01, 0.1, 1.0],
        "model__reg_lambda": [0.5, 1.0, 2.0, 5.0],
    }

    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring=pr_auc_scorer,
        refit=True,
        cv=cv,
        n_jobs=n_jobs,
        verbose=verbose,
        random_state=random_state,
        return_train_score=True,
    )
    search.fit(X, y)
    return search


def run_xgb_grid_search(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    random_state: int = 42,
    cv_splits: int = 5,
    n_jobs: int = -1,
    verbose: int = 1,
) -> GridSearchCV:
    """
    GridSearchCV over a smaller grid (still leakage-safe), optimizing PR-AUC.
    """
    pipe = build_xgb_pipeline(random_state=random_state)
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)

    param_grid: dict[str, list[Any]] = {
        "smote__k_neighbors": [5, 7],
        "model__n_estimators": [400, 800],
        "model__max_depth": [4, 6],
        "model__learning_rate": [0.03, 0.05],
        "model__subsample": [0.8, 1.0],
        "model__colsample_bytree": [0.8, 1.0],
        "model__min_child_weight": [1, 5],
        "model__reg_lambda": [1.0, 2.0],
    }

    search = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring=pr_auc_scorer,
        refit=True,
        cv=cv,
        n_jobs=n_jobs,
        verbose=verbose,
        return_train_score=True,
    )
    search.fit(X, y)
    return search


def choose_threshold_max_recall(
    y_true: np.ndarray | pd.Series,
    y_proba: np.ndarray,
    *,
    min_precision: float = 0.55,
) -> ThresholdTuningResult:
    """
    Pick a decision threshold that maximizes Recall subject to Precision >= min_precision.
    """
    y_true_arr = np.asarray(y_true).astype(int)
    precision, recall, thresholds = precision_recall_curve(y_true_arr, y_proba)

    # `thresholds` has length = len(precision)-1
    precision_t = precision[:-1]
    recall_t = recall[:-1]

    ok = precision_t >= min_precision
    if not np.any(ok):
        # No threshold satisfies constraint; fall back to 0.5 and report its metrics.
        thr = 0.5
        y_pred = (y_proba >= thr).astype(int)
        return ThresholdTuningResult(
            threshold=thr,
            precision=float(precision_score(y_true_arr, y_pred, zero_division=0)),
            recall=float(recall_score(y_true_arr, y_pred, zero_division=0)),
        )

    best_idx = int(np.argmax(recall_t[ok]))
    chosen_threshold = float(thresholds[np.where(ok)[0][best_idx]])
    y_pred = (y_proba >= chosen_threshold).astype(int)
    return ThresholdTuningResult(
        threshold=chosen_threshold,
        precision=float(precision_score(y_true_arr, y_pred, zero_division=0)),
        recall=float(recall_score(y_true_arr, y_pred, zero_division=0)),
    )


def predict_proba_positive(estimator: Any, X: pd.DataFrame) -> np.ndarray:
    """Convenience helper to get the positive-class probability."""
    return estimator.predict_proba(X)[:, 1]

