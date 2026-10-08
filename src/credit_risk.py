"""Reusable modeling utilities for the device-financing credit-risk case study."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


EXPECTED_COLUMNS = {
    "application_date",
    "fico",
    "device_cost",
    "down_payment",
    "payment_type",
    "borrower_age",
    "del90",
}
TARGET_LABEL = "90+ delinquent/default"
NUMERIC_FEATURES = [
    "fico",
    "borrower_age",
    "device_cost",
    "down_payment",
    "dp_ratio",
]
CATEGORICAL_FEATURES = ["payment_type"]
MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


class QuantileClipper(BaseEstimator, TransformerMixin):
    """Clip each numeric feature to training-set quantile bounds."""

    def __init__(self, lower: float = 0.01, upper: float = 0.99):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        values = np.asarray(X, dtype=float)
        self.lower_bounds_ = np.nanquantile(values, self.lower, axis=0)
        self.upper_bounds_ = np.nanquantile(values, self.upper, axis=0)
        return self

    def transform(self, X):
        values = np.asarray(X, dtype=float)
        return np.clip(values, self.lower_bounds_, self.upper_bounds_)

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return np.asarray(
                [f"feature_{index}" for index in range(len(self.lower_bounds_))],
                dtype=object,
            )
        return np.asarray(input_features, dtype=object)


@dataclass(frozen=True)
class SplitData:
    train: pd.DataFrame
    test: pd.DataFrame
    cutoff: pd.Timestamp


def load_data(path: str | Path) -> pd.DataFrame:
    """Load, validate, and feature-engineer the academic case-study dataset."""
    frame = pd.read_csv(path)
    missing_columns = EXPECTED_COLUMNS.difference(frame.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")
    if frame[list(EXPECTED_COLUMNS)].isna().any().any():
        raise ValueError("Required fields contain missing values")

    frame = frame.copy()
    frame["application_date"] = pd.to_datetime(frame["application_date"], errors="raise")
    frame["target"] = (frame["del90"] == TARGET_LABEL).astype(int)
    invalid_targets = frame.loc[
        ~frame["del90"].isin([TARGET_LABEL, "Performing/paid off"]), "del90"
    ].unique()
    if len(invalid_targets):
        raise ValueError(f"Unexpected target labels: {invalid_targets.tolist()}")
    if (frame["device_cost"] <= 0).any():
        raise ValueError("device_cost must be positive")
    if (frame["down_payment"] < 0).any():
        raise ValueError("down_payment must be non-negative")
    if (frame["down_payment"] > frame["device_cost"]).any():
        raise ValueError("down_payment cannot exceed device_cost")

    frame["dp_ratio"] = frame["down_payment"] / frame["device_cost"]
    frame["financed_amount"] = frame["device_cost"] - frame["down_payment"]
    return frame.sort_values("application_date").reset_index(drop=True)


def temporal_split(frame: pd.DataFrame, cutoff: str = "2025-01-01") -> SplitData:
    """Create a strict out-of-time holdout split."""
    cutoff_timestamp = pd.Timestamp(cutoff)
    train = frame.loc[frame["application_date"] < cutoff_timestamp].copy()
    test = frame.loc[frame["application_date"] >= cutoff_timestamp].copy()
    if train.empty or test.empty:
        raise ValueError("Temporal split produced an empty train or test set")
    return SplitData(train=train, test=test, cutoff=cutoff_timestamp)


def build_preprocessor() -> ColumnTransformer:
    """Create leakage-safe numeric and categorical preprocessing."""
    numeric_pipeline = Pipeline(
        [
            ("clip", QuantileClipper(0.01, 0.99)),
            ("scale", StandardScaler()),
        ]
    )
    categorical_pipeline = OneHotEncoder(
        drop="first", handle_unknown="ignore", sparse_output=False
    )
    return ColumnTransformer(
        [
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        verbose_feature_names_out=False,
    )


def build_models() -> dict[str, Pipeline]:
    """Return interpretable baselines plus a nonlinear benchmark."""
    estimators = {
        "logistic": LogisticRegression(max_iter=2000, random_state=42),
        "logistic_balanced": LogisticRegression(
            class_weight="balanced", max_iter=2000, random_state=42
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=400,
            max_depth=6,
            min_samples_leaf=20,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
    }
    return {
        name: Pipeline(
            [("preprocess", build_preprocessor()), ("model", estimator)]
        )
        for name, estimator in estimators.items()
    }


def ks_statistic(y_true: pd.Series | np.ndarray, probability: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(y_true, probability)
    return float(np.max(tpr - fpr))


def evaluate_model(
    model: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    threshold: float = 0.50,
) -> tuple[dict[str, float], np.ndarray]:
    """Calculate discrimination, calibration, and classification metrics."""
    probability = model.predict_proba(X)[:, 1]
    prediction = (probability >= threshold).astype(int)
    metrics = {
        "roc_auc": float(roc_auc_score(y, probability)),
        "pr_auc": float(average_precision_score(y, probability)),
        "brier_score": float(brier_score_loss(y, probability)),
        "ks_statistic": ks_statistic(y, probability),
        "accuracy_at_0_5": float(accuracy_score(y, prediction)),
        "balanced_accuracy_at_0_5": float(balanced_accuracy_score(y, prediction)),
        "precision_at_0_5": float(precision_score(y, prediction, zero_division=0)),
        "recall_at_0_5": float(recall_score(y, prediction, zero_division=0)),
        "f1_at_0_5": float(f1_score(y, prediction, zero_division=0)),
    }
    return metrics, probability


def threshold_strategy(
    probability: np.ndarray,
    y_true: pd.Series,
    financed_amount: pd.Series,
    thresholds: np.ndarray | None = None,
    performing_margin_rate: float = 0.12,
    loss_given_default: float = 0.70,
) -> pd.DataFrame:
    """Evaluate approval thresholds under explicit illustrative economics."""
    if thresholds is None:
        thresholds = np.round(np.arange(0.05, 0.71, 0.01), 2)
    y = np.asarray(y_true, dtype=int)
    amount = np.asarray(financed_amount, dtype=float)
    rows = []
    for threshold in thresholds:
        approved = probability < threshold
        approved_count = int(approved.sum())
        if approved_count:
            realized_value = np.where(
                y[approved] == 0,
                amount[approved] * performing_margin_rate,
                -amount[approved] * loss_given_default,
            )
            default_rate = float(y[approved].mean())
            total_value = float(realized_value.sum())
            value_per_approved = float(realized_value.mean())
        else:
            default_rate = np.nan
            total_value = 0.0
            value_per_approved = np.nan
        rows.append(
            {
                "pd_threshold": float(threshold),
                "approved_accounts": approved_count,
                "approval_rate": float(approved.mean()),
                "default_rate_approved": default_rate,
                "estimated_total_value": total_value,
                "value_per_approved": value_per_approved,
            }
        )
    return pd.DataFrame(rows)


def logistic_coefficients(model: Pipeline) -> pd.DataFrame:
    """Return standardized logistic coefficients and odds ratios."""
    feature_names = model.named_steps["preprocess"].get_feature_names_out()
    coefficients = model.named_steps["model"].coef_[0]
    table = pd.DataFrame(
        {
            "feature": feature_names,
            "coefficient": coefficients,
            "odds_ratio": np.exp(coefficients),
        }
    )
    intercept = pd.DataFrame(
        {
            "feature": ["intercept"],
            "coefficient": [float(model.named_steps["model"].intercept_[0])],
            "odds_ratio": [np.nan],
        }
    )
    return pd.concat([intercept, table], ignore_index=True)
