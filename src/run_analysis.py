"""Run the complete analysis and save recruiter-friendly outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay

from credit_risk import (
    MODEL_FEATURES,
    build_models,
    evaluate_model,
    load_data,
    logistic_coefficients,
    temporal_split,
    threshold_strategy,
)


COLORS = {
    "logistic": "#0B5963",
    "logistic_balanced": "#D97706",
    "random_forest": "#7C3AED",
}


def save_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")


def style_axis(axis, title: str, xlabel: str, ylabel: str) -> None:
    axis.set_title(title, loc="left", fontsize=12, fontweight="bold", color="#12324A")
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(alpha=0.20)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/raw/verizon_data.csv")
    parser.add_argument("--output", default="reports")
    parser.add_argument("--docs", default="docs")
    args = parser.parse_args()

    output_dir = Path(args.output)
    docs_dir = Path(args.docs)
    output_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)

    frame = load_data(args.data)
    split = temporal_split(frame)
    X_train, y_train = split.train[MODEL_FEATURES], split.train["target"]
    X_test, y_test = split.test[MODEL_FEATURES], split.test["target"]

    models = build_models()
    metrics_rows = []
    probabilities: dict[str, np.ndarray] = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        metrics, probability = evaluate_model(model, X_test, y_test)
        metrics_rows.append({"model": name, **metrics})
        probabilities[name] = probability

    metrics_table = pd.DataFrame(metrics_rows).sort_values("roc_auc", ascending=False)
    metrics_table.to_csv(output_dir / "model_metrics.csv", index=False)

    coefficients = logistic_coefficients(models["logistic"])
    coefficients.to_csv(output_dir / "logistic_coefficients.csv", index=False)

    threshold_table = threshold_strategy(
        probabilities["logistic"],
        y_test,
        split.test["financed_amount"],
    )
    threshold_table.to_csv(output_dir / "threshold_strategy.csv", index=False)
    best_row = threshold_table.loc[threshold_table["estimated_total_value"].idxmax()]

    annual_default = (
        frame.assign(year=frame["application_date"].dt.year)
        .groupby("year", as_index=False)
        .agg(applications=("target", "size"), default_rate=("target", "mean"))
    )
    annual_default.to_csv(output_dir / "annual_default_rate.csv", index=False)

    data_profile = {
        "rows": len(frame),
        "columns": len(frame.columns),
        "source_columns": 7,
        "date_start": frame["application_date"].min().date(),
        "date_end": frame["application_date"].max().date(),
        "duplicate_rows": int(frame.duplicated().sum()),
        "missing_required_values": int(frame.isna().sum().sum()),
        "overall_default_rate": float(frame["target"].mean()),
        "train_rows": len(split.train),
        "test_rows": len(split.test),
        "train_default_rate": float(split.train["target"].mean()),
        "test_default_rate": float(split.test["target"].mean()),
        "oot_cutoff": split.cutoff.date(),
        "illustrative_economics": {
            "performing_margin_rate": 0.12,
            "loss_given_default": 0.70,
            "best_threshold": float(best_row["pd_threshold"]),
            "approval_rate": float(best_row["approval_rate"]),
            "approved_default_rate": float(best_row["default_rate_approved"]),
        },
    }
    save_json(output_dir / "data_profile.json", data_profile)

    plt.rcParams.update({"font.size": 9, "figure.facecolor": "white"})

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for name in ["logistic", "logistic_balanced", "random_forest"]:
        RocCurveDisplay.from_predictions(
            y_test,
            probabilities[name],
            name=name.replace("_", " ").title(),
            curve_kwargs={"color": COLORS[name]},
            ax=axes[0],
        )
        PrecisionRecallDisplay.from_predictions(
            y_test,
            probabilities[name],
            name=name.replace("_", " ").title(),
            color=COLORS[name],
            ax=axes[1],
        )
    style_axis(axes[0], "Out-of-time ROC comparison", "False positive rate", "True positive rate")
    style_axis(axes[1], "Out-of-time precision-recall comparison", "Recall", "Precision")
    fig.tight_layout()
    fig.savefig(docs_dir / "model_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(6.4, 4.4))
    for name in ["logistic", "logistic_balanced", "random_forest"]:
        observed, predicted = calibration_curve(
            y_test, probabilities[name], n_bins=10, strategy="quantile"
        )
        axis.plot(
            predicted,
            observed,
            marker="o",
            label=name.replace("_", " ").title(),
            color=COLORS[name],
        )
    axis.plot([0, 1], [0, 1], linestyle="--", color="#64748B", label="Ideal")
    style_axis(axis, "Probability calibration on the OOT sample", "Predicted default probability", "Observed default rate")
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(docs_dir / "calibration.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].bar(annual_default["year"].astype(str), annual_default["default_rate"], color="#0B5963")
    axes[0].axvline(4.5, color="#D97706", linestyle="--", linewidth=1)
    style_axis(axes[0], "Default rate changes over time", "Application year", "Default rate")
    axes[0].set_ylim(0, max(annual_default["default_rate"]) * 1.18)
    axes[0].yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")

    axes[1].plot(
        threshold_table["approval_rate"],
        threshold_table["default_rate_approved"],
        color="#0B5963",
        linewidth=2,
    )
    axes[1].scatter(
        [best_row["approval_rate"]],
        [best_row["default_rate_approved"]],
        color="#D97706",
        s=50,
        label="Highest illustrative value",
        zorder=3,
    )
    style_axis(axes[1], "Approval-risk trade-off", "Approval rate", "Default rate among approved")
    axes[1].xaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    axes[1].yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(docs_dir / "business_strategy.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    top_coefficients = (
        coefficients.query("feature != 'intercept'")
        .assign(abs_coefficient=lambda value: value["coefficient"].abs())
        .sort_values("abs_coefficient")
    )
    fig, axis = plt.subplots(figsize=(7.2, 4.6))
    colors = np.where(top_coefficients["coefficient"] < 0, "#0B5963", "#D97706")
    axis.barh(top_coefficients["feature"], top_coefficients["coefficient"], color=colors)
    axis.axvline(0, color="#64748B", linewidth=1)
    style_axis(axis, "Standardized logistic coefficients", "Coefficient", "Feature")
    fig.tight_layout()
    fig.savefig(docs_dir / "logistic_coefficients.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(metrics_table.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nData profile")
    print(json.dumps(data_profile, indent=2, default=str))


if __name__ == "__main__":
    main()
