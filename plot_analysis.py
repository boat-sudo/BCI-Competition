from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve,
    auc,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from scipy.stats import mannwhitneyu


# =========================
# Utils
# =========================

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


# 默认要画的特征
FEATURES_TO_PLOT = [
    "hr_mean_bpm",
    "rmssd_ms",
    "resp_rate_mean_bpm",
]


# =========================
# Statistics
# =========================

def save_basic_descriptive_stats(df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    desc = df.groupby("label_name").describe().transpose()
    out_path = out_dir / "descriptive_stats.csv"
    desc.to_csv(out_path, encoding="utf-8-sig")
    return desc


def save_group_difference_stats(df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    rows = []
    for feat in FEATURES_TO_PLOT:
        if feat not in df.columns:
            continue

        base = df[df["label_name"] == "baseline"][feat]
        stress = df[df["label_name"] == "stress"][feat]

        if len(base) == 0 or len(stress) == 0:
            continue

        stat, p = mannwhitneyu(base, stress, alternative="two-sided")
        rows.append({
            "feature": feat,
            "mean_diff_stress_minus_baseline": stress.mean() - base.mean(),
            "mannwhitney_u_p": p,
        })

    diff_df = pd.DataFrame(rows)
    out_path = out_dir / "group_difference_stats.csv"
    diff_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    return diff_df


# =========================
# Plotting
# =========================

def _save(fig, path: Path):
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_boxplot(df: pd.DataFrame, feature: str, out_dir: Path) -> None:
    fig, ax = plt.subplots()
    df.boxplot(column=feature, by="label_name", ax=ax)
    ax.set_title(f"Boxplot of {feature}")
    ax.set_ylabel(feature)
    fig.suptitle("")
    _save(fig, out_dir / f"boxplot_{feature}.png")


def plot_histogram(df: pd.DataFrame, feature: str, out_dir: Path) -> None:
    fig, ax = plt.subplots()
    for label in df["label_name"].unique():
        subset = df[df["label_name"] == label]
        ax.hist(subset[feature], bins=30, alpha=0.5, label=label)

    ax.set_title(f"Histogram of {feature}")
    ax.set_xlabel(feature)
    ax.set_ylabel("Count")
    ax.legend()
    _save(fig, out_dir / f"hist_{feature}.png")


def plot_subject_level_mean(df: pd.DataFrame, feature: str, out_dir: Path) -> None:
    grouped = df.groupby(["subject", "label_name"])[feature].mean().unstack()

    fig, ax = plt.subplots()
    grouped.plot(kind="bar", ax=ax)
    ax.set_title(f"Subject-level mean of {feature}")
    ax.set_ylabel(feature)
    _save(fig, out_dir / f"subject_mean_{feature}.png")


# =========================
# Model evaluation plots
# =========================

def plot_roc_for_best_model(preds_df: pd.DataFrame, best_model: str, out_dir: Path):
    df = preds_df[preds_df["model"] == best_model]

    if "y_true" not in df or "y_score" not in df:
        return

    y_true = df["y_true"].values
    y_score = df["y_score"].values

    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title(f"ROC - {best_model}")
    ax.legend()

    _save(fig, out_dir / f"roc_{best_model}.png")


def plot_confusion_for_best_model(preds_df: pd.DataFrame, best_model: str, out_dir: Path):
    df = preds_df[preds_df["model"] == best_model]

    if "y_true" not in df or "y_pred" not in df:
        return

    cm = confusion_matrix(df["y_true"], df["y_pred"])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)

    fig, ax = plt.subplots()
    disp.plot(ax=ax)
    ax.set_title(f"Confusion Matrix - {best_model}")

    _save(fig, out_dir / f"confusion_matrix_{best_model}.png")


def plot_subject_score_distribution(preds_df: pd.DataFrame, best_model: str, out_dir: Path):
    df = preds_df[preds_df["model"] == best_model]

    if "subject" not in df or "y_score" not in df:
        return

    grouped = df.groupby("subject")["y_score"].mean()

    fig, ax = plt.subplots()
    grouped.plot(kind="bar", ax=ax)
    ax.set_title(f"Subject Score Distribution - {best_model}")
    ax.set_ylabel("Mean Score")

    _save(fig, out_dir / f"subject_score_{best_model}.png")


# =========================
# Model selection
# =========================

def pick_best_model(metrics_df: pd.DataFrame) -> str:
    # 默认按 F1
    return metrics_df.sort_values("f1_mean", ascending=False).iloc[0]["model"]


# =========================
# Summary
# =========================

def write_text_summary(metrics_df: pd.DataFrame, diff_df: pd.DataFrame, best_model: str, out_dir: Path):
    out_path = out_dir / "analysis_summary.txt"

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("=== Model Performance ===\n")
        f.write(metrics_df.to_string())
        f.write("\n\n=== Group Differences ===\n")
        f.write(diff_df.to_string())
        f.write(f"\n\nBest model: {best_model}\n")
