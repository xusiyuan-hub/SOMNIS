# -*- coding: utf-8 -*-
"""SOMNIS repository edition.

Command-line interface and input validation have been added around the
original model development, nested cross-fitting, bootstrap CI, DCA,
and visualization algorithms. Original numerical methods are unchanged.

For published reproducibility use --bootstrap 2000 --seed 42 and the
exact original locked discovery/independent cohorts. All model outputs
containing subject identifiers must remain private.
"""

# Leakage-safe nested-CV concatenated XGBoost baseline
# Early MSA vs PD

# Purpose
# -------
# This is the simple feature-concatenation baseline for comparison with the
# multimodal stacking model.

# All EEG + EOG + EMG features are concatenated into one 1390-feature vector
# and entered into a single XGBoost classifier.

# Fair-comparison settings shared with the current SOMNIS Stack model
# ----------------------------------------------------------------
# - SAME NPZ files and locked Discovery / Independent validation split.
# - SAME outer 5-fold stratified CV and RANDOM_SEED=42.
# - SAME XGBoost search space and fixed XGBoost settings.
# - SAME fold-local median imputation and z-score scaling.
# - SAME AUC-based hyperparameter-selection principle.
# - SAME threshold policy as the current SOMNIS Stack model:
#     1) Sens >= 0.75 and Spec >= 0.60 -> max BAC
#     2) if no threshold satisfies both constraints -> global max BAC
# - SAME stratified participant-level bootstrap settings and DCA range.
#
# Intentional architecture difference:
# - SOMNIS: EEG/EOG/EMG -> three modality XGBoost probabilities -> SVM fusion.
# - Concatenated: all EEG/EOG/EMG features -> one XGBoost classifier.
#
# The extra meta-SVM/Level-0 cross-fitting layers required by SOMNIS are not
# artificially added here. Outer validation and locked Independent validation
# remain the unbiased performance-comparison units.
# Input
# -----
# 

# Output
# ------
#     - clinical_performance_summary.png
#     - clinical_performance_summary.pdf
#     - model_performance_with_bootstrap_CI.csv
#     - predicted_probabilities_for_dca.csv
#     - final_discovery_threshold_operating_points.csv

# Important
# ---------
# No independent-validation subject is used for preprocessing, hyperparameter
# selection, threshold selection, model fitting, or internal validation.


import os
import argparse
import warnings
from pathlib import Path

# Avoid CPU oversubscription during nested model fitting.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)


# =============================================================================
# 0. Paths and analysis settings
# =============================================================================
# These are populated from CLI arguments immediately before each run.
MSA_FILE_PATH = None
PD_FILE_PATH = None
OUTPUT_DIR = None
RANDOM_SEED = 42
EXPECT_ORIGINAL_COHORT = False

# Strict nested-CV structure
N_OUTER_FOLDS = 5
N_INNER_TUNE_FOLDS = 5
N_TRAINING_OOF_FOLDS = 5

XGB_N_JOBS = 1

# Threshold policy: EXACTLY matched to the current SOMNIS Stack model.
#
# Rule:
# 1. Among thresholds with Sens >= 0.75 AND Spec >= 0.60, maximize BAC.
# 2. If no threshold satisfies both constraints, use the global maximum BAC.
#
# No additional sensitivity/specificity fallback levels and no weighted-Youden
# fallback are used. Thresholds are selected only from training-derived strict
# OOF probabilities; Independent-validation labels are never used.
MIN_SENSITIVITY = 0.75
MIN_SPECIFICITY = 0.60
# Bootstrap confidence intervals
N_BOOTSTRAP = 2000
BOOTSTRAP_SEED = 42

# Decision curve analysis range
DCA_MIN_THRESHOLD = 0.01
DCA_MAX_THRESHOLD = 0.70
DCA_STEP = 0.01

# Keep the identical XGBoost search space and fixed settings used by the
# stacking model's three Level-0 XGBoost base learners.
XGB_PARAM_GRID = [
    {"n_estimators": 50, "max_depth": 3, "learning_rate": 0.05},
    {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.10},
    {"n_estimators": 150, "max_depth": 5, "learning_rate": 0.05},
]

XGB_FIXED_PARAMS = {
    "subsample": 0.80,
    "colsample_bytree": 0.80,
    "min_child_weight": 1.0,
    "gamma": 0.0,
    "reg_alpha": 0.0,
    "reg_lambda": 1.0,
    "tree_method": "hist",
}


# =============================================================================
# 1. General data and split helpers
# =============================================================================
def as_clean_strings(values):
    """Convert bytes/object arrays to safely stripped strings."""
    output = []
    for value in np.asarray(values).ravel():
        if isinstance(value, bytes):
            value = value.decode("utf-8", errors="ignore")
        output.append(str(value).strip())
    return np.asarray(output, dtype=str)


def normalize_subject_ids(ids):
    """Conservative ID normalization for duplicate and overlap checks."""
    return np.asarray(
        [
            item.upper().replace(" ", "").replace("-", "_")
            for item in as_clean_strings(ids)
        ],
        dtype=str,
    )


def safe_auc(y_true, probabilities):
    """Return NaN rather than fail if only one class is present."""
    y_true = np.asarray(y_true, dtype=int)
    if len(np.unique(y_true)) < 2:
        return np.nan
    return float(roc_auc_score(y_true, probabilities))


def assert_cv_feasible(y, n_splits, context):
    """Ensure stratified CV is mathematically feasible."""
    y = np.asarray(y, dtype=int)
    counts = np.bincount(y, minlength=2)

    if np.any(counts == 0):
        raise ValueError(
            f"{context}: both classes are required; class counts={counts.tolist()}."
        )
    if int(np.min(counts)) < int(n_splits):
        raise ValueError(
            f"{context}: smallest class has {int(np.min(counts))} subjects, "
            f"but {n_splits}-fold stratification requires at least {n_splits}."
        )


def make_stratified_splits(y, n_splits, seed, context):
    """Create reproducible stratified folds after checking feasibility."""
    assert_cv_feasible(y, n_splits, context)
    splitter = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed,
    )
    return list(splitter.split(np.zeros(len(y)), np.asarray(y, dtype=int)))


def check_split_integrity(discovery_ids, independent_ids, y_discovery, y_independent):
    """Check for duplicate subjects and direct Discovery/test-set overlap."""
    discovery_ids = normalize_subject_ids(discovery_ids)
    independent_ids = normalize_subject_ids(independent_ids)

    duplicate_discovery = (
        pd.Series(discovery_ids)[pd.Series(discovery_ids).duplicated()].unique()
    )
    duplicate_independent = (
        pd.Series(independent_ids)[pd.Series(independent_ids).duplicated()].unique()
    )
    overlap = np.intersect1d(discovery_ids, independent_ids)

    if len(duplicate_discovery):
        raise ValueError(
            f"Duplicated IDs within Discovery set: "
            f"{duplicate_discovery[:10].tolist()}"
        )
    if len(duplicate_independent):
        raise ValueError(
            f"Duplicated IDs within Independent validation set: "
            f"{duplicate_independent[:10].tolist()}"
        )
    if len(overlap):
        raise ValueError(
            "DATA LEAKAGE DETECTED: IDs occur in both Discovery and "
            f"Independent validation sets: {overlap[:10].tolist()}"
        )

    assert_cv_feasible(y_discovery, N_OUTER_FOLDS, "Discovery set / outer CV")
    assert_cv_feasible(y_independent, 2, "Independent validation set")


# =============================================================================
# 2. Model fitting, threshold choice, and metrics
# =============================================================================
def fit_xgb_and_predict(X_fit, y_fit, X_predict, params):
    """
    Fit median imputation, standardization, and XGBoost only on X_fit,
    then predict probabilities for X_predict.

    Imputation, scaling, and class weight are therefore all fold-specific.
    """
    X_fit = np.asarray(X_fit, dtype=float)
    X_predict = np.asarray(X_predict, dtype=float)
    y_fit = np.asarray(y_fit, dtype=int)

    n_negative = int(np.sum(y_fit == 0))
    n_positive = int(np.sum(y_fit == 1))
    if n_negative == 0 or n_positive == 0:
        raise ValueError("Every XGBoost fit requires both classes.")

    imputer = SimpleImputer(strategy="median")
    X_fit_imputed = imputer.fit_transform(X_fit)
    X_predict_imputed = imputer.transform(X_predict)

    scaler = StandardScaler()
    X_fit_scaled = scaler.fit_transform(X_fit_imputed)
    X_predict_scaled = scaler.transform(X_predict_imputed)

    model = xgb.XGBClassifier(
        **XGB_FIXED_PARAMS,
        **params,
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=n_negative / n_positive,
        random_state=RANDOM_SEED,
        n_jobs=XGB_N_JOBS,
        verbosity=0,
    )
    model.fit(X_fit_scaled, y_fit)

    return model.predict_proba(X_predict_scaled)[:, 1]


def calculate_metrics(y_true, probabilities, threshold=None, predicted_labels=None):
    """
    Calculate probability-based AUC and threshold-dependent metrics.

    For pooled outer-fold OOF results, pass each subject's already assigned
    fold-specific prediction label instead of a common threshold.
    """
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)

    if predicted_labels is None:
        if threshold is None:
            raise ValueError("threshold is required unless predicted_labels are supplied.")
        predicted_labels = (probabilities >= float(threshold)).astype(int)
    else:
        predicted_labels = np.asarray(predicted_labels, dtype=int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predicted_labels,
        labels=[0, 1],
    ).ravel()

    sensitivity = recall_score(
        y_true,
        predicted_labels,
        pos_label=1,
        zero_division=0,
    )
    specificity = tn / (tn + fp) if (tn + fp) else np.nan
    ppv = precision_score(
        y_true,
        predicted_labels,
        pos_label=1,
        zero_division=0,
    )
    npv = tn / (tn + fn) if (tn + fn) else np.nan

    return {
        "AUC": safe_auc(y_true, probabilities),
        "ACC": float(accuracy_score(y_true, predicted_labels)),
        "BAC": float(balanced_accuracy_score(y_true, predicted_labels)),
        "Sens": float(sensitivity),
        "Spec": float(specificity),
        "PPV": float(ppv),
        "NPV": float(npv),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "predicted_labels": predicted_labels,
    }


def _roc_threshold_table(y_true, probabilities):
    """Return finite ROC operating points derived from training-only OOF values."""
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)

    fpr, tpr, thresholds = roc_curve(y_true, probabilities, pos_label=1)
    specificity = 1.0 - fpr
    bac = (tpr + specificity) / 2.0
    finite = np.isfinite(thresholds)

    table = pd.DataFrame(
        {
            "Threshold": thresholds[finite],
            "Sensitivity": tpr[finite],
            "Specificity": specificity[finite],
            "BAC": bac[finite],
        }
    )
    table["MeetsConstraint_0.75_0.60"] = (
        (table["Sensitivity"] >= MIN_SENSITIVITY)
        & (table["Specificity"] >= MIN_SPECIFICITY)
    )
    return table.sort_values("Threshold", ascending=False).reset_index(drop=True)


def select_threshold_from_training_oof(y_true, probabilities):
    """
    Select the classification threshold using training-only strict OOF probabilities.

    Rule
    ----
    1. Among all finite ROC thresholds satisfying:
           Sensitivity >= 0.75 AND Specificity >= 0.60
       choose the threshold with the maximum balanced accuracy (BAC).

    2. If no finite threshold satisfies both constraints, choose the global
       maximum-BAC threshold over all finite ROC operating points.

    Leakage control
    ---------------
    ``y_true`` and ``probabilities`` must come exclusively from training-derived
    strict OOF predictions. Outer-validation and Independent-validation labels
    must never be used here.

    Tie-breaking
    ------------
    Higher BAC -> higher specificity -> higher sensitivity -> higher threshold.
    """
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)

    fpr, tpr, thresholds = roc_curve(
        y_true,
        probabilities,
        pos_label=1,
    )
    specificity = 1.0 - fpr
    bac = (tpr + specificity) / 2.0
    finite = np.isfinite(thresholds)

    if not np.any(finite):
        return 0.50, "fallback_0.50"

    def choose_max_bac(candidate_idx):
        candidate_idx = np.asarray(candidate_idx, dtype=int)
        if len(candidate_idx) == 0:
            raise ValueError("choose_max_bac received an empty candidate set.")

        max_bac = np.max(bac[candidate_idx])
        tied = candidate_idx[np.isclose(bac[candidate_idx], max_bac)]

        best_idx = max(
            tied,
            key=lambda i: (
                float(specificity[i]),
                float(tpr[i]),
                float(thresholds[i]),
            ),
        )
        return int(best_idx)

    # Primary rule: Sens >= 0.75 and Spec >= 0.60, then maximum BAC.
    constrained_idx = np.where(
        finite
        & (tpr >= MIN_SENSITIVITY)
        & (specificity >= MIN_SPECIFICITY)
    )[0]

    if len(constrained_idx):
        best_idx = choose_max_bac(constrained_idx)
        return (
            float(thresholds[best_idx]),
            (
                f"Sens>={MIN_SENSITIVITY:.2f}_and_"
                f"Spec>={MIN_SPECIFICITY:.2f}_max_BAC"
            ),
        )

    # Fallback: no constrained operating point exists -> global maximum BAC.
    finite_idx = np.where(finite)[0]
    best_idx = choose_max_bac(finite_idx)
    return float(thresholds[best_idx]), "fallback_global_max_BAC"


# =============================================================================
# 3. Strict inner tuning and training-OOF probability construction
# =============================================================================
def tune_xgb_by_inner_cv_auc(X, y, seed, stage_label):
    """
    Tune candidate XGBoost configurations by mean 5-fold inner-CV AUC.

    Hyperparameter selection is threshold independent, while the classification
    threshold is selected separately from OOF probabilities.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)

    inner_splits = make_stratified_splits(
        y=y,
        n_splits=N_INNER_TUNE_FOLDS,
        seed=seed,
        context=f"{stage_label} / inner XGB tuning",
    )

    best_mean_auc = -np.inf
    best_params = None

    for params in XGB_PARAM_GRID:
        fold_auc = []

        for fit_idx, validation_idx in inner_splits:
            probability = fit_xgb_and_predict(
                X_fit=X[fit_idx],
                y_fit=y[fit_idx],
                X_predict=X[validation_idx],
                params=params,
            )
            fold_auc.append(safe_auc(y[validation_idx], probability))

        mean_auc = float(np.nanmean(fold_auc))

        # Preserve deterministic selection for exact ties.
        if mean_auc > best_mean_auc + 1e-12:
            best_mean_auc = mean_auc
            best_params = params.copy()

    return best_params, float(best_mean_auc)


def strict_training_oof_probabilities(X, y, seed, stage_label):
    """
    Create strict 5-fold cross-fitted probabilities for the current training set.

    For every OOF validation fold:
      1. tune XGBoost parameters solely within that fold's fit subset;
      2. fit preprocessing and XGBoost solely within that same fit subset;
      3. predict solely the held-out OOF validation subset.

    Thus, the label of a subject cannot affect parameter selection or model
    fitting for the model that produces that subject's OOF probability.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)

    oof_splits = make_stratified_splits(
        y=y,
        n_splits=N_TRAINING_OOF_FOLDS,
        seed=seed,
        context=f"{stage_label} / strict training OOF",
    )

    oof_probability = np.full(len(y), np.nan, dtype=float)

    for oof_fold, (fit_idx, validation_idx) in enumerate(oof_splits, start=1):
        params, _ = tune_xgb_by_inner_cv_auc(
            X=X[fit_idx],
            y=y[fit_idx],
            seed=seed + 10_000 * oof_fold,
            stage_label=f"{stage_label} | OOF_fold_{oof_fold}",
        )

        oof_probability[validation_idx] = fit_xgb_and_predict(
            X_fit=X[fit_idx],
            y_fit=y[fit_idx],
            X_predict=X[validation_idx],
            params=params,
        )

    if np.any(~np.isfinite(oof_probability)):
        raise RuntimeError(f"{stage_label}: incomplete strict OOF coverage.")

    return oof_probability


def fit_tuned_xgb_and_predict_holdout(X_train, y_train, X_holdout, seed, stage_label):
    """
    Tune XGBoost on the complete current training partition only, then fit it
    on all current training subjects and predict the held-out partition.
    """
    params, mean_inner_auc = tune_xgb_by_inner_cv_auc(
        X=X_train,
        y=y_train,
        seed=seed,
        stage_label=stage_label,
    )
    probability = fit_xgb_and_predict(
        X_fit=X_train,
        y_fit=y_train,
        X_predict=X_holdout,
        params=params,
    )
    return probability, params, mean_inner_auc


# =============================================================================
# 4. Stratified bootstrap 95% confidence intervals
# =============================================================================
def bootstrap_confidence_intervals(
    y_true,
    probabilities,
    predicted_labels,
    n_bootstrap=N_BOOTSTRAP,
    random_seed=BOOTSTRAP_SEED,
):
    """
    Stratified participant-level percentile bootstrap confidence intervals.

    PD and MSA are resampled separately with replacement to preserve the
    original class counts in every replicate. For Discovery-set results, the
    already assigned fold-specific predicted labels are resampled alongside the
    pooled outer-OOF probabilities.
    """
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    predicted_labels = np.asarray(predicted_labels, dtype=int)

    rng = np.random.default_rng(random_seed)
    pd_idx = np.where(y_true == 0)[0]
    msa_idx = np.where(y_true == 1)[0]

    if len(pd_idx) == 0 or len(msa_idx) == 0:
        raise ValueError("Bootstrap CI estimation requires both classes.")

    metric_names = ["AUC", "ACC", "BAC", "Sens", "Spec", "PPV", "NPV"]
    values = {name: [] for name in metric_names}

    for _ in range(n_bootstrap):
        sample_idx = np.concatenate(
            [
                rng.choice(pd_idx, size=len(pd_idx), replace=True),
                rng.choice(msa_idx, size=len(msa_idx), replace=True),
            ]
        )
        rng.shuffle(sample_idx)

        metrics = calculate_metrics(
            y_true=y_true[sample_idx],
            probabilities=probabilities[sample_idx],
            predicted_labels=predicted_labels[sample_idx],
        )

        for name in metric_names:
            values[name].append(metrics[name])

    ci = {}
    for name in metric_names:
        array = np.asarray(values[name], dtype=float)
        ci[name] = (
            float(np.nanpercentile(array, 2.5)),
            float(np.nanpercentile(array, 97.5)),
        )

    return ci


# =============================================================================
# 5. DCA and publication-style plotting
# =============================================================================
def decision_curve_net_benefit(y_true, probabilities, thresholds):
    """Calculate model, treat-all, and treat-none decision-curve net benefit."""
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)

    n_subjects = len(y_true)
    prevalence = float(np.mean(y_true == 1))

    model_net_benefit = []
    treat_all_net_benefit = []
    treat_none_net_benefit = np.zeros_like(thresholds, dtype=float)

    for threshold in thresholds:
        positive_prediction = probabilities >= threshold
        true_positive = np.sum((positive_prediction == 1) & (y_true == 1))
        false_positive = np.sum((positive_prediction == 1) & (y_true == 0))

        model_nb = (
            true_positive / n_subjects
            - (false_positive / n_subjects) * threshold / (1.0 - threshold)
        )
        treat_all_nb = prevalence - (1.0 - prevalence) * threshold / (1.0 - threshold)

        model_net_benefit.append(model_nb)
        treat_all_net_benefit.append(treat_all_nb)

    return (
        np.asarray(model_net_benefit, dtype=float),
        np.asarray(treat_all_net_benefit, dtype=float),
        treat_none_net_benefit,
    )


def draw_confusion_matrix(ax, matrix, title, cmap_name):
    """Render confusion matrix with count and row percentage labels."""
    image = ax.imshow(matrix, cmap=cmap_name, vmin=0)
    ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks([0, 1], ["PD", "MSA"])
    ax.set_yticks([0, 1], ["PD", "MSA"])
    ax.set_xlabel("Predicted diagnosis")
    ax.set_ylabel("True diagnosis")
    ax.set_title(title, fontweight="bold", pad=10)

    row_total = matrix.sum(axis=1, keepdims=True)
    row_proportion = np.divide(
        matrix,
        row_total,
        out=np.zeros_like(matrix, dtype=float),
        where=row_total != 0,
    )

    max_value = matrix.max() if matrix.size else 0
    for row in range(2):
        for col in range(2):
            color = "white" if matrix[row, col] > max_value * 0.55 else "black"
            ax.text(
                col,
                row,
                f"{int(matrix[row, col])}\n({row_proportion[row, col] * 100:.1f}%)",
                ha="center",
                va="center",
                fontsize=11,
                color=color,
                fontweight="bold",
            )


def format_ci(value, ci_pair):
    return f"{value:.3f} ({ci_pair[0]:.3f}–{ci_pair[1]:.3f})"


def save_clinical_performance_figure(
    y_discovery,
    discovery_probability,
    discovery_label,
    y_independent,
    independent_probability,
    independent_label,
    discovery_metrics,
    independent_metrics,
    discovery_ci,
    independent_ci,
):
    """
    Save a publication-style 2 x 3 figure:
      Row 1: Discovery set pooled outer-OOF performance (green)
      Row 2: Independent validation-set performance (blue)
      Columns: ROC, confusion matrix, DCA
    """
    thresholds = np.arange(
        DCA_MIN_THRESHOLD,
        DCA_MAX_THRESHOLD + DCA_STEP / 2,
        DCA_STEP,
    )

    fpr_discovery, tpr_discovery, _ = roc_curve(y_discovery, discovery_probability)
    fpr_independent, tpr_independent, _ = roc_curve(
        y_independent,
        independent_probability,
    )

    dca_discovery, all_discovery, none_discovery = decision_curve_net_benefit(
        y_discovery,
        discovery_probability,
        thresholds,
    )
    dca_independent, all_independent, none_independent = decision_curve_net_benefit(
        y_independent,
        independent_probability,
        thresholds,
    )

    green = "#2E8B57"
    green_light = "#E9F6EE"
    blue = "#2878B9"
    blue_light = "#EAF3FB"
    gray = "#77818C"
    dark = "#1F2937"

    fig, axes = plt.subplots(2, 3, figsize=(19, 11))
    fig.patch.set_facecolor("white")

    # Discovery ROC
    ax = axes[0, 0]
    ax.plot(fpr_discovery, tpr_discovery, color=green, lw=2.8, label="Model")
    ax.plot([0, 1], [0, 1], linestyle="--", color=gray, lw=1.2)
    ax.fill_between(fpr_discovery, tpr_discovery, 0, color=green, alpha=0.08)
    ax.set_xlim([-0.02, 1.0])
    ax.set_ylim([0.0, 1.03])
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Discovery Set: ROC Curve", fontweight="bold", pad=10)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, loc="lower right")
    ax.text(
        0.04,
        0.07,
        "AUC (95% CI)\n" + format_ci(discovery_metrics["AUC"], discovery_ci["AUC"]),
        transform=ax.transAxes,
        fontsize=11,
        color=dark,
        bbox={
            "boxstyle": "round,pad=0.45",
            "facecolor": green_light,
            "edgecolor": green,
            "alpha": 0.98,
        },
    )

    # Discovery confusion matrix
    draw_confusion_matrix(
        axes[0, 1],
        confusion_matrix(y_discovery, discovery_label, labels=[0, 1]),
        "Discovery Set: Outer-OOF Confusion Matrix",
        "Greens",
    )

    # Discovery DCA
    ax = axes[0, 2]
    ax.plot(thresholds, dca_discovery, color=green, lw=2.8, label="Model")
    ax.plot(
        thresholds,
        all_discovery,
        color="#9AA4AF",
        lw=1.7,
        linestyle="--",
        label="Treat all",
    )
    ax.plot(
        thresholds,
        none_discovery,
        color="#3E4854",
        lw=1.5,
        linestyle=":",
        label="Treat none",
    )
    ax.axvspan(0.10, DCA_MAX_THRESHOLD, color=green, alpha=0.035)
    ax.set_xlim(DCA_MIN_THRESHOLD, DCA_MAX_THRESHOLD)
    ax.set_xlabel("Threshold probability")
    ax.set_ylabel("Net benefit")
    ax.set_title("Discovery Set: Decision Curve", fontweight="bold", pad=10)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, loc="upper right")

    # Independent ROC
    ax = axes[1, 0]
    ax.plot(fpr_independent, tpr_independent, color=blue, lw=2.8, label="Model")
    ax.plot([0, 1], [0, 1], linestyle="--", color=gray, lw=1.2)
    ax.fill_between(fpr_independent, tpr_independent, 0, color=blue, alpha=0.08)
    ax.set_xlim([-0.02, 1.0])
    ax.set_ylim([0.0, 1.03])
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(
        "Independent Validation Set: ROC Curve",
        fontweight="bold",
        pad=10,
    )
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, loc="lower right")
    ax.text(
        0.04,
        0.07,
        "AUC (95% CI)\n"
        + format_ci(independent_metrics["AUC"], independent_ci["AUC"]),
        transform=ax.transAxes,
        fontsize=11,
        color=dark,
        bbox={
            "boxstyle": "round,pad=0.45",
            "facecolor": blue_light,
            "edgecolor": blue,
            "alpha": 0.98,
        },
    )

    # Independent confusion matrix
    draw_confusion_matrix(
        axes[1, 1],
        confusion_matrix(y_independent, independent_label, labels=[0, 1]),
        "Independent Validation Set: Confusion Matrix",
        "Blues",
    )

    # Independent DCA
    ax = axes[1, 2]
    ax.plot(thresholds, dca_independent, color=blue, lw=2.8, label="Model")
    ax.plot(
        thresholds,
        all_independent,
        color="#9AA4AF",
        lw=1.7,
        linestyle="--",
        label="Treat all",
    )
    ax.plot(
        thresholds,
        none_independent,
        color="#3E4854",
        lw=1.5,
        linestyle=":",
        label="Treat none",
    )
    ax.axvspan(0.10, DCA_MAX_THRESHOLD, color=blue, alpha=0.035)
    ax.set_xlim(DCA_MIN_THRESHOLD, DCA_MAX_THRESHOLD)
    ax.set_xlabel("Threshold probability")
    ax.set_ylabel("Net benefit")
    ax.set_title(
        "Independent Validation Set: Decision Curve",
        fontweight="bold",
        pad=10,
    )
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, loc="upper right")

    fig.text(
        0.01,
        0.75,
        "Discovery set\n(outer OOF)",
        va="center",
        ha="left",
        fontsize=14,
        fontweight="bold",
        color=green,
        rotation=90,
    )
    fig.text(
        0.01,
        0.26,
        "Independent validation set",
        va="center",
        ha="left",
        fontsize=14,
        fontweight="bold",
        color=blue,
        rotation=90,
    )

    fig.suptitle(
        "Clinical Performance of the Concatenated XGBoost Baseline",
        fontsize=18,
        fontweight="bold",
        y=0.995,
    )
    plt.tight_layout(rect=[0.035, 0.02, 1, 0.97])

    fig.savefig(
        OUTPUT_DIR / "clinical_performance_summary.png",
        dpi=100,
        bbox_inches="tight",
        facecolor="white",
    )
    fig.savefig(
        OUTPUT_DIR / "clinical_performance_summary.pdf",
        dpi=100,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


# =============================================================================
# 6. Data loading
# =============================================================================
def load_data():
    """Load NPZ data and form locked Discovery / Independent-validation sets."""
    for file_path, label in [(MSA_FILE_PATH, "MSA"), (PD_FILE_PATH, "PD")]:
        if not file_path.exists():
            raise FileNotFoundError(f"{label} file not found: {file_path}")

    msa_data = np.load(MSA_FILE_PATH, allow_pickle=True)
    pd_data = np.load(PD_FILE_PATH, allow_pickle=True)

    required_keys = {"X", "channels", "dataset", "subject_ids"}
    missing_msa = required_keys.difference(msa_data.files)
    missing_pd = required_keys.difference(pd_data.files)

    if missing_msa:
        raise KeyError(f"earlyMSA.npz missing keys: {sorted(missing_msa)}")
    if missing_pd:
        raise KeyError(f"earlyPD.npz missing keys: {sorted(missing_pd)}")

    msa_X = np.asarray(msa_data["X"], dtype=float)
    pd_X = np.asarray(pd_data["X"], dtype=float)

    msa_channels = as_clean_strings(msa_data["channels"])
    pd_channels = as_clean_strings(pd_data["channels"])
    msa_dataset = np.char.lower(as_clean_strings(msa_data["dataset"]))
    pd_dataset = np.char.lower(as_clean_strings(pd_data["dataset"]))
    msa_ids = np.asarray(msa_data["subject_ids"])
    pd_ids = np.asarray(pd_data["subject_ids"])

    if msa_X.ndim != 2 or pd_X.ndim != 2:
        raise ValueError("X arrays must be 2-dimensional: subjects × features.")
    if msa_X.shape[1] != pd_X.shape[1]:
        raise ValueError("MSA and PD feature dimensions do not match.")
    if not np.array_equal(msa_channels, pd_channels):
        raise ValueError("MSA and PD channels/modality annotations do not match.")
    if len(msa_channels) != msa_X.shape[1]:
        raise ValueError("MSA channels length does not match MSA feature count.")
    if len(pd_channels) != pd_X.shape[1]:
        raise ValueError("PD channels length does not match PD feature count.")

    msa_discovery = msa_dataset == "train"
    msa_independent = msa_dataset == "test"
    pd_discovery = pd_dataset == "train"
    pd_independent = pd_dataset == "test"

    if not np.all(msa_discovery | msa_independent):
        raise ValueError("MSA dataset labels must be only 'train' or 'test'.")
    if not np.all(pd_discovery | pd_independent):
        raise ValueError("PD dataset labels must be only 'train' or 'test'.")

    # The three modalities are concatenated by simply retaining every feature
    # in their original common feature-column order.
    X_discovery = np.vstack([msa_X[msa_discovery], pd_X[pd_discovery]])
    X_independent = np.vstack([msa_X[msa_independent], pd_X[pd_independent]])

    y_discovery = np.hstack(
        [
            np.ones(np.sum(msa_discovery), dtype=int),
            np.zeros(np.sum(pd_discovery), dtype=int),
        ]
    )
    y_independent = np.hstack(
        [
            np.ones(np.sum(msa_independent), dtype=int),
            np.zeros(np.sum(pd_independent), dtype=int),
        ]
    )

    discovery_ids = np.concatenate([msa_ids[msa_discovery], pd_ids[pd_discovery]])
    independent_ids = np.concatenate(
        [msa_ids[msa_independent], pd_ids[pd_independent]]
    )

    check_split_integrity(
        discovery_ids=discovery_ids,
        independent_ids=independent_ids,
        y_discovery=y_discovery,
        y_independent=y_independent,
    )

    return {
        "X_discovery": X_discovery,
        "X_independent": X_independent,
        "y_discovery": y_discovery,
        "y_independent": y_independent,
        "discovery_ids": discovery_ids,
        "independent_ids": independent_ids,
        "n_features": int(X_discovery.shape[1]),
    }


# =============================================================================
# 7. Main analysis
# =============================================================================
def print_metrics_with_ci(dataset_name, metrics, ci):
    """Print all reported performance metrics with bootstrap 95% CIs."""
    print(
        f"{dataset_name}:\n"
        f"  AUC  = {metrics['AUC']:.4f} "
        f"({ci['AUC'][0]:.4f}–{ci['AUC'][1]:.4f})\n"
        f"  ACC  = {metrics['ACC']:.4f} "
        f"({ci['ACC'][0]:.4f}–{ci['ACC'][1]:.4f})\n"
        f"  BAC  = {metrics['BAC']:.4f} "
        f"({ci['BAC'][0]:.4f}–{ci['BAC'][1]:.4f})\n"
        f"  Sens = {metrics['Sens']:.4f} "
        f"({ci['Sens'][0]:.4f}–{ci['Sens'][1]:.4f})\n"
        f"  Spec = {metrics['Spec']:.4f} "
        f"({ci['Spec'][0]:.4f}–{ci['Spec'][1]:.4f})\n"
        f"  PPV  = {metrics['PPV']:.4f} "
        f"({ci['PPV'][0]:.4f}–{ci['PPV'][1]:.4f})\n"
        f"  NPV  = {metrics['NPV']:.4f} "
        f"({ci['NPV'][0]:.4f}–{ci['NPV'][1]:.4f})",
        flush=True,
    )


def main():
    print("=" * 78, flush=True)
    print("Leakage-safe nested-CV concatenated XGBoost baseline: Early MSA vs PD")
    print(
        "AUC-driven tuning + SOMNIS-matched Sens>=0.75 / Spec>=0.60 constrained max-BAC threshold",
        flush=True,
    )
    print("=" * 78, flush=True)

    data = load_data()

    X_discovery = data["X_discovery"]
    X_independent = data["X_independent"]
    y_discovery = data["y_discovery"]
    y_independent = data["y_independent"]
    discovery_ids = data["discovery_ids"]
    independent_ids = data["independent_ids"]

    print(
        f"Discovery set: {X_discovery.shape}; "
        f"MSA={np.sum(y_discovery == 1)}, PD={np.sum(y_discovery == 0)}",
        flush=True,
    )
    print(
        f"Independent validation set: {X_independent.shape}; "
        f"MSA={np.sum(y_independent == 1)}, PD={np.sum(y_independent == 0)}",
        flush=True,
    )
    print(
        f"Concatenated features: {data['n_features']}",
        flush=True,
    )
    print("Integrity check passed: no duplicate or overlapping normalized IDs.", flush=True)
    print(
        "Threshold policy: Sens>=0.75 & Spec>=0.60 -> max BAC; "
        "if infeasible -> global max BAC.",
        flush=True,
    )

    # -------------------------------------------------------------------------
    # A. Discovery set: outer 5-fold nested validation
    # -------------------------------------------------------------------------
    outer_splits = make_stratified_splits(
        y=y_discovery,
        n_splits=N_OUTER_FOLDS,
        seed=RANDOM_SEED,
        context="Discovery set / outer 5-fold CV",
    )

    outer_oof_probability = np.full(len(y_discovery), np.nan, dtype=float)
    outer_oof_label = np.full(len(y_discovery), -1, dtype=int)
    outer_fold_id = np.full(len(y_discovery), -1, dtype=int)
    outer_fold_threshold = np.full(len(y_discovery), np.nan, dtype=float)

    print("\nOuter 5-fold nested CV", flush=True)

    for outer_fold, (outer_train_idx, outer_validation_idx) in enumerate(
        outer_splits,
        start=1,
    ):
        print(f"  Outer fold {outer_fold}/{N_OUTER_FOLDS}", flush=True)

        X_outer_train = X_discovery[outer_train_idx]
        y_outer_train = y_discovery[outer_train_idx]
        X_outer_validation = X_discovery[outer_validation_idx]
        y_outer_validation = y_discovery[outer_validation_idx]

        # Strict OOF predictions from outer-training subjects provide a
        # training-only basis for threshold selection.
        outer_train_oof_probability = strict_training_oof_probabilities(
            X=X_outer_train,
            y=y_outer_train,
            seed=RANDOM_SEED + 10_000_000 * outer_fold,
            stage_label=f"outer_fold_{outer_fold}_threshold_OOF",
        )
        threshold, _ = select_threshold_from_training_oof(
            y_true=y_outer_train,
            probabilities=outer_train_oof_probability,
        )

        # Select XGB by inner 5-fold CV AUC on all outer-training subjects,
        # then fit on all outer-training subjects and predict untouched fold.
        outer_validation_probability, _, _ = fit_tuned_xgb_and_predict_holdout(
            X_train=X_outer_train,
            y_train=y_outer_train,
            X_holdout=X_outer_validation,
            seed=RANDOM_SEED + 20_000_000 * outer_fold,
            stage_label=f"outer_fold_{outer_fold}_final_fit",
        )
        outer_validation_metrics = calculate_metrics(
            y_true=y_outer_validation,
            probabilities=outer_validation_probability,
            threshold=threshold,
        )

        outer_oof_probability[outer_validation_idx] = outer_validation_probability
        outer_oof_label[outer_validation_idx] = (
            outer_validation_metrics["predicted_labels"]
        )
        outer_fold_id[outer_validation_idx] = outer_fold
        outer_fold_threshold[outer_validation_idx] = threshold

        print(
            f"    AUC={outer_validation_metrics['AUC']:.3f}; "
            f"BAC={outer_validation_metrics['BAC']:.3f}; "
            f"Sens={outer_validation_metrics['Sens']:.3f}; "
            f"Spec={outer_validation_metrics['Spec']:.3f}; "
            f"threshold={threshold:.3f}",
            flush=True,
        )

    if np.any(~np.isfinite(outer_oof_probability)) or np.any(outer_oof_label < 0):
        raise RuntimeError("Incomplete outer-fold OOF prediction coverage.")

    discovery_metrics = calculate_metrics(
        y_true=y_discovery,
        probabilities=outer_oof_probability,
        predicted_labels=outer_oof_label,
    )

    # -------------------------------------------------------------------------
    # B. Final development on complete Discovery set, then locked evaluation
    # -------------------------------------------------------------------------
    print("\nFinal development on complete Discovery set", flush=True)

    final_training_oof_probability = strict_training_oof_probabilities(
        X=X_discovery,
        y=y_discovery,
        seed=RANDOM_SEED + 900_000_000,
        stage_label="final_discovery_threshold_OOF",
    )
    final_threshold, final_threshold_rule = select_threshold_from_training_oof(
        y_true=y_discovery,
        probabilities=final_training_oof_probability,
    )

    final_threshold_audit = _roc_threshold_table(
        y_true=y_discovery,
        probabilities=final_training_oof_probability,
    )
    final_threshold_audit["Selected"] = np.isclose(
        final_threshold_audit["Threshold"].to_numpy(dtype=float),
        float(final_threshold),
    )
    final_threshold_audit["SelectedRule"] = np.where(
        final_threshold_audit["Selected"],
        final_threshold_rule,
        "",
    )
    final_threshold_audit.to_csv(
        OUTPUT_DIR / "final_discovery_threshold_operating_points.csv",
        index=False,
        encoding="utf-8-sig",
    )

    independent_probability, final_xgb_params, final_inner_cv_auc = (
        fit_tuned_xgb_and_predict_holdout(
            X_train=X_discovery,
            y_train=y_discovery,
            X_holdout=X_independent,
            seed=RANDOM_SEED + 950_000_000,
            stage_label="final_discovery_fit",
        )
    )
    independent_metrics = calculate_metrics(
        y_true=y_independent,
        probabilities=independent_probability,
        threshold=final_threshold,
    )

    # -------------------------------------------------------------------------
    # C. Stratified bootstrap 95% CIs
    # -------------------------------------------------------------------------
    print(
        f"\nComputing stratified bootstrap 95% CIs ({N_BOOTSTRAP} resamples)",
        flush=True,
    )

    discovery_ci = bootstrap_confidence_intervals(
        y_true=y_discovery,
        probabilities=outer_oof_probability,
        predicted_labels=outer_oof_label,
        n_bootstrap=N_BOOTSTRAP,
        random_seed=BOOTSTRAP_SEED,
    )
    independent_ci = bootstrap_confidence_intervals(
        y_true=y_independent,
        probabilities=independent_probability,
        predicted_labels=independent_metrics["predicted_labels"],
        n_bootstrap=N_BOOTSTRAP,
        random_seed=BOOTSTRAP_SEED + 1,
    )

    # -------------------------------------------------------------------------
    # D. Save the concise comparison outputs
    # -------------------------------------------------------------------------
    save_clinical_performance_figure(
        y_discovery=y_discovery,
        discovery_probability=outer_oof_probability,
        discovery_label=outer_oof_label,
        y_independent=y_independent,
        independent_probability=independent_probability,
        independent_label=independent_metrics["predicted_labels"],
        discovery_metrics=discovery_metrics,
        independent_metrics=independent_metrics,
        discovery_ci=discovery_ci,
        independent_ci=independent_ci,
    )

    metric_names = ["AUC", "ACC", "BAC", "Sens", "Spec", "PPV", "NPV"]
    performance_rows = []

    for dataset_name, metrics, ci, threshold, threshold_rule in [
        (
            "Discovery set (pooled outer-fold OOF)",
            discovery_metrics,
            discovery_ci,
            "fold-specific",
            "selected separately from each outer-training partition",
        ),
        (
            "Independent validation set",
            independent_metrics,
            independent_ci,
            final_threshold,
            final_threshold_rule,
        ),
    ]:
        row = {
            "Model": "Concatenated XGBoost",
            "Dataset": dataset_name,
            "TN": metrics["TN"],
            "FP": metrics["FP"],
            "FN": metrics["FN"],
            "TP": metrics["TP"],
            "Threshold": threshold,
            "Threshold_selection_rule": threshold_rule,
        }
        for metric in metric_names:
            row[metric] = metrics[metric]
            row[f"{metric}_CI_lower"] = ci[metric][0]
            row[f"{metric}_CI_upper"] = ci[metric][1]
        performance_rows.append(row)

    pd.DataFrame(performance_rows).to_csv(
        OUTPUT_DIR / "model_performance_with_bootstrap_CI.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # Use this output for later DCA, calibration, or combined model comparison.
    prediction_df = pd.concat(
        [
            pd.DataFrame(
                {
                    "Model": "Concatenated XGBoost",
                    "Dataset": "Discovery set (outer-fold OOF)",
                    "SubjectID": normalize_subject_ids(discovery_ids),
                    "TrueLabel": y_discovery,
                    "MSA_probability": outer_oof_probability,
                    "PredictedLabel": outer_oof_label,
                    "AppliedThreshold": outer_fold_threshold,
                    "OuterFold": outer_fold_id,
                }
            ),
            pd.DataFrame(
                {
                    "Model": "Concatenated XGBoost",
                    "Dataset": "Independent validation set",
                    "SubjectID": normalize_subject_ids(independent_ids),
                    "TrueLabel": y_independent,
                    "MSA_probability": independent_probability,
                    "PredictedLabel": independent_metrics["predicted_labels"],
                    "AppliedThreshold": final_threshold,
                    "OuterFold": np.nan,
                }
            ),
        ],
        ignore_index=True,
    )
    prediction_df.to_csv(
        OUTPUT_DIR / "predicted_probabilities_for_dca.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # -------------------------------------------------------------------------
    # E. Console summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 78, flush=True)
    print("Results", flush=True)
    print("=" * 78, flush=True)

    print_metrics_with_ci(
        "Discovery set (pooled outer-OOF)",
        discovery_metrics,
        discovery_ci,
    )
    print_metrics_with_ci(
        "Independent validation set",
        independent_metrics,
        independent_ci,
    )

    print(
        f"Final constrained threshold = {final_threshold:.4f} "
        f"({final_threshold_rule})",
        flush=True,
    )
    print(
        "Selected XGBoost internal tuning performance "
        f"(mean 5-fold inner-CV AUC) = {final_inner_cv_auc:.4f}; "
        f"params = {final_xgb_params}",
        flush=True,
    )

    print("\nSaved outputs:", flush=True)
    print(f"  {OUTPUT_DIR / 'clinical_performance_summary.png'}", flush=True)
    print(f"  {OUTPUT_DIR / 'clinical_performance_summary.pdf'}", flush=True)
    print(f"  {OUTPUT_DIR / 'model_performance_with_bootstrap_CI.csv'}", flush=True)
    print(f"  {OUTPUT_DIR / 'predicted_probabilities_for_dca.csv'}", flush=True)
    print(f"  {OUTPUT_DIR / 'final_discovery_threshold_operating_points.csv'}", flush=True)





def validate_input_feature_schema(msa_file, pd_file, representation):
    """Reject mismatched feature meanings, dimensions, or study allocations.

    Validation never modifies features or the train/test assignment.
    """
    expected_dim = 1390 if representation == "stages" else 278
    expected_modality_counts = (
        {"EEG": 990, "EOG": 240, "EMG": 160}
        if representation == "stages"
        else {"EEG": 198, "EOG": 48, "EMG": 32}
    )
    info = {}
    for group, path in (("MSA", msa_file), ("PD", pd_file)):
        if not path.is_file():
            raise FileNotFoundError(f"Missing {group} input: {path}")
        with np.load(path, allow_pickle=True) as npz:
            required = {"X", "channels", "subject_ids", "dataset", "feature_names"}
            missing = required.difference(npz.files)
            if missing:
                raise KeyError(f"{path}: missing required keys: {sorted(missing)}")
            X = np.asarray(npz["X"], dtype=float)
            features = as_clean_strings(npz["feature_names"])
            modalities = np.char.upper(as_clean_strings(npz["channels"]))
            identifiers = normalize_subject_ids(npz["subject_ids"])
            partition = np.char.lower(as_clean_strings(npz["dataset"]))
        if X.ndim != 2 or X.shape[1] != expected_dim:
            raise ValueError(f"{path}: expected X.shape=(n,{expected_dim}), got {X.shape}")
        if not (X.shape[0] == len(identifiers) == len(partition)):
            raise ValueError(f"{path}: subject/dataset row count mismatch")
        if not (len(features) == len(modalities) == expected_dim):
            raise ValueError(f"{path}: feature names / channels count mismatch")
        if len(set(features)) != len(features):
            raise ValueError(f"{path}: duplicate feature names")
        if len(set(identifiers)) != len(identifiers):
            raise ValueError(f"{path}: duplicate subject IDs")
        if not set(partition).issubset({"train", "test"}):
            raise ValueError(f"{path}: dataset must contain only 'train' or 'test'")
        if not {"train", "test"}.issubset(set(partition)):
            raise ValueError(f"{path}: both train and test subjects are required")
        if np.isinf(X).any():
            raise ValueError(f"{path}: X contains infinity (NaN can be imputed fold-locally)")
        counts = {m: int((modalities == m).sum()) for m in expected_modality_counts}
        if counts != expected_modality_counts:
            raise ValueError(f"{path}: expected modality counts {expected_modality_counts}, got {counts}")
        # Keep the manuscript's original fixed discovery/independent allocation.
        if EXPECT_ORIGINAL_COHORT:
            expected = {"MSA": (77, 33), "PD": (64, 28)}[group]
            actual = (int((partition == "train").sum()),
                      int((partition == "test").sum()))
            if actual != expected:
                raise ValueError(f"{path}: expected study train/test counts {expected}; got {actual}")
        info[group] = {"features": features, "modalities": modalities,
                       "ids": identifiers, "partition": partition}
    if not np.array_equal(info["MSA"]["features"], info["PD"]["features"]):
        raise ValueError("MSA/PD feature_names (including column order) do not match")
    if not np.array_equal(info["MSA"]["modalities"], info["PD"]["modalities"]):
        raise ValueError("MSA/PD modality labels (including column order) do not match")
    all_ids = np.concatenate([info["MSA"]["ids"], info["PD"]["ids"]])
    if len(set(all_ids)) != len(all_ids):
        raise ValueError("Subject IDs overlap across MSA and PD")
    return info


def _resolve_paths(args, mode):
    if args.msa_file is not None:
        if args.mode == "both":
            raise ValueError("--msa-file/--pd-file require a single --mode, not 'both'")
        return args.msa_file, args.pd_file
    base = args.input_root / mode
    if mode == "stages":
        return base / "earlyMSA.npz", base / "earlyPD.npz"
    return base / "earlyMSA_wostages.npz", base / "earlyPD_wostages.npz"


def _check_two_representations(stage_info, night_info):
    """Verify both SOMNIS analyses use the exact same fixed study split."""
    for group in ("MSA", "PD"):
        left = set(zip(stage_info[group]["ids"], stage_info[group]["partition"]))
        right = set(zip(night_info[group]["ids"], night_info[group]["partition"]))
        if left != right:
            raise ValueError(f"{group}: Stage-resolved and Whole-night use different subjects/splits")


def cli(argv=None):
    parser = argparse.ArgumentParser(
        description="SOMNIS: reproducible nested-CV MSA vs PD model (concatenated_xgboost.py)."
    )
    parser.add_argument("--input-root", type=Path, default=Path("data/datasets/early"),
                        help="Root containing stages/ and wholenight/ from features/prepare_dataset.py")
    parser.add_argument("--msa-file", type=Path, default=None,
                        help="Optional direct MSA NPZ path (must also set --pd-file)")
    parser.add_argument("--pd-file", type=Path, default=None,
                        help="Optional direct PD NPZ path (must also set --msa-file)")
    parser.add_argument("--output-root", type=Path, default=Path("results/models"),
                        help="Output root (patient-level CSVs are confidential and must not be committed)")
    parser.add_argument("--mode", choices=["stages"], default="stages",
                        help="Fixed to stage-resolved PSG features (1390 columns)")
    parser.add_argument("--bootstrap", type=int, default=2000,
                        help="Number of subject-level bootstrap resamples (paper: 2000)")
    parser.add_argument("--seed", type=int, default=42, help="Original CV seed: 42")
    parser.add_argument("--expect-original-cohort", action="store_true",
                        help="Assert MSA 77/33 and PD 64/28 train/test subjects")
    parser.add_argument("--check-only", action="store_true",
                        help="Validate input metadata and partitions without training")
    args = parser.parse_args(argv)
    if (args.msa_file is None) != (args.pd_file is None):
        parser.error("--msa-file and --pd-file must be provided together")
    if args.bootstrap < 1:
        parser.error("--bootstrap must be >= 1")
    modes = ("stages", "wholenight") if args.mode == "both" else (args.mode,)
    global EXPECT_ORIGINAL_COHORT
    EXPECT_ORIGINAL_COHORT = args.expect_original_cohort
    prepared = {}
    for mode in modes:
        msa_file, pd_file = _resolve_paths(args, mode)
        info = validate_input_feature_schema(msa_file, pd_file, mode)
        prepared[mode] = (msa_file, pd_file, info)
        print(f"Input validated: {mode} | MSA={msa_file} | PD={pd_file}", flush=True)
    if len(modes) == 2:
        _check_two_representations(prepared["stages"][2], prepared["wholenight"][2])
    if args.check_only:
        print("All input schema and split checks passed. No model was trained.")
        return 0
    global MSA_FILE_PATH, PD_FILE_PATH, OUTPUT_DIR, RANDOM_SEED, BOOTSTRAP_SEED
    global N_BOOTSTRAP
    RANDOM_SEED = args.seed
    BOOTSTRAP_SEED = args.seed
    N_BOOTSTRAP = args.bootstrap
    EXPECT_ORIGINAL_COHORT = args.expect_original_cohort
    for mode in modes:
        MSA_FILE_PATH, PD_FILE_PATH, _ = prepared[mode]
        OUTPUT_DIR = args.output_root / "concatenated" / mode
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        print(f"\n=== concatenated | {mode} ===", flush=True)
        main()
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
