# -*- coding: utf-8 -*-
"""SOMNIS repository edition.

Command-line interface and input validation have been added around the
original model development, nested cross-fitting, bootstrap CI, DCA,
and visualization algorithms. Original numerical methods are unchanged.

For published reproducibility use --bootstrap 2000 --seed 42 and the
exact original locked discovery/independent cohorts. All model outputs
containing subject identifiers must remain private.
"""

# Strict nested cross-fitting stacking for early MSA vs PD
# Sens>=0.75 / Spec>=0.60 constrained max-BAC version
# Illustrator-safe flat confusion matrices with high-contrast native colormap fills

# Input
# -----

# Output
# ------
#     - clinical_performance_summary.png
#     - clinical_performance_summary.pdf
#     - model_performance_with_bootstrap_CI.csv
#       (all AUC, ACC, BAC, Sens, Spec, PPV, and NPV values include 95% CIs)
#     - predicted_probabilities_for_dca.csv

# Validation design
# -----------------
# 1. Discovery set:
#    - 5-fold outer stratified CV for internal validation.
#    - 5-fold strict meta-CV inside each outer-training partition.
#    - 5-fold XGBoost tuning and strict Level-0 OOF cross-fitting.
# 2. Independent validation set:
#    - locked and evaluated once only after final model development.
# 3. Hyperparameter selection:
#    - XGBoost and SVM candidates are selected by mean CV AUC,
#      not BAC at a fixed threshold.
# 4. Threshold selection:
#    - primary: among thresholds with Sens >= 0.75 and Spec >= 0.60,
#      select the threshold with maximum BAC;
#    - if no threshold satisfies both constraints, select the global maximum BAC;
#    - selected only from training-derived strict meta-CV OOF probabilities.
#    - the final Discovery strict-OOF ROC operating points are saved for audit.
# 5. Bootstrap:
#    - stratified bootstrap 95% CIs for Discovery outer-OOF and independent
#      validation performance.


import os
import argparse
import warnings
from pathlib import Path

# Avoid excessive nested multi-threading on Windows / BLAS.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
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
from sklearn.svm import SVC

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

# Nested-CV structure
N_OUTER_FOLDS = 5
N_META_TUNE_FOLDS = 5
N_BASE_TUNE_FOLDS = 5
N_L0_OOF_FOLDS = 5

XGB_N_JOBS = 1

# Threshold policy: single constrained-BAC rule with global-BAC fallback.
#
# Goal:
#   use one transparent, prespecified operating-point constraint.
# All thresholds are selected ONLY from training-derived strict OOF probabilities.
#
# Rule:
# 1. Among thresholds with Sens >= 0.75 AND Spec >= 0.60, maximize BAC.
# 2. If no threshold satisfies both constraints, use the global maximum BAC.
#
# No additional sensitivity/specificity fallback levels and no weighted-Youden
# fallback are used.
MIN_SENSITIVITY = 0.75
MIN_SPECIFICITY = 0.60

# Bootstrap configuration
N_BOOTSTRAP = 2000
BOOTSTRAP_SEED = 42

# DCA range
DCA_MIN_THRESHOLD = 0.01
DCA_MAX_THRESHOLD = 0.70
DCA_STEP = 0.01

# Compact, prespecified candidate grids.
XGB_PARAM_GRID = [
    {"n_estimators": 50, "max_depth": 3, "learning_rate": 0.05},
    {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.10},
    {"n_estimators": 150, "max_depth": 5, "learning_rate": 0.05},
]

SVM_PARAM_GRID = [
    {"C": 0.1, "kernel": "linear", "class_weight": None},
    {"C": 1.0, "kernel": "linear", "class_weight": None},
    {"C": 10.0, "kernel": "linear", "class_weight": None},
    {"C": 0.1, "kernel": "rbf", "gamma": "scale", "class_weight": "balanced"},
    {"C": 1.0, "kernel": "rbf", "gamma": "scale", "class_weight": "balanced"},
    {"C": 10.0, "kernel": "rbf", "gamma": "scale", "class_weight": "balanced"},
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
# 1. Data, splitting, and utility helpers
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
    """Conservative normalization for duplicate/overlap checks."""
    return np.asarray(
        [
            item.upper().replace(" ", "").replace("-", "_")
            for item in as_clean_strings(ids)
        ],
        dtype=str,
    )


def safe_auc(y_true, probabilities):
    """Return NaN instead of failing if a class is absent."""
    y_true = np.asarray(y_true, dtype=int)
    if len(np.unique(y_true)) < 2:
        return np.nan
    return float(roc_auc_score(y_true, probabilities))


def assert_cv_feasible(y, n_splits, context):
    """Check that requested stratified splitting is valid."""
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
    assert_cv_feasible(y, n_splits, context)
    splitter = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed,
    )
    return list(splitter.split(np.zeros(len(y)), np.asarray(y, dtype=int)))


def check_split_integrity(discovery_ids, independent_ids, y_discovery, y_independent):
    """Detect duplicate IDs and direct overlap between locked data partitions."""
    discovery_ids = normalize_subject_ids(discovery_ids)
    independent_ids = normalize_subject_ids(independent_ids)

    duplicated_discovery = (
        pd.Series(discovery_ids)[pd.Series(discovery_ids).duplicated()].unique()
    )
    duplicated_independent = (
        pd.Series(independent_ids)[pd.Series(independent_ids).duplicated()].unique()
    )
    overlap = np.intersect1d(discovery_ids, independent_ids)

    if len(duplicated_discovery):
        raise ValueError(
            f"Duplicated IDs within Discovery set: "
            f"{duplicated_discovery[:10].tolist()}"
        )
    if len(duplicated_independent):
        raise ValueError(
            f"Duplicated IDs within Independent validation set: "
            f"{duplicated_independent[:10].tolist()}"
        )
    if len(overlap):
        raise ValueError(
            "DATA LEAKAGE DETECTED: IDs occur in both Discovery and "
            f"Independent validation sets: {overlap[:10].tolist()}"
        )

    assert_cv_feasible(y_discovery, N_OUTER_FOLDS, "Discovery set / outer CV")
    assert_cv_feasible(
        y_independent,
        2,
        "Independent validation set",
    )


# =============================================================================
# 2. Model fitting and performance helpers
# =============================================================================
def fit_xgb_and_predict(X_fit, y_fit, X_predict, params):
    """
    Fit median imputer + z-score scaler + XGBoost on X_fit only.
    The current fitting subset determines imputation, scaling, and class weight.
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


def fit_svm_and_predict(L1_fit, y_fit, L1_predict, params):
    """Fit Level-1 scaling and SVM on L1_fit only, then predict L1_predict."""
    L1_fit = np.asarray(L1_fit, dtype=float)
    L1_predict = np.asarray(L1_predict, dtype=float)
    y_fit = np.asarray(y_fit, dtype=int)

    scaler = StandardScaler()
    L1_fit_scaled = scaler.fit_transform(L1_fit)
    L1_predict_scaled = scaler.transform(L1_predict)

    model = SVC(
        **params,
        probability=True,
        random_state=RANDOM_SEED,
    )
    model.fit(L1_fit_scaled, y_fit)

    return model.predict_proba(L1_predict_scaled)[:, 1]


def calculate_metrics(y_true, probabilities, threshold=None, predicted_labels=None):
    """
    Calculate probability-based AUC and threshold-based classification metrics.

    For Discovery outer-OOF performance, provide predicted_labels because every
    outer fold applies its own threshold selected from its own outer-training set.
    """
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)

    if predicted_labels is None:
        if threshold is None:
            raise ValueError("threshold is required unless predicted_labels are given.")
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
    """Return finite ROC operating points derived from the supplied labels/probabilities."""
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
# 3. AUC-driven XGBoost tuning and strict Level-0 cross-fitting
# =============================================================================
def tune_one_base_model(X, y, seed, stage_label, modality_name):
    """
    Tune one modality-specific XGBoost model using mean 5-fold CV AUC.

    AUC is threshold-independent, so it matches the separation of:
      (a) model/hyperparameter selection and
      (b) subsequent high-sensitivity threshold selection.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)

    splits = make_stratified_splits(
        y=y,
        n_splits=N_BASE_TUNE_FOLDS,
        seed=seed,
        context=f"{stage_label} / {modality_name} / base XGB tuning",
    )

    best_mean_auc = -np.inf
    best_params = None

    for params in XGB_PARAM_GRID:
        fold_auc = []

        for fit_idx, validation_idx in splits:
            probability = fit_xgb_and_predict(
                X_fit=X[fit_idx],
                y_fit=y[fit_idx],
                X_predict=X[validation_idx],
                params=params,
            )
            fold_auc.append(safe_auc(y[validation_idx], probability))

        mean_auc = float(np.nanmean(fold_auc))

        # First candidate wins exact ties for reproducibility.
        if mean_auc > best_mean_auc + 1e-12:
            best_mean_auc = mean_auc
            best_params = params.copy()

    return best_params, float(best_mean_auc)


def strict_oof_probabilities_for_one_modality(
    X_group,
    y,
    seed,
    stage_label,
    modality_name,
):
    """
    Build strict Level-0 OOF probabilities for a single modality.

    Every held-out Level-0 subject is predicted by a base model whose:
      - XGB hyperparameters were tuned using only the corresponding fit subset;
      - imputer/scaler/model were fitted only in that same fit subset.
    """
    X_group = np.asarray(X_group, dtype=float)
    y = np.asarray(y, dtype=int)

    l0_splits = make_stratified_splits(
        y=y,
        n_splits=N_L0_OOF_FOLDS,
        seed=seed,
        context=f"{stage_label} / {modality_name} / strict L0 OOF",
    )

    oof_probability = np.full(len(y), np.nan, dtype=float)

    for fold_number, (fit_idx, validation_idx) in enumerate(l0_splits, start=1):
        params, _ = tune_one_base_model(
            X=X_group[fit_idx],
            y=y[fit_idx],
            seed=seed + 10_000 * fold_number,
            stage_label=stage_label,
            modality_name=modality_name,
        )

        oof_probability[validation_idx] = fit_xgb_and_predict(
            X_fit=X_group[fit_idx],
            y_fit=y[fit_idx],
            X_predict=X_group[validation_idx],
            params=params,
        )

    if np.any(~np.isfinite(oof_probability)):
        raise RuntimeError(
            f"{stage_label} / {modality_name}: incomplete Level-0 OOF coverage."
        )

    return oof_probability


def build_strict_l1_features_and_holdout(
    X_fit,
    y_fit,
    X_holdout,
    feature_groups,
    group_names,
    seed,
    stage_label,
):
    """
    Create Level-1 features.

    L1_fit:
      strict L0 OOF probabilities for all current fitting subjects.

    L1_holdout:
      tune each XGB on all current fit subjects only, refit on all current fit
      subjects, then predict X_holdout.
    """
    X_fit = np.asarray(X_fit, dtype=float)
    y_fit = np.asarray(y_fit, dtype=int)
    X_holdout = np.asarray(X_holdout, dtype=float)

    L1_fit = np.zeros((len(y_fit), len(feature_groups)), dtype=float)
    L1_holdout = np.zeros((len(X_holdout), len(feature_groups)), dtype=float)

    selected_base_params = {}

    for group_index, (feature_mask, group_name) in enumerate(
        zip(feature_groups, group_names)
    ):
        X_fit_modality = X_fit[:, feature_mask]
        X_holdout_modality = X_holdout[:, feature_mask]

        L1_fit[:, group_index] = strict_oof_probabilities_for_one_modality(
            X_group=X_fit_modality,
            y=y_fit,
            seed=seed + 100_000 * (group_index + 1),
            stage_label=stage_label,
            modality_name=group_name,
        )

        params, _ = tune_one_base_model(
            X=X_fit_modality,
            y=y_fit,
            seed=seed + 200_000 * (group_index + 1),
            stage_label=stage_label,
            modality_name=group_name,
        )
        selected_base_params[group_name] = params

        L1_holdout[:, group_index] = fit_xgb_and_predict(
            X_fit=X_fit_modality,
            y_fit=y_fit,
            X_predict=X_holdout_modality,
            params=params,
        )

    return L1_fit, L1_holdout, selected_base_params


# =============================================================================
# 4. AUC-driven strict 5-fold meta-SVM tuning
# =============================================================================
def tune_meta_svm_strict(X, y, feature_groups, group_names, seed, stage_label):
    """
    Select the SVM configuration by mean strict 5-fold meta-CV AUC.

    For each meta-validation fold, all base-model tuning, strict L0 feature
    construction, and SVM fitting are restricted to its meta-training subset.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)

    meta_splits = make_stratified_splits(
        y=y,
        n_splits=N_META_TUNE_FOLDS,
        seed=seed,
        context=f"{stage_label} / strict meta-SVM tuning",
    )

    candidate_probability = np.full(
        (len(y), len(SVM_PARAM_GRID)),
        np.nan,
        dtype=float,
    )
    candidate_fold_auc = [[] for _ in SVM_PARAM_GRID]

    for meta_fold, (meta_train_idx, meta_validation_idx) in enumerate(
        meta_splits,
        start=1,
    ):
        X_meta_train = X[meta_train_idx]
        y_meta_train = y[meta_train_idx]
        X_meta_validation = X[meta_validation_idx]
        y_meta_validation = y[meta_validation_idx]

        L1_meta_train, L1_meta_validation, _ = (
            build_strict_l1_features_and_holdout(
                X_fit=X_meta_train,
                y_fit=y_meta_train,
                X_holdout=X_meta_validation,
                feature_groups=feature_groups,
                group_names=group_names,
                seed=seed + 1_000_000 * meta_fold,
                stage_label=f"{stage_label} | meta_fold_{meta_fold}",
            )
        )

        for candidate_index, svm_params in enumerate(SVM_PARAM_GRID):
            probability = fit_svm_and_predict(
                L1_fit=L1_meta_train,
                y_fit=y_meta_train,
                L1_predict=L1_meta_validation,
                params=svm_params,
            )

            candidate_probability[
                meta_validation_idx,
                candidate_index,
            ] = probability
            candidate_fold_auc[candidate_index].append(
                safe_auc(y_meta_validation, probability)
            )

    if np.any(~np.isfinite(candidate_probability)):
        raise RuntimeError(
            f"{stage_label}: incomplete strict meta-CV OOF probability coverage."
        )

    best_index = None
    best_mean_auc = -np.inf

    for candidate_index, fold_auc in enumerate(candidate_fold_auc):
        mean_auc = float(np.nanmean(fold_auc))
        if mean_auc > best_mean_auc + 1e-12:
            best_mean_auc = mean_auc
            best_index = candidate_index

    return (
        SVM_PARAM_GRID[best_index].copy(),
        float(best_mean_auc),
        candidate_probability[:, best_index].copy(),
    )


# =============================================================================
# 5. Stratified bootstrap confidence intervals
# =============================================================================
def bootstrap_confidence_intervals(
    y_true,
    probabilities,
    predicted_labels,
    n_bootstrap=N_BOOTSTRAP,
    random_seed=BOOTSTRAP_SEED,
):
    """
    Stratified subject-level bootstrap 95% CIs.

    Resampling is performed with replacement within PD and MSA separately.
    This preserves class counts and ensures every bootstrap replicate contains
    both classes. Prediction probabilities and fixed predicted labels are
    resampled together.

    For Discovery set, labels are the original outer-fold predictions, each
    generated with its own outer-training-derived threshold.
    For Independent validation set, labels use the fixed final threshold.
    """
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    predicted_labels = np.asarray(predicted_labels, dtype=int)

    rng = np.random.default_rng(random_seed)
    class0_idx = np.where(y_true == 0)[0]
    class1_idx = np.where(y_true == 1)[0]

    if len(class0_idx) == 0 or len(class1_idx) == 0:
        raise ValueError("Bootstrap requires both classes.")

    metric_names = ["AUC", "ACC", "BAC", "Sens", "Spec", "PPV", "NPV"]
    bootstrap_values = {metric: [] for metric in metric_names}

    for _ in range(n_bootstrap):
        sampled_idx = np.concatenate(
            [
                rng.choice(class0_idx, size=len(class0_idx), replace=True),
                rng.choice(class1_idx, size=len(class1_idx), replace=True),
            ]
        )
        rng.shuffle(sampled_idx)

        metrics = calculate_metrics(
            y_true=y_true[sampled_idx],
            probabilities=probabilities[sampled_idx],
            predicted_labels=predicted_labels[sampled_idx],
        )

        for metric in metric_names:
            bootstrap_values[metric].append(metrics[metric])

    ci = {}
    for metric in metric_names:
        values = np.asarray(bootstrap_values[metric], dtype=float)
        ci[metric] = (
            float(np.nanpercentile(values, 2.5)),
            float(np.nanpercentile(values, 97.5)),
        )

    return ci


# =============================================================================
# 6. Decision curve analysis and Illustrator-safe publication-style figure
# =============================================================================
def decision_curve_net_benefit(y_true, probabilities, thresholds):
    """
    Calculate model, treat-all, and treat-none net benefit curves.

    Positive class = MSA (label 1).
    """
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)

    n = len(y_true)
    prevalence = float(np.mean(y_true == 1))

    model_net_benefit = []
    treat_all_net_benefit = []
    treat_none_net_benefit = np.zeros_like(thresholds, dtype=float)

    for threshold in thresholds:
        predicted = probabilities >= threshold
        tp = np.sum((predicted == 1) & (y_true == 1))
        fp = np.sum((predicted == 1) & (y_true == 0))

        model_nb = (tp / n) - (fp / n) * (threshold / (1.0 - threshold))
        treat_all_nb = prevalence - (1.0 - prevalence) * (
            threshold / (1.0 - threshold)
        )

        model_net_benefit.append(model_nb)
        treat_all_net_benefit.append(treat_all_nb)

    return (
        np.asarray(model_net_benefit, dtype=float),
        np.asarray(treat_all_net_benefit, dtype=float),
        treat_none_net_benefit,
    )


def _hex_to_rgb(hex_color):
    """Convert a #RRGGBB colour to a 0-1 RGB tuple."""
    hex_color = hex_color.lstrip("#")
    if len(hex_color) != 6:
        raise ValueError(f"Expected #RRGGBB colour, got: {hex_color!r}")
    return tuple(int(hex_color[i:i + 2], 16) / 255.0 for i in (0, 2, 4))


def _blend_with_white(base_color, strength):
    """Return an opaque RGB colour blended from white to base_color.

    This deliberately returns an explicit RGB tuple rather than a Matplotlib
    colormap output. It prevents the PDF backend from embedding the confusion
    matrix as an indexed/raster image, which can lose colour information when
    the PDF is opened in Adobe Illustrator.
    """
    base_rgb = _hex_to_rgb(base_color)
    strength = float(np.clip(strength, 0.0, 1.0))
    return tuple(1.0 - (1.0 - channel) * strength for channel in base_rgb)


def draw_confusion_matrix(
    ax,
    matrix,
    title,
    cmap_name,
    panel_label,
    subtitle=None,
    show_y_label=True,
):
    """Render a high-contrast, Illustrator-safe 2 x 2 confusion matrix.

    Each cell is a separate opaque vector Rectangle, so the exported PDF remains
    editable in Adobe Illustrator. The fill colour follows Matplotlib's native
    ``Greens`` / ``Blues`` gradients after within-matrix min--max normalization.
    This reproduces the strong dark-to-light contrast of the reference figure
    while avoiding ``imshow`` / embedded indexed-image colour issues in PDF.
    """
    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape != (2, 2):
        raise ValueError("Confusion matrix must have shape (2, 2).")

    data_min = float(np.min(matrix))
    data_max = float(np.max(matrix))
    data_range = data_max - data_min
    cmap = plt.get_cmap(cmap_name)
    ax.set_facecolor("white")

    # The native colormap range deliberately spans from a near-white minimum
    # to a dark saturated maximum.  This is substantially stronger than the
    # previous manual white blending, especially for low-to-mid cell counts.
    for row in range(2):
        for col in range(2):
            value = float(matrix[row, col])
            if data_range <= 1e-12:
                normalized = 1.0
            else:
                normalized = (value - data_min) / data_range

            # Explicit RGB tuple keeps every cell an opaque vector object.
            fill_rgb = tuple(cmap(float(np.clip(normalized, 0.0, 1.0)))[:3])
            cell = Rectangle(
                (col, 1 - row),
                width=1.0,
                height=1.0,
                facecolor=fill_rgb,
                edgecolor="white",
                linewidth=1.5,
                antialiased=False,
                rasterized=False,
                clip_on=True,
                zorder=1,
            )
            ax.add_patch(cell)

            # Match the reference style: dark cells receive white text while
            # light/mid cells retain black text for contrast.
            text_color = "white" if normalized >= 0.50 else "black"
            ax.text(
                col + 0.5,
                1.5 - row,
                f"{int(round(value)):d}",
                ha="center",
                va="center",
                fontsize=15,
                fontweight="normal",
                color=text_color,
                zorder=2,
            )

    ax.set_xlim(0.0, 2.0)
    ax.set_ylim(0.0, 2.0)
    ax.set_aspect(0.62, adjustable="box")
    ax.set_anchor("C")

    ax.set_xticks([0.5, 1.5], ["PD", "MSA"])
    ax.set_yticks([1.5, 0.5], ["PD", "MSA"])
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label" if show_y_label else "")
    ax.set_title(title if subtitle is None else f"{title}\n{subtitle}", pad=10)
    ax.tick_params(axis="both", which="major", length=0)

    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.text(
        -0.18,
        1.16,
        panel_label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=16,
        fontweight="bold",
        color="black",
        clip_on=False,
    )


def _style_line_axis(ax):
    """Apply the restrained journal-style axis formatting used by ROC/DCA."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)
    ax.tick_params(axis="both", which="major", length=3.5, width=1.0)
    ax.grid(False)


def _add_panel_label(ax, panel_label):
    """Place a bold panel label just outside the upper-left plot corner."""
    ax.text(
        -0.13,
        1.10,
        panel_label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=16,
        fontweight="bold",
        color="black",
        clip_on=False,
    )


def format_ci(value, ci_pair):
    """Format a point estimate and bootstrap CI for concise plot labels."""
    return f"{value:.3f} ({ci_pair[0]:.3f}-{ci_pair[1]:.3f})"


def save_clinical_performance_figure(
    y_discovery,
    discovery_oof_probability,
    discovery_oof_label,
    y_independent,
    independent_probability,
    independent_label,
    discovery_metrics,
    independent_metrics,
    discovery_ci,
    independent_ci,
    independent_threshold,
):
    """
    Save a 2 x 2 publication-style clinical performance figure.

    Panels follow the visual language of reference panels b-e:
      b. DCA: Discovery outer-OOF and independent-validation curves together.
      c. Discovery outer-OOF confusion matrix.
      d. Independent-validation confusion matrix.
      e. ROC: Discovery outer-OOF and independent-validation curves together.

    No modelling, validation, threshold, or output-data logic is changed here.
    """
    thresholds = np.arange(
        DCA_MIN_THRESHOLD,
        DCA_MAX_THRESHOLD + DCA_STEP / 2,
        DCA_STEP,
    )

    fpr_discovery, tpr_discovery, _ = roc_curve(
        y_discovery,
        discovery_oof_probability,
    )
    fpr_independent, tpr_independent, _ = roc_curve(
        y_independent,
        independent_probability,
    )

    dca_discovery, treat_all_discovery, treat_none_discovery = (
        decision_curve_net_benefit(
            y_discovery,
            discovery_oof_probability,
            thresholds,
        )
    )
    dca_independent, _, _ = decision_curve_net_benefit(
        y_independent,
        independent_probability,
        thresholds,
    )

    # Consistent dataset identity across all four panels.
    discovery_green = "#16803C"
    independent_blue = "#3F83B9"
    diagonal_blue = "#8496D7"
    black = "#111111"
    neutral_gray = "#555555"

    with plt.rc_context(
        {
            "font.family": "Arial",
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.labelsize": 12,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 10.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            # Keep every confusion-matrix cell as an opaque vector patch.
            "pdf.compression": 6,
        }
    ):
        fig = plt.figure(figsize=(14.2, 10.2), facecolor="white")
        grid = fig.add_gridspec(2, 2, wspace=0.34, hspace=0.42)

        # ------------------------------------------------------------------
        # b. Decision curve analysis
        # ------------------------------------------------------------------
        ax = fig.add_subplot(grid[0, 0])
        ax.plot(
            thresholds,
            dca_discovery,
            color=discovery_green,
            lw=2.0,
            linestyle=(0, (7, 3.5)),
            label="Discovery outer-OOF",
        )
        ax.plot(
            thresholds,
            dca_independent,
            color=independent_blue,
            lw=2.1,
            label="Independent validation",
        )
        ax.plot(
            thresholds,
            treat_all_discovery,
            color=black,
            lw=1.55,
            linestyle=(0, (5, 2.5)),
            label="Treat all",
        )
        ax.plot(
            thresholds,
            treat_none_discovery,
            color=neutral_gray,
            lw=1.2,
            linestyle=(0, (8, 3.5)),
            label="Treat none",
        )

        # Keep the clinically useful net-benefit range prominent, as in the
        # reference DCA panel; the treat-all curve is allowed to exit below
        # -0.10 at high thresholds rather than dominating the panel.
        y_upper = max(0.60, float(np.nanmax([dca_discovery, dca_independent])) + 0.03)
        ax.set_xlim(0.0, DCA_MAX_THRESHOLD)
        ax.set_ylim(-0.10, y_upper)
        ax.set_xticks(np.arange(0.0, DCA_MAX_THRESHOLD + 0.001, 0.10))
        ax.set_yticks(np.arange(-0.10, y_upper + 0.001, 0.10))
        ax.set_xlabel("Threshold probability")
        ax.set_ylabel("Net benefit")
        ax.set_title("Decision Curve Analysis")
        _style_line_axis(ax)
        ax.legend(
            loc="upper right",
            frameon=False,
            handlelength=2.7,
            labelspacing=0.32,
            borderpad=0.15,
        )
        _add_panel_label(ax, "b.")

        # ------------------------------------------------------------------
        # c. Discovery confusion matrix
        # ------------------------------------------------------------------
        draw_confusion_matrix(
            ax=fig.add_subplot(grid[0, 1]),
            matrix=confusion_matrix(y_discovery, discovery_oof_label, labels=[0, 1]),
            title="Discovery Set Confusion Matrix",
            subtitle="Outer-OOF predictions; fold-specific thresholds",
            cmap_name="Greens",
            panel_label="c.",
            show_y_label=True,
        )

        # ------------------------------------------------------------------
        # d. Independent-validation confusion matrix
        # ------------------------------------------------------------------
        draw_confusion_matrix(
            ax=fig.add_subplot(grid[1, 0]),
            matrix=confusion_matrix(y_independent, independent_label, labels=[0, 1]),
            title="Independent Validation Set Confusion Matrix",
            subtitle=f"Fixed threshold = {independent_threshold:.3f}",
            cmap_name="Blues",
            panel_label="d.",
            show_y_label=True,
        )

        # ------------------------------------------------------------------
        # e. ROC curves
        # ------------------------------------------------------------------
        ax = fig.add_subplot(grid[1, 1])
        ax.plot(
            [0, 1],
            [0, 1],
            color=diagonal_blue,
            lw=1.2,
            linestyle=(0, (1.5, 2.5)),
            zorder=1,
            label="Reference",
        )
        ax.plot(
            fpr_discovery,
            tpr_discovery,
            color=discovery_green,
            lw=2.0,
            linestyle=(0, (7, 3.5)),
            zorder=3,
            label=(
                "Discovery outer-OOF "
                f"(AUC = {format_ci(discovery_metrics['AUC'], discovery_ci['AUC'])})"
            ),
        )
        ax.plot(
            fpr_independent,
            tpr_independent,
            color=independent_blue,
            lw=2.1,
            zorder=4,
            label=(
                "Independent validation "
                f"(AUC = {format_ci(independent_metrics['AUC'], independent_ci['AUC'])})"
            ),
        )
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.03)
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.set_title("Receiver Operating Characteristic Curve")
        _style_line_axis(ax)
        ax.legend(
            loc="lower right",
            frameon=False,
            handlelength=2.7,
            labelspacing=0.38,
            borderpad=0.15,
        )
        _add_panel_label(ax, "e.")

        fig.subplots_adjust(
            left=0.075,
            right=0.985,
            bottom=0.07,
            top=0.95,
            wspace=0.34,
            hspace=0.42,
        )

        png_path = OUTPUT_DIR / "clinical_performance_summary.png"
        pdf_path = OUTPUT_DIR / "clinical_performance_summary.pdf"
        fig.savefig(
            png_path,
            dpi=400,
            bbox_inches="tight",
            facecolor="white",
            transparent=False,
        )
        fig.savefig(
            pdf_path,
            dpi=400,
            bbox_inches="tight",
            facecolor="white",
            transparent=False,
        )
        plt.close(fig)


# =============================================================================
# 7. Data loading
# =============================================================================
def load_data():
    """Load the fixed MSA/PD NPZ files and create locked data partitions."""
    for path, label in [(MSA_FILE_PATH, "MSA"), (PD_FILE_PATH, "PD")]:
        if not path.exists():
            raise FileNotFoundError(f"{label} file not found: {path}")

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
    msa_subject_ids = np.asarray(msa_data["subject_ids"])
    pd_subject_ids = np.asarray(pd_data["subject_ids"])

    if msa_X.ndim != 2 or pd_X.ndim != 2:
        raise ValueError("Both X arrays must have shape: subjects × features.")
    if msa_X.shape[1] != pd_X.shape[1]:
        raise ValueError("MSA and PD feature dimensions do not match.")
    if not np.array_equal(msa_channels, pd_channels):
        raise ValueError("MSA and PD channel/modality labels do not match.")
    if len(msa_channels) != msa_X.shape[1]:
        raise ValueError("MSA channels length does not match MSA feature count.")
    if len(pd_channels) != pd_X.shape[1]:
        raise ValueError("PD channels length does not match PD feature count.")

    msa_train = msa_dataset == "train"
    msa_test = msa_dataset == "test"
    pd_train = pd_dataset == "train"
    pd_test = pd_dataset == "test"

    if not np.all(msa_train | msa_test):
        raise ValueError("MSA 'dataset' values must be only 'train' or 'test'.")
    if not np.all(pd_train | pd_test):
        raise ValueError("PD 'dataset' values must be only 'train' or 'test'.")

    X_discovery = np.vstack([msa_X[msa_train], pd_X[pd_train]])
    X_independent = np.vstack([msa_X[msa_test], pd_X[pd_test]])

    y_discovery = np.hstack(
        [
            np.ones(np.sum(msa_train), dtype=int),
            np.zeros(np.sum(pd_train), dtype=int),
        ]
    )
    y_independent = np.hstack(
        [
            np.ones(np.sum(msa_test), dtype=int),
            np.zeros(np.sum(pd_test), dtype=int),
        ]
    )

    discovery_ids = np.concatenate(
        [msa_subject_ids[msa_train], pd_subject_ids[pd_train]]
    )
    independent_ids = np.concatenate(
        [msa_subject_ids[msa_test], pd_subject_ids[pd_test]]
    )

    channels_upper = np.char.upper(msa_channels)
    feature_groups = [
        channels_upper == "EEG",
        channels_upper == "EOG",
        channels_upper == "EMG",
    ]
    group_names = ["EEG", "EOG", "EMG"]

    if not all(np.any(group) for group in feature_groups):
        raise ValueError("Expected EEG, EOG, and EMG labels in the channels array.")

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
        "feature_groups": feature_groups,
        "group_names": group_names,
    }


# =============================================================================
# 8. Main analysis
# =============================================================================
def main():
    print("=" * 78, flush=True)
    print("Strict nested cross-fitting stacking: Early MSA vs PD", flush=True)
    print("AUC-driven hyperparameter selection + Sens>=0.75 / Spec>=0.60 max-BAC threshold", flush=True)
    print("=" * 78, flush=True)

    data = load_data()

    X_discovery = data["X_discovery"]
    X_independent = data["X_independent"]
    y_discovery = data["y_discovery"]
    y_independent = data["y_independent"]
    discovery_ids = data["discovery_ids"]
    independent_ids = data["independent_ids"]
    feature_groups = data["feature_groups"]
    group_names = data["group_names"]

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
    print("Integrity check passed: no duplicate or overlapping normalized IDs.", flush=True)
    print(
        "Threshold policy: Sens>=0.75 & Spec>=0.60 -> max BAC; "
        "if infeasible -> global max BAC.",
        flush=True,
    )

    # -------------------------------------------------------------------------
    # A. Discovery-set outer 5-fold nested CV
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

        # 1. Select SVM by strict meta-CV AUC within outer training only.
        best_svm_params, _, meta_oof_probability = tune_meta_svm_strict(
            X=X_outer_train,
            y=y_outer_train,
            feature_groups=feature_groups,
            group_names=group_names,
            seed=RANDOM_SEED + 10_000_000 * outer_fold,
            stage_label=f"outer_fold_{outer_fold}",
        )

        # 2. Select Sens>=0.75 / Spec>=0.60 max-BAC threshold from outer-training OOF only.
        threshold, _ = select_threshold_from_training_oof(
            y_true=y_outer_train,
            probabilities=meta_oof_probability,
        )

        # 3. Refit all components in outer training and predict untouched fold.
        L1_outer_train, L1_outer_validation, _ = (
            build_strict_l1_features_and_holdout(
                X_fit=X_outer_train,
                y_fit=y_outer_train,
                X_holdout=X_outer_validation,
                feature_groups=feature_groups,
                group_names=group_names,
                seed=RANDOM_SEED + 20_000_000 * outer_fold,
                stage_label=f"outer_fold_{outer_fold}_final_fit",
            )
        )

        outer_validation_probability = fit_svm_and_predict(
            L1_fit=L1_outer_train,
            y_fit=y_outer_train,
            L1_predict=L1_outer_validation,
            params=best_svm_params,
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
        raise RuntimeError("Incomplete outer-fold OOF predictions.")

    discovery_metrics = calculate_metrics(
        y_true=y_discovery,
        probabilities=outer_oof_probability,
        predicted_labels=outer_oof_label,
    )

    # -------------------------------------------------------------------------
    # B. Final development using complete Discovery set, then one locked test
    # -------------------------------------------------------------------------
    print("\nFinal development on complete Discovery set", flush=True)

    final_svm_params, final_svm_auc, final_meta_oof_probability = (
        tune_meta_svm_strict(
            X=X_discovery,
            y=y_discovery,
            feature_groups=feature_groups,
            group_names=group_names,
            seed=RANDOM_SEED + 900_000_000,
            stage_label="final_discovery_training",
        )
    )

    final_threshold, final_threshold_rule = select_threshold_from_training_oof(
        y_true=y_discovery,
        probabilities=final_meta_oof_probability,
    )

    # Save every finite Discovery strict-OOF ROC operating point for transparent
    # threshold auditing. Independent-validation labels are NOT used here.
    final_threshold_audit = _roc_threshold_table(
        y_true=y_discovery,
        probabilities=final_meta_oof_probability,
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

    L1_discovery, L1_independent, _ = build_strict_l1_features_and_holdout(
        X_fit=X_discovery,
        y_fit=y_discovery,
        X_holdout=X_independent,
        feature_groups=feature_groups,
        group_names=group_names,
        seed=RANDOM_SEED + 950_000_000,
        stage_label="final_discovery_fit",
    )

    independent_probability = fit_svm_and_predict(
        L1_fit=L1_discovery,
        y_fit=y_discovery,
        L1_predict=L1_independent,
        params=final_svm_params,
    )
    independent_metrics = calculate_metrics(
        y_true=y_independent,
        probabilities=independent_probability,
        threshold=final_threshold,
    )

    # -------------------------------------------------------------------------
    # C. Stratified bootstrap 95% CIs
    # -------------------------------------------------------------------------
    print(f"\nComputing stratified bootstrap 95% CIs ({N_BOOTSTRAP} resamples)", flush=True)

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
    # D. Save required output files only
    # -------------------------------------------------------------------------
    save_clinical_performance_figure(
        y_discovery=y_discovery,
        discovery_oof_probability=outer_oof_probability,
        discovery_oof_label=outer_oof_label,
        y_independent=y_independent,
        independent_probability=independent_probability,
        independent_label=independent_metrics["predicted_labels"],
        discovery_metrics=discovery_metrics,
        independent_metrics=independent_metrics,
        discovery_ci=discovery_ci,
        independent_ci=independent_ci,
        independent_threshold=final_threshold,
    )

    metric_names = ["AUC", "ACC", "BAC", "Sens", "Spec", "PPV", "NPV"]
    performance_rows = []

    for dataset_name, metrics, ci, threshold, threshold_rule in [
        (
            "Discovery set (outer-fold OOF)",
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

    performance_df = pd.DataFrame(performance_rows)
    performance_df.to_csv(
        OUTPUT_DIR / "model_performance_with_bootstrap_CI.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # For DCA or future calibration plots: labels and probabilities only.
    prediction_df = pd.concat(
        [
            pd.DataFrame(
                {
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
    # E. Print final summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 78, flush=True)
    print("Results", flush=True)
    print("=" * 78, flush=True)
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
        f"Final selected threshold = {final_threshold:.4f} "
        f"({final_threshold_rule})",
        flush=True,
    )
    print(
        "Selected SVM internal tuning performance "
        f"(mean strict 5-fold meta-CV AUC) = {final_svm_auc:.4f}; "
        f"params = {final_svm_params}",
        flush=True,
    )
    print("\nSaved outputs:", flush=True)
    print(f"  {OUTPUT_DIR / 'clinical_performance_summary.png'}", flush=True)
    print(f"  {OUTPUT_DIR / 'clinical_performance_summary.pdf'}", flush=True)
    print(
        f"  {OUTPUT_DIR / 'model_performance_with_bootstrap_CI.csv'}",
        flush=True,
    )
    print(
        f"  {OUTPUT_DIR / 'predicted_probabilities_for_dca.csv'}",
        flush=True,
    )





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
        description="SOMNIS: reproducible nested-CV MSA vs PD model (somnis_stacking.py)."
    )
    parser.add_argument("--input-root", type=Path, default=Path("data/datasets/early"),
                        help="Root containing stages/ and wholenight/ from features/prepare_dataset.py")
    parser.add_argument("--msa-file", type=Path, default=None,
                        help="Optional direct MSA NPZ path (must also set --pd-file)")
    parser.add_argument("--pd-file", type=Path, default=None,
                        help="Optional direct PD NPZ path (must also set --msa-file)")
    parser.add_argument("--output-root", type=Path, default=Path("results/models"),
                        help="Output root (patient-level CSVs are confidential and must not be committed)")
    parser.add_argument("--mode", choices=["stages", "wholenight", "both"], default="both",
                        help="Feature representation; 'both' runs TWO separate full stacking experiments")
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
        OUTPUT_DIR = args.output_root / "somnis" / mode
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        print(f"\n=== somnis | {mode} ===", flush=True)
        main()
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
