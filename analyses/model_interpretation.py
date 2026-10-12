# -*- coding: utf-8 -*-
"""SOMNIS stage-resolved model interpretation.

Uses FINAL DISCOVERY-ONLY model development functions from models/somnis_stacking.py.
Does not change the nested-CV procedure, threshold selection rule, or model
hyperparameter grids, and does not re-evaluate model performance.

The fixed SOMNIS main script currently writes predictions but not fitted model
objects. Therefore the final Discovery-only development (SVM tuning plus Level-0
cross-fitting) is REPLAYED using exactly the final model's seeds. It does NOT
re-run the outer five-fold validation, and it does NOT select features using
Independent validation data. If available, original main-model validation
probabilities are checked against the replayed final model.

Run from repository root: python analyses/model_interpretation.py --help
Never commit patient-level inputs or prediction CSVs to a public repository.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from math import factorial

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
from scipy.special import gammaln
from scipy.stats import ttest_ind
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.impute import SimpleImputer
import xgboost as xgb

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from models import somnis_stacking as somnis

# The paper's original interpretability settings. Bootstrap can be modified
# for smoke tests via CLI, but use 3000 replicates for publication.
RANDOM_SEED = 42
OUTPUT_DIR = Path("results/interpretation/stages")
TOP_N_FEATURES = 10
SHAP_BACKGROUND_MAX = 100
INTERPRETABILITY_DPI = 350
N_EFFECT_BOOTSTRAP = 3000
EFFECT_BOOTSTRAP_SEED = 20260702
EFFECT_DPI = 400
EFFECT_FIGSIZE = (19.0, 11.0)
EFFECT_CI_LEVEL = 0.95
EFFECT_DISCOVERY_COLOR = "#8C8C8C"
EFFECT_MSA_HIGHER_COLOR = "#C43C39"
EFFECT_MSA_LOWER_COLOR = "#3973B8"
JOINT_FIGURE_DPI = 400
JOINT_FIGURE_FIGSIZE = (20.5, 18.8)
MODALITY_COLORS = {"EEG": "#984EA3", "EOG": "#FF7F00", "EMG": "#4DAF4A"}
GAIN_EDGE_COLOR = "#4D4D4D"

def extract_xgb_gain_importance(
    model,
    feature_names,
    global_feature_indices,
    top_n=TOP_N_FEATURES,
):
    """
    Extract top features using XGBoost's average split gain.

    The model has been fitted on scaled values, but feature order is unchanged,
    so each booster key f0, f1, ... maps directly to the supplied feature name.
    Global feature indices are retained so that the identical Discovery-selected
    features can be assessed in both datasets for effect-size replication.
    """
    feature_names = np.asarray(feature_names, dtype=str)
    global_feature_indices = np.asarray(global_feature_indices, dtype=int)

    if len(feature_names) != len(global_feature_indices):
        raise ValueError(
            "feature_names and global_feature_indices must have the same length."
        )

    gain_values = np.zeros(len(feature_names), dtype=float)

    raw_gain = model.get_booster().get_score(importance_type="gain")
    for booster_name, gain in raw_gain.items():
        if not booster_name.startswith("f"):
            continue
        try:
            feature_index = int(booster_name[1:])
        except ValueError:
            continue
        if 0 <= feature_index < len(gain_values):
            gain_values[feature_index] = float(gain)

    n_keep = min(int(top_n), len(feature_names))
    order = np.argsort(gain_values)[::-1][:n_keep]

    return pd.DataFrame(
        {
            "Feature": feature_names[order],
            "Feature_index_within_modality": order.astype(int),
            "Global_feature_index": global_feature_indices[order].astype(int),
            "Gain": gain_values[order],
            "Rank": np.arange(1, len(order) + 1),
        }
    )


def _svm_probability_on_original_l1_features(scaler, model, X_l1):
    """Predict MSA probability from unscaled Level-1 modality probabilities."""
    X_l1 = np.asarray(X_l1, dtype=float)
    return model.predict_proba(scaler.transform(X_l1))[:, 1]


def compute_exact_interventional_shap_values(
    scaler,
    model,
    X_explain,
    X_background,
):
    """
    Calculate exact interventional SHAP values for the 3-input SVM meta-model.

    For each participant and each subset of the three modality probabilities,
    missing modalities are replaced by rows from the empirical background set.
    All 2^3 coalitions are enumerated, yielding exact Shapley values for this
    background distribution without an external SHAP package.

    SHAP values are expressed on the SVM's MSA probability scale.
    """
    X_explain = np.asarray(X_explain, dtype=float)
    X_background = np.asarray(X_background, dtype=float)

    if X_explain.ndim != 2 or X_background.ndim != 2:
        raise ValueError("SHAP inputs must be two-dimensional arrays.")
    if X_explain.shape[1] != X_background.shape[1]:
        raise ValueError("SHAP evaluation and background feature counts differ.")

    n_features = X_explain.shape[1]
    if n_features != 3:
        raise ValueError(
            "This exact SHAP routine is designed for the 3 modality-level "
            "inputs of the stacking SVM."
        )

    n_background = len(X_background)
    if n_background == 0:
        raise ValueError("At least one background observation is required.")

    # Cap background size only for speed; fixed evenly spaced rows preserve
    # deterministic and broad Discovery-set coverage.
    if n_background > SHAP_BACKGROUND_MAX:
        background_idx = np.linspace(
            0,
            n_background - 1,
            SHAP_BACKGROUND_MAX,
            dtype=int,
        )
        X_background = X_background[background_idx]
        n_background = len(X_background)

    all_masks = list(range(1 << n_features))
    factorial_m = factorial(n_features)
    shap_values = np.zeros_like(X_explain, dtype=float)

    for sample_index, sample in enumerate(X_explain):
        coalition_value = {}

        for mask in all_masks:
            synthetic = X_background.copy()
            selected_features = [
                feature_index
                for feature_index in range(n_features)
                if mask & (1 << feature_index)
            ]
            if selected_features:
                synthetic[:, selected_features] = sample[selected_features]

            coalition_value[mask] = float(
                np.mean(
                    _svm_probability_on_original_l1_features(
                        scaler=scaler,
                        model=model,
                        X_l1=synthetic,
                    )
                )
            )

        for feature_index in range(n_features):
            shap_value = 0.0
            for mask in all_masks:
                if mask & (1 << feature_index):
                    continue

                subset_size = int(mask.bit_count())
                weight = (
                    factorial(subset_size)
                    * factorial(n_features - subset_size - 1)
                    / factorial_m
                )
                mask_with_feature = mask | (1 << feature_index)
                shap_value += weight * (
                    coalition_value[mask_with_feature] - coalition_value[mask]
                )

            shap_values[sample_index, feature_index] = shap_value

    return shap_values


def _style_interpretability_axis(ax):
    """Use a calm, journal-style axis treatment."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.9)
    ax.spines["bottom"].set_linewidth(0.9)
    ax.tick_params(axis="both", which="major", length=3.0, width=0.85)
    ax.grid(axis="x", color="#D9D9D9", linewidth=0.65, alpha=0.75)
    ax.set_axisbelow(True)


def draw_xgb_gain_panel(ax, importance_df, modality_name, color):
    """Draw a horizontal top-10 XGBoost gain panel for one modality."""
    plot_df = importance_df.sort_values("Gain", ascending=True).copy()
    gains = plot_df["Gain"].to_numpy(dtype=float)

    if np.allclose(gains, 0.0):
        ax.text(
            0.5,
            0.5,
            "No non-zero XGBoost gain was available.",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=11,
        )
        ax.set_axis_off()
        return

    bars = ax.barh(
        y=plot_df["Feature"],
        width=gains,
        color=color,
        edgecolor="none",
        height=0.62,
    )

    axis_max = max(float(np.max(gains)) * 1.18, 1e-6)
    ax.set_xlim(0.0, axis_max)
    ax.set_xlabel("XGBoost gain")
    ax.set_ylabel("Feature")
    ax.set_title(
        f"{modality_name}: Top {len(plot_df)} Features by XGBoost Gain",
        loc="left",
        pad=9,
        fontweight="bold",
    )
    ax.tick_params(axis="y", labelsize=8.8)

    # Concise numeric annotation makes small between-feature differences clear.
    for bar, gain in zip(bars, gains):
        ax.text(
            min(bar.get_width() + axis_max * 0.015, axis_max * 0.985),
            bar.get_y() + bar.get_height() / 2,
            f"{gain:.3f}",
            va="center",
            ha="left",
            fontsize=8.1,
            color="#333333",
        )

    _style_interpretability_axis(ax)


def draw_svm_shap_panel(ax, shap_importance_df):
    """Draw global mean absolute SHAP importances for the SVM meta-model."""
    modality_colors = MODALITY_COLORS

    plot_df = shap_importance_df.copy()
    bars = ax.bar(
        plot_df["Modality"],
        plot_df["Mean_abs_SHAP"],
        color=[modality_colors[name] for name in plot_df["Modality"]],
        edgecolor="none",
        width=0.62,
    )

    values = plot_df["Mean_abs_SHAP"].to_numpy(dtype=float)
    axis_max = max(float(np.max(values)) * 1.22, 1e-6)
    ax.set_ylim(0.0, axis_max)
    ax.set_xlabel("Level-0 modality probability")
    ax.set_ylabel("Mean |SHAP value|")
    ax.set_title(
        "SVM Meta-model: Modality Importance by SHAP",
        loc="left",
        pad=9,
        fontweight="bold",
    )

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            min(bar.get_height() + axis_max * 0.025, axis_max * 0.985),
            f"{value:.3f}",
            va="bottom",
            ha="center",
            fontsize=10,
            color="#333333",
        )

    _style_interpretability_axis(ax)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.65, alpha=0.75)
    ax.grid(axis="x", visible=False)


def save_interpretability_figure(
    eeg_gain_df,
    eog_gain_df,
    emg_gain_df,
    svm_shap_importance_df,
):
    """
    Save the requested 2 x 2 interpretability figure.

    Layout:
      top-left     EEG top-10 XGBoost gain features
      top-right    EOG top-10 XGBoost gain features
      bottom-left  EMG top-10 XGBoost gain features
      bottom-right SVM meta-model mean absolute SHAP modality importance
    """
    with plt.rc_context(
        {
            "font.family": ["Arial", "DejaVu Sans"],
            "font.size": 10.5,
            "axes.titlesize": 13,
            "axes.labelsize": 10.5,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    ):
        fig, axes = plt.subplots(
            2,
            2,
            figsize=(16.2, 12.0),
            facecolor="white",
            constrained_layout=False,
        )

        draw_xgb_gain_panel(
            axes[0, 0],
            eeg_gain_df,
            modality_name="EEG",
            color="#984EA3",
        )
        draw_xgb_gain_panel(
            axes[0, 1],
            eog_gain_df,
            modality_name="EOG",
            color="#FF7F00",
        )
        draw_xgb_gain_panel(
            axes[1, 0],
            emg_gain_df,
            modality_name="EMG",
            color="#4DAF4A",
        )
        draw_svm_shap_panel(
            axes[1, 1],
            svm_shap_importance_df,
        )

        fig.suptitle(
            "Interpretability of the Multimodal Stacking Model",
            fontsize=17,
            fontweight="bold",
            y=0.982,
        )
        fig.subplots_adjust(
            left=0.13,
            right=0.985,
            bottom=0.075,
            top=0.925,
            wspace=0.42,
            hspace=0.38,
        )

        png_path = OUTPUT_DIR / "stacking_model_interpretability.png"
        pdf_path = OUTPUT_DIR / "stacking_model_interpretability.pdf"
        fig.savefig(
            png_path,
            dpi=INTERPRETABILITY_DPI,
            bbox_inches="tight",
            facecolor="white",
        )
        fig.savefig(
            pdf_path,
            dpi=INTERPRETABILITY_DPI,
            bbox_inches="tight",
            facecolor="white",
        )
        plt.close(fig)


def _hedges_g_from_groups(msa_values, pd_values):
    """
    Calculate Hedges' g for MSA minus PD.

    A positive value denotes a higher mean feature value in MSA; a negative
    value denotes a lower mean feature value in MSA. The small-sample correction
    J is calculated using gammaln for numerical stability.
    """
    msa_values = np.asarray(msa_values, dtype=float)
    pd_values = np.asarray(pd_values, dtype=float)

    n_msa = len(msa_values)
    n_pd = len(pd_values)
    df = n_msa + n_pd - 2

    if n_msa < 2 or n_pd < 2 or df <= 0:
        return np.nan

    var_msa = np.var(msa_values, ddof=1)
    var_pd = np.var(pd_values, ddof=1)
    pooled_variance = (
        (n_msa - 1) * var_msa + (n_pd - 1) * var_pd
    ) / df

    if not np.isfinite(pooled_variance) or pooled_variance <= 0:
        return np.nan

    cohen_d = (np.mean(msa_values) - np.mean(pd_values)) / np.sqrt(
        pooled_variance
    )
    correction_j = np.exp(
        gammaln(df / 2.0)
        - 0.5 * np.log(df / 2.0)
        - gammaln((df - 1.0) / 2.0)
    )
    return float(correction_j * cohen_d)


def _bootstrap_hedges_g_ci(
    msa_values,
    pd_values,
    n_bootstrap=N_EFFECT_BOOTSTRAP,
    random_seed=EFFECT_BOOTSTRAP_SEED,
):
    """
    Stratified non-parametric bootstrap 95% CI for Hedges' g.

    MSA and PD participants are resampled independently with replacement so
    the observed group sizes are preserved in each bootstrap replicate.
    """
    msa_values = np.asarray(msa_values, dtype=float)
    pd_values = np.asarray(pd_values, dtype=float)

    if len(msa_values) < 2 or len(pd_values) < 2:
        return np.nan, np.nan

    rng = np.random.default_rng(int(random_seed))
    bootstrap_g = np.full(int(n_bootstrap), np.nan, dtype=float)

    for bootstrap_index in range(int(n_bootstrap)):
        sampled_msa = rng.choice(
            msa_values,
            size=len(msa_values),
            replace=True,
        )
        sampled_pd = rng.choice(
            pd_values,
            size=len(pd_values),
            replace=True,
        )
        bootstrap_g[bootstrap_index] = _hedges_g_from_groups(
            sampled_msa,
            sampled_pd,
        )

    bootstrap_g = bootstrap_g[np.isfinite(bootstrap_g)]
    if len(bootstrap_g) < max(100, int(0.80 * n_bootstrap)):
        return np.nan, np.nan

    alpha = 1.0 - float(EFFECT_CI_LEVEL)
    return (
        float(np.quantile(bootstrap_g, alpha / 2.0)),
        float(np.quantile(bootstrap_g, 1.0 - alpha / 2.0)),
    )


def _benjamini_hochberg(p_values):
    """Return BH-FDR-adjusted q values while preserving NaNs."""
    p_values = np.asarray(p_values, dtype=float)
    q_values = np.full(len(p_values), np.nan, dtype=float)

    valid = np.isfinite(p_values)
    if not np.any(valid):
        return q_values

    valid_p = p_values[valid]
    n_tests = len(valid_p)
    order = np.argsort(valid_p)
    ordered_p = valid_p[order]

    adjusted = ordered_p * n_tests / np.arange(1, n_tests + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)

    restored = np.empty(n_tests, dtype=float)
    restored[order] = adjusted
    q_values[valid] = restored
    return q_values


def _effect_direction(g_value):
    """Return a human-readable direction based on Hedges' g (MSA minus PD)."""
    if not np.isfinite(g_value):
        return "Not estimable"
    if g_value > 0:
        return "MSA > PD"
    if g_value < 0:
        return "MSA < PD"
    return "MSA = PD"


def _effect_statistics_for_feature(
    raw_feature_values,
    labels,
    dataset_name,
    random_seed,
):
    """
    Calculate unadjusted descriptive statistics, Hedges' g, bootstrap CI, and
    Welch's two-sided t-test from raw finite feature values only.

    No imputation, z-scoring, or model-fit transformation is applied here.
    """
    raw_feature_values = np.asarray(raw_feature_values, dtype=float)
    labels = np.asarray(labels, dtype=int)

    finite = np.isfinite(raw_feature_values)
    msa_values = raw_feature_values[finite & (labels == 1)]
    pd_values = raw_feature_values[finite & (labels == 0)]

    result = {
        "Dataset": dataset_name,
        "n_MSA": int(len(msa_values)),
        "n_PD": int(len(pd_values)),
        "MSA_mean_raw": float(np.mean(msa_values)) if len(msa_values) else np.nan,
        "PD_mean_raw": float(np.mean(pd_values)) if len(pd_values) else np.nan,
        "MSA_SD_raw": (
            float(np.std(msa_values, ddof=1)) if len(msa_values) >= 2 else np.nan
        ),
        "PD_SD_raw": (
            float(np.std(pd_values, ddof=1)) if len(pd_values) >= 2 else np.nan
        ),
        "Hedges_g_MSA_minus_PD": np.nan,
        "Hedges_g_CI_lower": np.nan,
        "Hedges_g_CI_upper": np.nan,
        "Welch_t": np.nan,
        "P_value_Welch": np.nan,
    }

    if len(msa_values) < 2 or len(pd_values) < 2:
        return result

    g_value = _hedges_g_from_groups(msa_values, pd_values)
    ci_lower, ci_upper = _bootstrap_hedges_g_ci(
        msa_values=msa_values,
        pd_values=pd_values,
        n_bootstrap=N_EFFECT_BOOTSTRAP,
        random_seed=random_seed,
    )
    welch_result = ttest_ind(
        msa_values,
        pd_values,
        equal_var=False,
        nan_policy="omit",
        alternative="two-sided",
    )

    result.update(
        {
            "Hedges_g_MSA_minus_PD": g_value,
            "Hedges_g_CI_lower": ci_lower,
            "Hedges_g_CI_upper": ci_upper,
            "Welch_t": float(welch_result.statistic),
            "P_value_Welch": float(welch_result.pvalue),
        }
    )
    return result


def compute_selected_feature_effect_sizes(
    xgb_gain_tables,
    X_discovery,
    y_discovery,
    X_independent,
    y_independent,
):
    """
    Replicate group-level effect sizes for final Discovery-selected Top-10
    XGBoost-gain features in EEG, EOG, and EMG.

    Feature selection is exclusively based on the final XGBoost models trained
    on the complete Discovery set. The same locked 30-feature list is then
    evaluated in both datasets. BH-FDR correction is performed separately
    across all 30 selected features within each dataset.
    """
    records = []
    analysis_seed = EFFECT_BOOTSTRAP_SEED

    for modality_index, modality_name in enumerate(["EEG", "EOG", "EMG"]):
        top_features = (
            xgb_gain_tables[modality_name]
            .sort_values("Rank")
            .reset_index(drop=True)
        )

        if len(top_features) != TOP_N_FEATURES:
            raise RuntimeError(
                f"{modality_name}: expected {TOP_N_FEATURES} selected features, "
                f"but found {len(top_features)}."
            )

        for row_index, feature_row in top_features.iterrows():
            global_feature_index = int(feature_row["Global_feature_index"])

            common_metadata = {
                "Modality": modality_name,
                "Rank_by_final_XGB_gain": int(feature_row["Rank"]),
                "Feature": str(feature_row["Feature"]),
                "Global_feature_index": global_feature_index,
                "Feature_index_within_modality": int(
                    feature_row["Feature_index_within_modality"]
                ),
                "Final_XGB_gain": float(feature_row["Gain"]),
                "Effect_size_definition": "Hedges_g (MSA minus PD)",
                "Feature_values": "raw_finite_values_no_imputation_no_scaling",
            }

            discovery_stats = _effect_statistics_for_feature(
                raw_feature_values=X_discovery[:, global_feature_index],
                labels=y_discovery,
                dataset_name="Discovery set",
                random_seed=analysis_seed
                + 100_000 * modality_index
                + 1_000 * row_index
                + 1,
            )
            independent_stats = _effect_statistics_for_feature(
                raw_feature_values=X_independent[:, global_feature_index],
                labels=y_independent,
                dataset_name="Independent validation set",
                random_seed=analysis_seed
                + 1_000_000
                + 100_000 * modality_index
                + 1_000 * row_index
                + 1,
            )

            records.append({**common_metadata, **discovery_stats})
            records.append({**common_metadata, **independent_stats})

    effect_df = pd.DataFrame(records)

    # FDR is controlled across the full 30-feature family separately in each
    # dataset; this is more conservative than modality-specific correction.
    effect_df["Q_value_BH_FDR_30_features"] = np.nan
    for dataset_name in ["Discovery set", "Independent validation set"]:
        dataset_mask = effect_df["Dataset"] == dataset_name
        effect_df.loc[
            dataset_mask,
            "Q_value_BH_FDR_30_features",
        ] = _benjamini_hochberg(
            effect_df.loc[dataset_mask, "P_value_Welch"].to_numpy(dtype=float)
        )

    effect_df["Direction"] = effect_df[
        "Hedges_g_MSA_minus_PD"
    ].map(_effect_direction)
    effect_df["CI_excludes_zero"] = (
        (effect_df["Hedges_g_CI_lower"] > 0)
        | (effect_df["Hedges_g_CI_upper"] < 0)
    )

    return effect_df


def make_effect_size_wide_table(effect_df):
    """Create one row per selected feature with both dataset estimates."""
    id_columns = [
        "Modality",
        "Rank_by_final_XGB_gain",
        "Feature",
        "Global_feature_index",
        "Feature_index_within_modality",
        "Final_XGB_gain",
        "Effect_size_definition",
        "Feature_values",
    ]
    value_columns = [
        "n_MSA",
        "n_PD",
        "MSA_mean_raw",
        "PD_mean_raw",
        "MSA_SD_raw",
        "PD_SD_raw",
        "Hedges_g_MSA_minus_PD",
        "Hedges_g_CI_lower",
        "Hedges_g_CI_upper",
        "Welch_t",
        "P_value_Welch",
        "Q_value_BH_FDR_30_features",
        "Direction",
        "CI_excludes_zero",
    ]

    wide = (
        effect_df
        .set_index(id_columns + ["Dataset"])[value_columns]
        .unstack("Dataset")
    )
    wide.columns = [
        f"{dataset.replace(' ', '_')}_{column}"
        for column, dataset in wide.columns
    ]
    wide = wide.reset_index()
    return wide


def _format_independent_q_value(q_value):
    """Format independent-validation q values for figure annotations."""
    if not np.isfinite(q_value):
        return "q=NA"
    if q_value < 0.001:
        return "q<0.001***"
    if q_value < 0.01:
        return f"q={q_value:.3f}**"
    if q_value < 0.05:
        return f"q={q_value:.3f}*"
    return f"q={q_value:.3f}"


def _effect_color_for_independent_point(g_value):
    """Color independent-validation effects by MSA-minus-PD direction."""
    if not np.isfinite(g_value):
        return EFFECT_DISCOVERY_COLOR
    if g_value >= 0:
        return EFFECT_MSA_HIGHER_COLOR
    return EFFECT_MSA_LOWER_COLOR


def save_effect_size_replication_figure(effect_df):
    """Save modality-coloured paired forest plots for the selected features."""
    modalities = ["EEG", "EOG", "EMG"]
    ci_values = effect_df[["Hedges_g_CI_lower", "Hedges_g_CI_upper"]].to_numpy(float)
    finite_ci = ci_values[np.isfinite(ci_values)]
    lim = max(np.max(np.abs(finite_ci)) if len(finite_ci) else 1.0, 0.30)
    lim = np.ceil((lim + 0.15) * 10) / 10
    annotation_limit = lim + 0.62

    with plt.rc_context({"font.family": ["Arial", "DejaVu Sans"], "font.size": 10,
                         "pdf.fonttype": 42, "ps.fonttype": 42}):
        fig, axes = plt.subplots(1, 3, figsize=EFFECT_FIGSIZE, sharex=True,
                                 facecolor="white", constrained_layout=False)
        for ax, modality in zip(axes, modalities):
            color = MODALITY_COLORS[modality]
            table = effect_df.loc[effect_df["Modality"] == modality]
            discovery = table.loc[table["Dataset"] == "Discovery set"].sort_values(
                "Rank_by_final_XGB_gain").reset_index(drop=True)
            independent = table.loc[table["Dataset"] == "Independent validation set"].sort_values(
                "Rank_by_final_XGB_gain").reset_index(drop=True)
            if len(discovery) != TOP_N_FEATURES or len(independent) != TOP_N_FEATURES:
                raise RuntimeError(f"{modality}: incomplete effect-size table.")

            y = np.arange(TOP_N_FEATURES)[::-1]
            ax.axvline(0, color="#5F5F5F", lw=1.0, ls="--", zorder=0)
            for yy, row in zip(y, discovery.to_dict("records")):
                g, lo, hi = (float(row[k]) for k in ["Hedges_g_MSA_minus_PD",
                                                      "Hedges_g_CI_lower", "Hedges_g_CI_upper"])
                if np.isfinite([g, lo, hi]).all():
                    ax.errorbar(g, yy + 0.15, xerr=[[max(g-lo, 0)], [max(hi-g, 0)]],
                                fmt="o", ms=6, mfc="white", mec=color, mew=1.3,
                                ecolor=color, elinewidth=1.1, capsize=2.3, zorder=3)
            for yy, row in zip(y, independent.to_dict("records")):
                g, lo, hi = (float(row[k]) for k in ["Hedges_g_MSA_minus_PD",
                                                      "Hedges_g_CI_lower", "Hedges_g_CI_upper"])
                if np.isfinite([g, lo, hi]).all():
                    ax.errorbar(g, yy - 0.15, xerr=[[max(g-lo, 0)], [max(hi-g, 0)]],
                                fmt="o", ms=6, mfc=color, mec=color, mew=0.8,
                                ecolor=color, elinewidth=1.2, capsize=2.3, zorder=4)
                ax.text(lim + 0.06, yy - 0.15,
                        _format_independent_q_value(row["Q_value_BH_FDR_30_features"]),
                        ha="left", va="center", fontsize=8.5, color="#333333", clip_on=False)

            ax.set(yticks=y, yticklabels=discovery["Feature"].tolist(), ylim=(-0.7, TOP_N_FEATURES-0.3),
                   xlim=(-lim, annotation_limit), xlabel="Hedges' g (MSA − PD)")
            ax.set_title(f"{modality}: Top {TOP_N_FEATURES} Final-model Gain Features",
                         loc="left", fontsize=11.3, fontweight="bold", pad=10)
            ax.text(lim + 0.06, TOP_N_FEATURES - 0.08, "Independent\nBH-FDR q",
                    ha="left", va="bottom", fontsize=8.2, fontweight="bold", clip_on=False)
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(axis="x", color="#D9D9D9", lw=0.65, ls=":")
            ax.set_axisbelow(True)
            ax.tick_params(axis="y", length=0, pad=3, labelsize=8.8)

        fig.legend(handles=[
            plt.Line2D([0], [0], marker="o", color="#555555", mfc="white", ms=7,
                       lw=1.1, label="Discovery set (open circles)"),
            plt.Line2D([0], [0], marker="o", color="#555555", mfc="#555555", ms=7,
                       lw=1.1, label="Independent validation (filled circles)"),
        ], loc="lower center", ncol=2, frameon=False, fontsize=10,
           bbox_to_anchor=(0.50, 0.015), columnspacing=2.2)
        fig.suptitle("Effect-size Consistency of Discovery-selected Model-important Features",
                     fontsize=15.5, fontweight="bold", y=0.985)
        fig.text(0.50, 0.052,
                 "Colours identify modality; position relative to 0 indicates MSA − PD direction. "
                 "Horizontal lines: bootstrap 95% CIs; q values: BH-FDR across 30 selected features.",
                 ha="center", va="center", fontsize=8.8, color="#333333")
        fig.subplots_adjust(left=0.20, right=0.965, bottom=0.115, top=0.91, wspace=0.90)
        fig.savefig(OUTPUT_DIR / "selected_feature_effect_size_replication.png", dpi=EFFECT_DPI,
                    bbox_inches="tight", facecolor="white")
        fig.savefig(OUTPUT_DIR / "selected_feature_effect_size_replication.pdf", dpi=EFFECT_DPI,
                    bbox_inches="tight", facecolor="white")
        plt.close(fig)


def _joint_modality_tables(xgb_gain_tables, effect_df, modality_name):
    """Return strictly aligned Top-10 gain and paired effect-size rows."""
    gain_df = (
        xgb_gain_tables[modality_name]
        .sort_values("Rank")
        .reset_index(drop=True)
        .copy()
    )
    discovery_df = (
        effect_df.loc[
            (effect_df["Modality"] == modality_name)
            & (effect_df["Dataset"] == "Discovery set")
        ]
        .sort_values("Rank_by_final_XGB_gain")
        .reset_index(drop=True)
        .copy()
    )
    independent_df = (
        effect_df.loc[
            (effect_df["Modality"] == modality_name)
            & (effect_df["Dataset"] == "Independent validation set")
        ]
        .sort_values("Rank_by_final_XGB_gain")
        .reset_index(drop=True)
        .copy()
    )

    if not (
        len(gain_df) == len(discovery_df) == len(independent_df) == TOP_N_FEATURES
    ):
        raise RuntimeError(
            f"{modality_name}: expected {TOP_N_FEATURES} aligned selected features "
            "for the joint gain/effect figure."
        )

    gain_features = gain_df["Feature"].astype(str).to_numpy()
    discovery_features = discovery_df["Feature"].astype(str).to_numpy()
    independent_features = independent_df["Feature"].astype(str).to_numpy()

    if not np.array_equal(gain_features, discovery_features):
        raise RuntimeError(
            f"{modality_name}: gain and Discovery effect feature order differ."
        )
    if not np.array_equal(gain_features, independent_features):
        raise RuntimeError(
            f"{modality_name}: gain and independent effect feature order differ."
        )

    return gain_df, discovery_df, independent_df


def _shared_effect_xlim(effect_df):
    """Return a symmetric x-axis limit plus annotation space for all effects."""
    ci_values = effect_df[
        ["Hedges_g_CI_lower", "Hedges_g_CI_upper"]
    ].to_numpy(dtype=float)
    finite_ci_values = ci_values[np.isfinite(ci_values)]

    if len(finite_ci_values):
        observed_min = float(np.min(finite_ci_values))
        observed_max = float(np.max(finite_ci_values))
    else:
        observed_min, observed_max = -1.0, 1.0

    data_limit = max(abs(observed_min), abs(observed_max), 0.30)
    data_limit = np.ceil((data_limit + 0.15) * 10.0) / 10.0
    annotation_limit = data_limit + 0.62
    return float(data_limit), float(annotation_limit)


def _style_joint_axis(ax, show_y_labels=False, y_on_right=False):
    """Minimal shared axis styling for the gain/effect rows."""
    ax.spines[["top", "left", "right"]].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.85)
    ax.grid(axis="x", color="#D9D9D9", linewidth=0.65, linestyle=":")
    ax.grid(axis="y", visible=False)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", labelsize=8.8, length=3.0, width=0.8)
    ax.tick_params(axis="y", length=0, labelsize=8.4)
    if y_on_right:
        ax.yaxis.tick_right()
        ax.tick_params(axis="y", labelright=True, labelleft=False, pad=9)
    elif not show_y_labels:
        ax.tick_params(axis="y", labelleft=False, labelright=False)


def _draw_joint_gain_panel(ax, gain_df, modality_name, y_positions):
    """Right-aligned gain bars extending left toward the feature labels."""
    gains = gain_df["Gain"].to_numpy(float)
    features = gain_df["Feature"].astype(str).tolist()
    color = MODALITY_COLORS[modality_name]
    if not np.isfinite(gains).all() or np.allclose(gains, 0):
        ax.set_axis_off()
        return

    x_max = max(float(np.max(gains)) * 1.18, 1e-6)
    bars = ax.barh(
        y_positions, gains, left=-gains, height=0.62,
        color=color, edgecolor="none", zorder=2,
    )
    ax.set(
        xlim=(-x_max, 0.0), yticks=y_positions, yticklabels=features,
        ylim=(-0.70, TOP_N_FEATURES - 0.30),
    )
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{abs(x):g}"))
    for bar, gain in zip(bars, gains):
        ax.text(
            -0.018 * x_max,
            bar.get_y() + bar.get_height() / 2,
            f"{gain:.3f}",
            ha="right", va="center", fontsize=7.7, color="white", zorder=3,
        )
    ax.text(
        -0.18, 0.50, modality_name, transform=ax.transAxes, rotation=90,
        ha="center", va="center", fontsize=12.5, fontweight="bold", color=color,
    )
    _style_joint_axis(ax, show_y_labels=True, y_on_right=True)


def _draw_joint_effect_panel(ax, discovery_df, independent_df, modality_name,
                             y_positions, x_data_limit, x_annotation_limit):
    """Right panel: paired effect sizes in the modality colour; x-position gives direction."""
    color = MODALITY_COLORS[modality_name]
    ax.axvline(0, color="#5F5F5F", lw=1.0, ls="--", zorder=0)
    for y, row in zip(y_positions, discovery_df.to_dict("records")):
        g, lo, hi = (float(row[k]) for k in ["Hedges_g_MSA_minus_PD",
                                              "Hedges_g_CI_lower", "Hedges_g_CI_upper"])
        if np.isfinite([g, lo, hi]).all():
            ax.errorbar(g, y + 0.15, xerr=[[max(g-lo, 0)], [max(hi-g, 0)]],
                        fmt="o", ms=5.8, mfc="white", mec=color, mew=1.25,
                        ecolor=color, elinewidth=1.10, capsize=2.3, zorder=3)
    for y, row in zip(y_positions, independent_df.to_dict("records")):
        g, lo, hi = (float(row[k]) for k in ["Hedges_g_MSA_minus_PD",
                                              "Hedges_g_CI_lower", "Hedges_g_CI_upper"])
        if np.isfinite([g, lo, hi]).all():
            ax.errorbar(g, y - 0.15, xerr=[[max(g-lo, 0)], [max(hi-g, 0)]],
                        fmt="o", ms=5.8, mfc=color, mec=color, mew=0.75,
                        ecolor=color, elinewidth=1.20, capsize=2.3, zorder=4)
        ax.text(x_data_limit + 0.06, y - 0.15,
                _format_independent_q_value(row["Q_value_BH_FDR_30_features"]),
                ha="left", va="center", fontsize=8.0, color="#333333", clip_on=False)
    ax.set(yticks=y_positions, ylim=(-0.70, TOP_N_FEATURES - 0.30),
           xlim=(-x_data_limit, x_annotation_limit))
    ax.text(x_data_limit + 0.06, TOP_N_FEATURES - 0.10, "Independent\nBH-FDR q",
            ha="left", va="bottom", fontsize=7.8, fontweight="bold", color="#333333", clip_on=False)
    _style_joint_axis(ax)


def _draw_joint_svm_shap_panel(ax, shap_importance_df):
    """Compact, centred SVM modality-importance panel."""
    order = ["EEG", "EOG", "EMG"]
    plot_df = shap_importance_df.copy()
    plot_df["Modality"] = pd.Categorical(plot_df["Modality"], order, ordered=True)
    plot_df = plot_df.sort_values("Modality")
    values = plot_df["Mean_abs_SHAP"].to_numpy(float)
    labels = plot_df["Modality"].astype(str).tolist()
    bars = ax.bar(
        labels, values, color=[MODALITY_COLORS[x] for x in labels],
        edgecolor="none", width=0.36, zorder=3,
    )
    y_max = max(float(np.nanmax(values)) * 1.30, 1e-6)
    ax.set(
        ylim=(0, y_max), ylabel="Mean |SHAP value|",
        xlabel="Level-0 modality probability",
        title="SVM Meta-model Modality Importance (Discovery set)",
    )
    ax.margins(x=0.35)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            min(value + 0.040 * y_max, 0.97 * y_max),
            f"{value:.3f}", ha="center", va="bottom", fontsize=9.4, color="#333333",
        )
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#D9D9D9", lw=0.65, ls=":")
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", labelsize=9.0, length=3.0, width=0.8)


def save_joint_gain_effect_figure(xgb_gain_tables, effect_df, svm_shap_importance_df):
    """Save the compact 3-row gain/effect figure with central feature labels."""
    modalities = ["EEG", "EOG", "EMG"]
    x_data_limit, x_annotation_limit = _shared_effect_xlim(effect_df)
    with plt.rc_context({"font.family": ["Arial", "DejaVu Sans"], "font.size": 10,
                         "pdf.fonttype": 42, "ps.fonttype": 42}):
        fig = plt.figure(figsize=JOINT_FIGURE_FIGSIZE, facecolor="white")
        grid = fig.add_gridspec(4, 2, height_ratios=[1, 1, 1, 0.58],
                                width_ratios=[0.90, 1.10], left=0.08, right=0.965,
                                bottom=0.105, top=0.910, wspace=0.66, hspace=0.36)
        y = np.arange(TOP_N_FEATURES)[::-1]

        for i, modality in enumerate(modalities):
            gain_df, discovery_df, independent_df = _joint_modality_tables(
                xgb_gain_tables, effect_df, modality)
            gain_ax = fig.add_subplot(grid[i, 0])
            effect_ax = fig.add_subplot(grid[i, 1])
            _draw_joint_gain_panel(gain_ax, gain_df, modality, y)
            _draw_joint_effect_panel(effect_ax, discovery_df, independent_df, modality, y,
                                     x_data_limit, x_annotation_limit)
            if i == 2:
                gain_ax.set_xlabel("XGBoost average split gain", fontsize=10.5, labelpad=5)
                effect_ax.set_xlabel("Hedges' g (MSA − PD)", fontsize=10.5, labelpad=5)
            else:
                gain_ax.tick_params(axis="x", labelbottom=False)
                effect_ax.tick_params(axis="x", labelbottom=False)

        shap_ax = fig.add_subplot(grid[3, :])
        _draw_joint_svm_shap_panel(shap_ax, svm_shap_importance_df)
        # Narrow and centre the bottom panel instead of stretching it across both columns.
        pos = shap_ax.get_position()
        shap_ax.set_position([pos.x0 + pos.width * 0.24, pos.y0, pos.width * 0.52, pos.height])

        fig.suptitle("Discovery-derived Feature Importance and Effect-size Replication",
                     fontsize=16.2, fontweight="bold", y=0.982)
        fig.text(0.22, 0.942, "Final Discovery-set XGBoost feature importance",
                 ha="center", va="center", fontsize=11, fontweight="bold", color="#333333")
        fig.text(0.71, 0.942, "Group-level effect sizes for the same locked features",
                 ha="center", va="center", fontsize=11, fontweight="bold", color="#333333")
        fig.text(0.012, 0.982, "a.", fontsize=16.2, fontweight="bold", va="top")

        fig.legend(handles=[
            plt.Line2D([0], [0], marker="o", color="#555555", mfc="white", ms=7,
                       lw=1.1, label="Discovery set (open circle)"),
            plt.Line2D([0], [0], marker="o", color="#555555", mfc="#555555", ms=7,
                       lw=1.1, label="Independent validation (filled circle)"),
            plt.Line2D([0], [0], marker="s", color=MODALITY_COLORS["EEG"], ms=7,
                       lw=0, label="EEG"),
            plt.Line2D([0], [0], marker="s", color=MODALITY_COLORS["EOG"], ms=7,
                       lw=0, label="EOG"),
            plt.Line2D([0], [0], marker="s", color=MODALITY_COLORS["EMG"], ms=7,
                       lw=0, label="EMG"),
        ], loc="lower center", ncol=5, frameon=False, fontsize=9.0,
           bbox_to_anchor=(0.52, 0.012), columnspacing=1.25, handletextpad=0.45)
        fig.text(0.60, 0.072,
                 "Effect direction is indicated by position relative to 0; horizontal lines = bootstrap 95% CIs. "
                 "* q<0.05, ** q<0.01, *** q<0.001 (BH-FDR across 30 selected features).",
                 ha="center", va="center", fontsize=8.3, color="#333333")
        fig.savefig(OUTPUT_DIR / "stacking_gain_and_effect_size_replication.png", dpi=JOINT_FIGURE_DPI,
                    bbox_inches="tight", facecolor="white")
        fig.savefig(OUTPUT_DIR / "stacking_gain_and_effect_size_replication.pdf", dpi=JOINT_FIGURE_DPI,
                    bbox_inches="tight", facecolor="white")
        plt.close(fig)


def save_effect_analysis_note():
    """Save a concise, traceable description of the effect-size workflow."""
    note = f"""Effect-size replication analysis

Feature selection
-----------------
For each modality (EEG, EOG, EMG), the Top {TOP_N_FEATURES} features were selected
solely by average XGBoost split gain from the final model refitted on the complete
Discovery set. The Independent validation set was not used to select, reorder, or
replace features.

Interpretation of gain
----------------------
XGBoost gain is the average reduction in the model loss produced by splits using a
feature. It is intrinsically non-negative and does not encode whether higher or
lower feature values favour MSA. Gain bars are coloured by modality (EEG purple,
EOG orange, EMG green) only for visual consistency; biological direction is
represented only by Hedges' g (MSA minus PD): positive = MSA > PD and negative =
MSA < PD. The fourth row shows SVM meta-model modality importance by mean absolute
SHAP using the same modality colours.

Effect-size calculation
-----------------------
For the locked 30-feature list, Hedges' g was calculated as MSA minus PD from raw,
finite feature values in each dataset. No imputation, z-scoring, or model-derived
feature transformation was applied. Positive g denotes MSA > PD; negative g denotes
MSA < PD.

Uncertainty and hypothesis tests
--------------------------------
Bootstrap 95% CIs were calculated using {N_EFFECT_BOOTSTRAP} stratified resamples
within MSA and PD. Two-sided Welch t-test P values are unadjusted. BH-FDR q values
were adjusted separately within the Discovery and Independent validation datasets
across all 30 selected features. The Discovery estimates are exploratory because the
features were selected using Discovery-set model importance; Independent validation
results provide the replication analysis.

Covariates
----------
This script performs unadjusted group comparisons because age/sex covariates are not
loaded from the NPZ inputs. If covariate-adjusted inference is required, use a
pre-specified regression model such as:
Feature ~ Diagnosis + Age + Sex
and apply BH-FDR correction to the Diagnosis-term P values across the same 30-feature
family.
"""
    (OUTPUT_DIR / "effect_size_analysis_notes.txt").write_text(
        note,
        encoding="utf-8",
    )


def _fit_final_svm(L1_discovery, y_discovery, final_svm_params, seed):
    """Reconstruct the EXACT final SVM fit called by the fixed model."""
    scaler = StandardScaler()
    scaled = scaler.fit_transform(L1_discovery)
    svm = SVC(**final_svm_params, probability=True, random_state=seed)
    svm.fit(scaled, y_discovery)
    return scaler, svm


def _fit_final_xgb(X_discovery_modality, y_discovery, params, seed):
    """Reconstruct fixed somnis.fit_xgb_and_predict but retain estimator for gain."""
    imputer = SimpleImputer(strategy="median")
    scaled = StandardScaler()
    x_imp = imputer.fit_transform(X_discovery_modality)
    x_scale = scaled.fit_transform(x_imp)
    n_negative = int(np.sum(y_discovery == 0))
    n_positive = int(np.sum(y_discovery == 1))
    model = xgb.XGBClassifier(
        **somnis.XGB_FIXED_PARAMS, **params,
        objective="binary:logistic", eval_metric="logloss",
        scale_pos_weight=n_negative / n_positive,
        random_state=seed, n_jobs=somnis.XGB_N_JOBS, verbosity=0,
    )
    model.fit(x_scale, y_discovery)
    # sklearn's SimpleImputer default drops columns that were 100% NaN in
    # Discovery, shifting XGBoost's f0/f1/... indices. Preserve the true mapping
    # from post-imputation model features back to original feature names.
    kept_columns = np.flatnonzero(np.isfinite(imputer.statistics_))
    if x_scale.shape[1] != len(kept_columns):
        raise RuntimeError("Unexpected feature removal in SimpleImputer")
    return model, kept_columns


def _verify_against_main_predictions(reference_csv, ids, actual_probabilities, tol):
    """Check final model identity against existing SOMNIS independent predictions."""
    df = pd.read_csv(reference_csv)
    expected_cols = {"Dataset", "SubjectID", "MSA_probability"}
    if not expected_cols.issubset(df.columns):
        raise ValueError(f"Reference predictions missing columns: {sorted(expected_cols-set(df.columns))}")
    df = df.loc[df["Dataset"] == "Independent validation set"].copy()
    ref_id = somnis.normalize_subject_ids(df["SubjectID"].to_numpy())
    tgt_id = somnis.normalize_subject_ids(ids)
    if len(ref_id) != len(tgt_id) or len(set(ref_id)) != len(ref_id):
        raise ValueError("Reference prediction subject count/IDs are inconsistent")
    lookup = dict(zip(ref_id, df["MSA_probability"].to_numpy(dtype=float)))
    if set(lookup) != set(tgt_id):
        raise ValueError("Reference prediction subjects differ from current input")
    ref_prob = np.array([lookup[sid] for sid in tgt_id], dtype=float)
    if not np.isfinite(ref_prob).all():
        raise ValueError("Reference probabilities contain invalid values")
    maxdiff = float(np.max(np.abs(ref_prob - actual_probabilities)))
    print(f"Model identity check: max |P_replay - P_main| = {maxdiff:.12g}", flush=True)
    if maxdiff > tol:
        raise RuntimeError(
            f"Replayed model probabilities differ from the fixed SOMNIS by {maxdiff:.8g} "
            f"> tolerance {tol}. DO NOT label these interpretation results as from "
            "the fixed model. Check data, seed, library versions, and source code."
        )
    return maxdiff


def run(args):
    global OUTPUT_DIR, RANDOM_SEED, N_EFFECT_BOOTSTRAP, SHAP_BACKGROUND_MAX
    OUTPUT_DIR = args.output_root / "stages"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RANDOM_SEED = args.seed
    N_EFFECT_BOOTSTRAP = args.effect_bootstrap
    SHAP_BACKGROUND_MAX = args.shap_background_max

    msa_file = args.msa_file or (args.input_root / "stages" / "earlyMSA.npz")
    pd_file = args.pd_file or (args.input_root / "stages" / "earlyPD.npz")
    somnis.EXPECT_ORIGINAL_COHORT = args.expect_original_cohort
    somnis.validate_input_feature_schema(msa_file, pd_file, "stages")
    with np.load(msa_file, allow_pickle=True) as m, np.load(pd_file, allow_pickle=True) as p:
        names = somnis.as_clean_strings(m["feature_names"])
        if not np.array_equal(names, somnis.as_clean_strings(p["feature_names"])):
            raise ValueError("Feature names differ between MSA and PD")
    somnis.MSA_FILE_PATH = msa_file
    somnis.PD_FILE_PATH = pd_file
    somnis.RANDOM_SEED = args.seed
    somnis.BOOTSTRAP_SEED = args.seed
    data = somnis.load_data()

    if args.check_only:
        print("Stage-resolved inputs validated. No model was fitted.")
        return

    xd = data["X_discovery"]
    yd = data["y_discovery"]
    xi = data["X_independent"]
    yi = data["y_independent"]
    masks = data["feature_groups"]
    group_names = data["group_names"]
    print("Replaying final Discovery-only model fit using fixed SOMNIS routines...", flush=True)
    final_svm_params, cv_auc, final_oof = somnis.tune_meta_svm_strict(
        X=xd, y=yd, feature_groups=masks, group_names=group_names,
        seed=args.seed + 900_000_000, stage_label="final_discovery_training")
    l1_discovery, l1_independent, final_base_params = somnis.build_strict_l1_features_and_holdout(
        X_fit=xd, y_fit=yd, X_holdout=xi,
        feature_groups=masks, group_names=group_names,
        seed=args.seed + 950_000_000, stage_label="final_discovery_fit")
    svm_scaler, svm_model = _fit_final_svm(l1_discovery, yd, final_svm_params, args.seed)
    independent_probability = svm_model.predict_proba(svm_scaler.transform(l1_independent))[:, 1]
    verified = False
    max_prob_delta = None
    if args.reference_predictions is not None:
        if not args.reference_predictions.is_file():
            raise FileNotFoundError(args.reference_predictions)
        max_prob_delta = _verify_against_main_predictions(
            args.reference_predictions, data["independent_ids"],
            independent_probability, args.probability_tolerance)
        verified = True
    else:
        print("WARNING: No --reference-predictions given; model identity has not been "
              "numerically verified against the published SOMNIS run.", flush=True)
    # Threshold rule is NOT redefined: simply report the threshold selected by the
    # frozen main model from the same strict Discovery-only OOF probabilities.
    final_threshold, threshold_rule = somnis.select_threshold_from_training_oof(yd, final_oof)
    print(f"Fixed main-model threshold policy gives threshold {final_threshold:.6f} ({threshold_rule}).", flush=True)

    gain_tables = {}
    for mask, modality in zip(masks, group_names):
        model, kept_columns = _fit_final_xgb(
            xd[:, mask], yd, final_base_params[modality], args.seed)
        original_indices = np.flatnonzero(mask)
        if len(kept_columns) < len(original_indices):
            print(f"{modality}: {len(original_indices)-len(kept_columns)} all-NaN "
                  "Discovery columns omitted by sklearn imputer; corrected "
                  "XGBoost gain feature-index mapping.", flush=True)
        gain_tables[modality] = extract_xgb_gain_importance(
            model, names[original_indices[kept_columns]],
            original_indices[kept_columns], top_n=TOP_N_FEATURES)
        # Restore within-modality indices after all-NaN Discovery columns are
        # dropped by sklearn's median imputer. Global indices were already right.
        kept_rank = gain_tables[modality]['Feature_index_within_modality'].to_numpy(dtype=int)
        gain_tables[modality]['Feature_index_within_modality'] = kept_columns[kept_rank]
    gain_df = pd.concat(
        [t.assign(Modality=k) for k,t in gain_tables.items()],
        ignore_index=True)[["Modality","Rank","Global_feature_index",
                       "Feature_index_within_modality","Feature","Gain"]]
    gain_df.to_csv(OUTPUT_DIR / "xgb_gain_top10_features.csv", index=False, encoding="utf-8-sig")

    shap_values = compute_exact_interventional_shap_values(
        scaler=svm_scaler, model=svm_model,
        X_explain=l1_discovery, X_background=l1_discovery)
    shap_df = pd.DataFrame({"Modality": group_names,
                            "Mean_abs_SHAP": np.mean(np.abs(shap_values), axis=0),
                            "Mean_SHAP": np.mean(shap_values, axis=0)})
    shap_df.to_csv(OUTPUT_DIR / "svm_meta_model_shap_importance.csv", index=False, encoding="utf-8-sig")

    effect_df = compute_selected_feature_effect_sizes(
        gain_tables, xd, yd, xi, yi)
    effect_df.to_csv(OUTPUT_DIR / "selected_top30_feature_effect_sizes_long.csv",
                     index=False, encoding="utf-8-sig")
    make_effect_size_wide_table(effect_df).to_csv(
        OUTPUT_DIR / "selected_top30_feature_effect_sizes_wide.csv",
        index=False, encoding="utf-8-sig")

    save_interpretability_figure(gain_tables["EEG"], gain_tables["EOG"],
                                 gain_tables["EMG"], shap_df)
    save_joint_gain_effect_figure(gain_tables, effect_df, shap_df)
    save_effect_analysis_note()

    metadata = {
        "source_model": "models/somnis_stacking.py",
        "model_representation": "stages",
        "seed": args.seed,
        "n_discovery": len(yd), "n_independent": len(yi),
        "final_svm_params": final_svm_params,
        "final_base_params": final_base_params,
        "final_cv_auc": float(cv_auc),
        "fixed_threshold_from_oof": float(final_threshold),
        "fixed_threshold_rule": threshold_rule,
        "reference_prediction_file_checked": bool(verified),
        "max_independent_probability_difference": max_prob_delta,
        "important": "Exploratory unadjusted feature effects. Do not share patient-level NPZs.",
    }
    (OUTPUT_DIR / "interpretation_run_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Finished. Figures and aggregate tables: {OUTPUT_DIR}")


def cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=Path("data/datasets/early"))
    parser.add_argument("--msa-file", type=Path)
    parser.add_argument("--pd-file", type=Path)
    parser.add_argument("--output-root", type=Path, default=Path("results/interpretation"))
    parser.add_argument("--reference-predictions", type=Path,
                        help="Fixed main-model independent predictions CSV, for strict identity verification")
    parser.add_argument("--probability-tolerance", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--effect-bootstrap", type=int, default=3000)
    parser.add_argument("--shap-background-max", type=int, default=100)
    parser.add_argument("--expect-original-cohort", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if (args.msa_file is None) != (args.pd_file is None):
        parser.error("--msa-file and --pd-file must be supplied together")
    if args.effect_bootstrap < 1 or args.shap_background_max < 1:
        parser.error("--effect-bootstrap and --shap-background-max must be >= 1")
    if args.probability_tolerance < 0:
        parser.error("--probability-tolerance must be non-negative")
    run(args)


if __name__ == "__main__":
    cli()
