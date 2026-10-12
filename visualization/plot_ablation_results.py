r"""
Visualize complete modality- and sleep-stage-ablation performance with 95% CIs.

Inputs
------
results/ablation/sleep_stage/sleep_stage_ablation_summary.csv
results/ablation/modality/modality_ablation_summary.csv

Output
------
1. ablation_all_metrics_with_95CI.png/.pdf/.svg
2. modality_ablation_all_metrics_with_95CI.png/.pdf/.svg
3. sleep_stage_ablation_all_metrics_with_95CI.png/.pdf/.svg
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# =============================================================================
# 0. Paths from CLI (no fixed Windows directories)
# =============================================================================
import argparse
STAGE_CSV = None
MODALITY_CSV = None
OUTPUT_DIR = None


# =============================================================================
# 1. Plot settings
# =============================================================================
METRICS = ["AUC", "ACC", "BAC", "Sens", "Spec", "PPV", "NPV"]

METRIC_LABELS = {
    "AUC": "AUC",
    "ACC": "Accuracy",
    "BAC": "Balanced accuracy",
    "Sens": "Sensitivity",
    "Spec": "Specificity",
    "PPV": "PPV",
    "NPV": "NPV",
}

MODALITY_ORDER = ["Full", "-EEG", "-EOG", "-EMG"]
STAGE_ORDER = ["Full", "-W", "-N1", "-N2", "-N3", "-R"]

DISCOVERY_DATASET = "Discovery set (outer-fold OOF)"
INDEPENDENT_DATASET = "Independent validation set"

MODALITY_COLORS = {
    "Full": "#222222",
    "-EEG": "#4C78A8",
    "-EOG": "#F58518",
    "-EMG": "#54A24B",
}

STAGE_COLORS = {
    "Full": "#222222",
    "-W": "#4C78A8",
    "-N1": "#F58518",
    "-N2": "#54A24B",
    "-N3": "#B279A2",
    "-R": "#E45756",
}

POINT_SIZE = 5.3
CI_LINEWIDTH = 1.45
CAP_SIZE = 2.5

MODALITY_OFFSET_STEP = 0.13
STAGE_OFFSET_STEP = 0.085

X_MIN = 0.40
X_MAX = 1.01
X_TICKS = np.arange(0.4, 1.01, 0.1)


# =============================================================================
# 2. Data validation
# =============================================================================
def load_and_validate(csv_path, expected_conditions, analysis_name):
    if not csv_path.exists():
        raise FileNotFoundError(f"{analysis_name} CSV not found:\n{csv_path}")

    df = pd.read_csv(csv_path)

    required = ["Condition", "Dataset"]
    for metric in METRICS:
        required.extend(
            [
                metric,
                f"{metric}_CI_lower",
                f"{metric}_CI_upper",
            ]
        )

    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(
            f"{analysis_name}: missing required columns:\n{missing}\n\n"
            f"Available columns:\n{df.columns.tolist()}"
        )

    df["Condition"] = df["Condition"].astype(str).str.strip()
    df["Dataset"] = df["Dataset"].astype(str).str.strip()

    expected_datasets = {DISCOVERY_DATASET, INDEPENDENT_DATASET}
    missing_datasets = expected_datasets.difference(set(df["Dataset"].unique()))
    if missing_datasets:
        raise ValueError(
            f"{analysis_name}: missing Dataset rows: {sorted(missing_datasets)}"
        )

    missing_conditions = set(expected_conditions).difference(
        set(df["Condition"].unique())
    )
    if missing_conditions:
        raise ValueError(
            f"{analysis_name}: missing conditions: {sorted(missing_conditions)}"
        )

    numeric_cols = []
    for metric in METRICS:
        numeric_cols.extend(
            [metric, f"{metric}_CI_lower", f"{metric}_CI_upper"]
        )

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if df[numeric_cols].isna().any().any():
        bad_cols = df[numeric_cols].columns[
            df[numeric_cols].isna().any()
        ].tolist()
        raise ValueError(
            f"{analysis_name}: NaN/non-numeric values found in {bad_cols}"
        )

    for metric in METRICS:
        est = df[metric].to_numpy(dtype=float)
        low = df[f"{metric}_CI_lower"].to_numpy(dtype=float)
        high = df[f"{metric}_CI_upper"].to_numpy(dtype=float)

        if np.any(low > high):
            raise ValueError(
                f"{analysis_name}: {metric} contains CI_lower > CI_upper."
            )

        outside = (est < low - 1e-12) | (est > high + 1e-12)
        if np.any(outside):
            print(
                f"WARNING: {analysis_name} / {metric}: "
                f"{int(np.sum(outside))} estimate(s) lie outside their 95% CI."
            )

    return df


def get_dataset_rows(df, dataset_name, condition_order):
    subset = df.loc[df["Dataset"] == dataset_name].copy()

    duplicated = subset["Condition"].duplicated(keep=False)
    if duplicated.any():
        dup_names = sorted(subset.loc[duplicated, "Condition"].unique())
        raise ValueError(
            f"{dataset_name}: duplicated condition rows found: {dup_names}"
        )

    subset = subset.set_index("Condition").reindex(condition_order)

    missing = subset.index[subset["Dataset"].isna()].tolist()
    if missing:
        raise ValueError(
            f"{dataset_name}: missing condition rows: {missing}"
        )

    return subset


# =============================================================================
# 3. Forest-style plotting
# =============================================================================
def draw_forest_panel(
    ax,
    df,
    dataset_name,
    condition_order,
    colors,
    title,
    panel_label,
    offset_step,
    show_y_labels=True,
):
    plot_df = get_dataset_rows(
        df=df,
        dataset_name=dataset_name,
        condition_order=condition_order,
    )

    base_y = np.arange(len(METRICS))[::-1].astype(float)

    offsets = (
        np.arange(len(condition_order))
        - (len(condition_order) - 1) / 2.0
    ) * offset_step

    for condition_index, condition in enumerate(condition_order):
        row = plot_df.loc[condition]
        color = colors[condition]

        for metric_index, metric in enumerate(METRICS):
            estimate = float(row[metric])
            lower = float(row[f"{metric}_CI_lower"])
            upper = float(row[f"{metric}_CI_upper"])

            xerr = np.array(
                [
                    [max(0.0, estimate - lower)],
                    [max(0.0, upper - estimate)],
                ]
            )

            y = base_y[metric_index] + offsets[condition_index]

            ax.errorbar(
                estimate,
                y,
                xerr=xerr,
                fmt="o",
                markersize=POINT_SIZE,
                color=color,
                ecolor=color,
                elinewidth=CI_LINEWIDTH,
                capsize=CAP_SIZE,
                capthick=CI_LINEWIDTH,
                markeredgewidth=0,
                label=condition if metric_index == 0 else None,
                zorder=3 + condition_index,
            )

    for y in np.arange(0.5, len(METRICS) - 0.5, 1.0):
        ax.axhline(
            y,
            color="#E5E5E5",
            linewidth=0.65,
            linestyle=":",
            zorder=0,
        )

    ax.set_xlim(X_MIN, X_MAX)
    ax.set_xticks(X_TICKS)
    ax.set_xlabel("Performance (95% CI)")
    ax.set_title(title, pad=10)

    ax.set_yticks(base_y)
    if show_y_labels:
        ax.set_yticklabels([METRIC_LABELS[m] for m in METRICS])
    else:
        ax.set_yticklabels([])

    ax.set_ylim(-0.55, len(METRICS) - 0.45)

    ax.grid(
        axis="x",
        color="#E2E2E2",
        linestyle=":",
        linewidth=0.7,
        zorder=0,
    )
    ax.set_axisbelow(True)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.9)
    ax.spines["bottom"].set_linewidth(0.9)

    ax.tick_params(axis="both", which="major", length=3.5, width=0.9)

    ax.text(
        -0.12,
        1.055,
        panel_label,
        transform=ax.transAxes,
        fontsize=14.5,
        fontweight="bold",
        ha="left",
        va="bottom",
        clip_on=False,
    )


def make_legend_handles(condition_order, colors):
    return [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="-",
            linewidth=1.4,
            markersize=5.5,
            color=colors[c],
            label=c,
        )
        for c in condition_order
    ]


# =============================================================================
# 4. Combined 2 x 2 figure
# =============================================================================
def save_combined_figure(modality_df, stage_df):
    with plt.rc_context(
        {
            "font.family": "Arial",
            "font.size": 10.5,
            "axes.titlesize": 12.5,
            "axes.titleweight": "bold",
            "axes.labelsize": 11.2,
            "xtick.labelsize": 9.8,
            "ytick.labelsize": 10.3,
            "legend.fontsize": 9.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    ):
        fig, axes = plt.subplots(
            2,
            2,
            figsize=(13.8, 10.8),
            facecolor="white",
            sharex=True,
        )

        draw_forest_panel(
            axes[0, 0],
            modality_df,
            DISCOVERY_DATASET,
            MODALITY_ORDER,
            MODALITY_COLORS,
            "Discovery Set",
            "a.",
            MODALITY_OFFSET_STEP,
            True,
        )

        draw_forest_panel(
            axes[0, 1],
            modality_df,
            INDEPENDENT_DATASET,
            MODALITY_ORDER,
            MODALITY_COLORS,
            "Independent Validation Set",
            "b.",
            MODALITY_OFFSET_STEP,
            False,
        )

        draw_forest_panel(
            axes[1, 0],
            stage_df,
            DISCOVERY_DATASET,
            STAGE_ORDER,
            STAGE_COLORS,
            "Discovery Set",
            "c.",
            STAGE_OFFSET_STEP,
            True,
        )

        draw_forest_panel(
            axes[1, 1],
            stage_df,
            INDEPENDENT_DATASET,
            STAGE_ORDER,
            STAGE_COLORS,
            "Independent Validation Set",
            "d.",
            STAGE_OFFSET_STEP,
            False,
        )

        fig.text(
            0.5,
            0.982,
            "Leave-one-modality-out Ablation of SOMNIS",
            ha="center",
            va="top",
            fontsize=15.5,
            fontweight="bold",
        )

        fig.text(
            0.5,
            0.505,
            "Sleep-stage Leave-one-stage-out Ablation of SOMNIS",
            ha="center",
            va="bottom",
            fontsize=15.5,
            fontweight="bold",
        )

        axes[0, 1].legend(
            handles=make_legend_handles(MODALITY_ORDER, MODALITY_COLORS),
            title="Condition",
            loc="lower right",
            frameon=False,
            ncol=2,
            handlelength=1.7,
            columnspacing=1.0,
            labelspacing=0.35,
        )

        axes[1, 1].legend(
            handles=make_legend_handles(STAGE_ORDER, STAGE_COLORS),
            title="Condition",
            loc="lower right",
            frameon=False,
            ncol=2,
            handlelength=1.7,
            columnspacing=1.0,
            labelspacing=0.35,
        )

        fig.subplots_adjust(
            left=0.115,
            right=0.985,
            bottom=0.075,
            top=0.935,
            wspace=0.12,
            hspace=0.36,
        )

        stem = OUTPUT_DIR / "ablation_all_metrics_with_95CI"

        fig.savefig(
            stem.with_suffix(".png"),
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
            transparent=False,
        )
        fig.savefig(
            stem.with_suffix(".pdf"),
            bbox_inches="tight",
            facecolor="white",
            transparent=False,
        )
        fig.savefig(
            stem.with_suffix(".svg"),
            bbox_inches="tight",
            facecolor="white",
            transparent=False,
        )
        plt.close(fig)

        print("\nSaved combined figure:")
        for suffix in [".png", ".pdf", ".svg"]:
            print(f"  {stem.with_suffix(suffix)}")


# =============================================================================
# 5. Separate 1 x 2 figures
# =============================================================================
def save_single_analysis_figure(
    df,
    condition_order,
    colors,
    analysis_title,
    filename_stem,
    offset_step,
):
    with plt.rc_context(
        {
            "font.family": "Arial",
            "font.size": 10.5,
            "axes.titlesize": 12.5,
            "axes.titleweight": "bold",
            "axes.labelsize": 11.2,
            "xtick.labelsize": 9.8,
            "ytick.labelsize": 10.3,
            "legend.fontsize": 9.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    ):
        fig, axes = plt.subplots(
            1,
            2,
            figsize=(13.5, 5.8),
            facecolor="white",
            sharex=True,
        )

        draw_forest_panel(
            axes[0],
            df,
            DISCOVERY_DATASET,
            condition_order,
            colors,
            "Discovery Set",
            "a.",
            offset_step,
            True,
        )

        draw_forest_panel(
            axes[1],
            df,
            INDEPENDENT_DATASET,
            condition_order,
            colors,
            "Independent Validation Set",
            "b.",
            offset_step,
            False,
        )

        axes[1].legend(
            handles=make_legend_handles(condition_order, colors),
            title="Condition",
            loc="lower right",
            frameon=False,
            ncol=2,
            handlelength=1.7,
            columnspacing=1.0,
            labelspacing=0.35,
        )

        fig.suptitle(
            analysis_title,
            fontsize=15.5,
            fontweight="bold",
            y=0.985,
        )

        fig.subplots_adjust(
            left=0.115,
            right=0.985,
            bottom=0.12,
            top=0.87,
            wspace=0.12,
        )

        stem = OUTPUT_DIR / filename_stem

        fig.savefig(
            stem.with_suffix(".png"),
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
            transparent=False,
        )
        fig.savefig(
            stem.with_suffix(".pdf"),
            bbox_inches="tight",
            facecolor="white",
            transparent=False,
        )
        fig.savefig(
            stem.with_suffix(".svg"),
            bbox_inches="tight",
            facecolor="white",
            transparent=False,
        )
        plt.close(fig)

        print(f"\nSaved {analysis_title}:")
        for suffix in [".png", ".pdf", ".svg"]:
            print(f"  {stem.with_suffix(suffix)}")


# =============================================================================
# 6. Main
# =============================================================================
def main(argv=None):
    """Read SAVED aggregate performance only; never fit or access patient NPZ."""
    global OUTPUT_DIR, STAGE_CSV, MODALITY_CSV, X_MIN, X_MAX, X_TICKS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-root', type=Path, default=Path('results/ablation'),
                        help='Root containing modality/ and sleep_stage/ results')
    parser.add_argument('--modality-csv', type=Path, default=None)
    parser.add_argument('--stage-csv', type=Path, default=None)
    parser.add_argument('--output-dir', type=Path, default=Path('results/ablation/visualization'))
    parser.add_argument('--which', choices=['both', 'modality', 'sleep_stage'], default='both')
    parser.add_argument('--xmin', type=float, default=0.40)
    parser.add_argument('--xmax', type=float, default=1.01)
    args = parser.parse_args(argv)
    if not 0 <= args.xmin < args.xmax:
        parser.error('--xmin must be less than --xmax, and non-negative')
    OUTPUT_DIR = args.output_dir
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    STAGE_CSV = args.stage_csv or (args.input_root / 'sleep_stage' / 'sleep_stage_ablation_summary.csv')
    MODALITY_CSV = args.modality_csv or (args.input_root / 'modality' / 'modality_ablation_summary.csv')
    X_MIN, X_MAX = args.xmin, args.xmax
    X_TICKS = np.linspace(X_MIN, X_MAX, 7)
    modality_df, stage_df = None, None
    if args.which in ('both', 'modality'):
        modality_df = load_and_validate(MODALITY_CSV, MODALITY_ORDER, 'Modality ablation')
    if args.which in ('both', 'sleep_stage'):
        stage_df = load_and_validate(STAGE_CSV, STAGE_ORDER, 'Sleep-stage ablation')
    if args.which == 'both':
        save_combined_figure(modality_df, stage_df)
    if modality_df is not None:
        save_single_analysis_figure(
            df=modality_df, condition_order=MODALITY_ORDER,
            colors=MODALITY_COLORS,
            analysis_title='Leave-one-modality-out Ablation of SOMNIS',
            filename_stem='modality_ablation_all_metrics_with_95CI',
            offset_step=MODALITY_OFFSET_STEP)
    if stage_df is not None:
        save_single_analysis_figure(
            df=stage_df, condition_order=STAGE_ORDER, colors=STAGE_COLORS,
            analysis_title='Sleep-stage Leave-one-stage-out Ablation of SOMNIS',
            filename_stem='sleep_stage_ablation_all_metrics_with_95CI',
            offset_step=STAGE_OFFSET_STEP)
    print('Done. No model was trained; output:', OUTPUT_DIR)


if __name__ == '__main__':
    main()
