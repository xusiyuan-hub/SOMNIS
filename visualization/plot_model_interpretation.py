# -*- coding: utf-8 -*-
"""Replot SOMNIS explanation figures FROM SAVED AGGREGATE CSVs.

No model fitting and no original patient-level NPZ inputs required.

Example:
    python visualization/plot_model_interpretation.py \
      --analysis-dir results/interpretation/stages --which all
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from analyses import model_interpretation as interp


def cli(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--analysis-dir", type=Path,
                   default=Path("results/interpretation/stages"))
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--which", choices=("all", "joint", "importance", "effect"),
                   default="all")
    args = p.parse_args(argv)
    input_dir = args.analysis_dir
    output_dir = args.output_dir or input_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    interp.OUTPUT_DIR = output_dir

    gain = pd.read_csv(input_dir / "xgb_gain_top10_features.csv")
    shap = pd.read_csv(input_dir / "svm_meta_model_shap_importance.csv")
    effect = (pd.read_csv(input_dir / "selected_top30_feature_effect_sizes_long.csv")
              if args.which in ("all", "joint", "effect") else None)
    needed_gain = {"Modality", "Rank", "Feature", "Gain", "Global_feature_index"}
    needed_shap = {"Modality", "Mean_abs_SHAP"}
    needed_effect = {"Dataset", "Modality", "Feature", "Hedges_g_MSA_minus_PD",
                     "Hedges_g_CI_lower", "Hedges_g_CI_upper", "Q_value_BH_FDR_30_features"}
    tables = [(gain, needed_gain, "gain"), (shap, needed_shap, "SHAP")]
    if effect is not None:
        tables.append((effect, needed_effect, "effect"))
    for table, required, label in tables:
        missing = required.difference(table.columns)
        if missing:
            raise ValueError(f"{label} CSV missing required columns: {sorted(missing)}")
    if set(shap["Modality"].astype(str)) != {"EEG", "EOG", "EMG"}:
        raise ValueError("SHAP table must contain EEG, EOG and EMG")
    gains = {m: gain.loc[gain["Modality"] == m].sort_values("Rank").copy()
             for m in ("EEG", "EOG", "EMG")}
    if any(len(t) != 10 for t in gains.values()):
        raise ValueError("Expected exactly 10 Gain features per modality")
    if args.which in ("all", "importance"):
        interp.save_interpretability_figure(gains["EEG"], gains["EOG"],
                                            gains["EMG"], shap)
    if args.which in ("all", "joint"):
        interp.save_joint_gain_effect_figure(gains, effect, shap)
    if args.which in ("all", "effect"):
        interp.save_effect_size_replication_figure(effect)
    print(f"Rendered {args.which} figure(s) in {output_dir}. No model was trained.")


if __name__ == "__main__":
    cli()
