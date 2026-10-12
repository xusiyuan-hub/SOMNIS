#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Append EEG/EOG/EMG column labels to SOMNIS 1390- or 278-feature NPZ files.

Accept both stage-resolved (e.g. W_F3_std) and whole-night (e.g. F3_std)
feature names. Keeps every numerical X value and its column order unchanged.
Normalizes subject IDs to GROUP_<ID> once, without adding double prefixes.

Examples:
  python features/add_modality_labels.py --input-root data/features \\
    --output-root data/datasets/all --groups MSA PD --mode both

Inputs:  data/features/stages/MSA.npz
         data/features/wholenight/MSA_wostages.npz
Outputs: data/datasets/all/stages/MSA_ch.npz
         data/datasets/all/wholenight/MSA_wostages_ch.npz
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
from extract_stage_features import (ALL_CHANNELS, CHANNEL_MODALITY, EPOCH_FEATURE_NAMES,
                                    STAGE_FEATURE_NAMES, STAGES, feature_type)
LOG = logging.getLogger("somnis.features.modality_labels")


def strings(x) -> np.ndarray:
    values = np.asarray(x).reshape(-1)
    return np.asarray([v.decode("utf8").strip() if isinstance(v, bytes)
                       else str(v).strip() for v in values], dtype=str)


def modality_of(name: str, mode: str) -> str:
    """Parse the actual channel token; 'R' stage must not equal EOG channel R."""
    tokens = name.split("_")
    if mode == "stages":
        if len(tokens) < 3 or tokens[0] not in STAGES:
            raise ValueError(f"Bad stage-resolved feature name: {name}")
        channel = tokens[1]
    else:
        if len(tokens) < 2:
            raise ValueError(f"Bad whole-night feature name: {name}")
        channel = tokens[0]
    if channel not in ALL_CHANNELS:
        raise ValueError(f"Unknown channel {channel!r} in {name!r}")
    return CHANNEL_MODALITY[channel]


def normalize_subject_id(raw_id: str, group: str) -> str:
    sid = raw_id.strip()
    if not sid:
        raise ValueError("Blank subject ID")
    if sid.upper().startswith(group.upper() + "_"):
        return group + "_" + sid[len(group) + 1:]
    if sid.upper().startswith(("MSA_", "PD_")):
        raise ValueError(f"Unexpected group prefix in {sid!r} for {group}")
    return group + "_" + sid


def add_labels(input_file: Path, output_file: Path, mode: str, group: str,
               overwrite: bool = False) -> None:
    if output_file.exists() and not overwrite:
        raise FileExistsError(f"Output exists; pass --overwrite to replace: {output_file}")
    with np.load(input_file, allow_pickle=False) as d:
        X = np.asarray(d["X"], dtype=float)
        names = strings(d["feature_names"])
        ids = strings(d["subject_ids"])
    expected = STAGE_FEATURE_NAMES if mode == "stages" else EPOCH_FEATURE_NAMES
    if X.ndim != 2 or X.shape[1] != len(expected):
        raise ValueError(f"{input_file} wrong X shape {X.shape}; expected (_, {len(expected)})")
    if len(ids) != len(X) or len(set(ids)) != len(ids):
        raise ValueError(f"Wrong subject ID count or duplicated IDs in {input_file}")
    if not np.array_equal(names, expected):
        raise ValueError(f"Unexpected feature names/order in {input_file}")
    normalized = np.asarray([normalize_subject_id(x, group) for x in ids])
    if len(set(normalized)) != len(normalized):
        raise ValueError("Duplicate subject IDs after normalization")
    channels = np.asarray([modality_of(x, mode) for x in names])
    feature_types = np.asarray([feature_type(x) for x in names])
    if any(not np.any(channels == type_) for type_ in ("EEG", "EOG", "EMG")):
        raise ValueError("Missing modality label")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_file, X=X, feature_names=names, feature_types=feature_types,
                        channels=channels, subject_ids=normalized)
    LOG.info("Saved %s: X=%s | EEG=%d EOG=%d EMG=%d", output_file, X.shape,
             (channels == "EEG").sum(), (channels == "EOG").sum(),
             (channels == "EMG").sum())


def arguments(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-root", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--groups", nargs="+", default=["MSA", "PD"])
    p.add_argument("--mode", choices=["stages", "wholenight", "both"], default="both")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = arguments(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    modes = ("stages", "wholenight") if args.mode == "both" else (args.mode,)
    for mode in modes:
        for group in args.groups:
            filename = f"{group}.npz" if mode == "stages" else f"{group}_wostages.npz"
            outfile = f"{group}_ch.npz" if mode == "stages" else f"{group}_wostages_ch.npz"
            add_labels(args.input_root / mode / filename,
                       args.output_root / mode / outfile, mode, group, args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
