#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SOMNIS whole-night *epoch feature mean* baseline (MSA and PD).

The whole-night comparator follows the archived MSA_Avg_merge_files.py:
  1. Extract precisely the SAME 278 PSG features from each 30-s epoch as
     extract_stage_features.py (or load its already-extracted epoch files).
  2. Average each feature across ALL available full-night epochs, WITHOUT
     reading GSSC predictions or stratifying by W/N1/N2/N3/REM.
  3. Save one 278-vector per participant, preserving canonical feature order.

Example:
  python features/extract_whole_night_features.py \\
    --preprocess-root data/preprocess --output-root data/features --groups MSA PD

Input:  data/preprocess/MSA/MSA_001_processed.npz
Output: data/features/wholenight/MSA_wostages.npz
Shared epoch cache: data/features/epoch/MSA/MSA_001_features.npz
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

from extract_stage_features import (  # import beside this script, no extra module needed
    EPOCH_FEATURE_NAMES, average_all_epochs, check_epoch_archive,
    extract_epoch_archive, feature_type, list_processed_files,
)

LOG = logging.getLogger("somnis.features.whole_night")


def run_whole_group(preprocess_root: Path, output_root: Path, group: str,
                    overwrite_epochs: bool, invalid_policy: str,
                    fail_fast: bool = False) -> tuple[int, int]:
    vectors: list[np.ndarray] = []
    ids: list[str] = []
    failed = 0
    for subject_id, input_file in list_processed_files(preprocess_root, group):
        epoch_path = output_root / "epoch" / group / f"{subject_id}_features.npz"
        try:
            extract_epoch_archive(input_file, epoch_path, invalid_policy, overwrite_epochs)
            epoch_features, _ = check_epoch_archive(epoch_path)
            vectors.append(average_all_epochs(epoch_features))
            ids.append(subject_id)
            LOG.info("Whole-night mean %-20s epochs=%d", subject_id, len(epoch_features))
        except Exception:
            failed += 1
            LOG.exception("Failed whole-night aggregation for %s", subject_id)
            if fail_fast:
                raise
    if not ids:
        raise RuntimeError(f"No successful whole-night inputs for {group}")
    output = output_root / "wholenight" / f"{group}_wostages.npz"
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output, X=np.vstack(vectors),
        feature_names=np.asarray(EPOCH_FEATURE_NAMES),
        feature_types=np.asarray([feature_type(x) for x in EPOCH_FEATURE_NAMES]),
        subject_ids=np.asarray(ids), representation="whole-night-epoch-mean",
    )
    LOG.info("Saved %s: shape=(%d,278), failed=%d", output, len(ids), failed)
    return len(ids), failed


def arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--preprocess-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--groups", nargs="+", default=["MSA", "PD"])
    parser.add_argument("--invalid-policy", choices=["nan", "legacy-zero"], default="nan")
    parser.add_argument("--overwrite-epochs", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = arguments(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")
    failed = 0
    for group in args.groups:
        _, bad = run_whole_group(args.preprocess_root, args.output_root, group,
                                 args.overwrite_epochs, args.invalid_policy,
                                 args.fail_fast)
        failed += bad
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
