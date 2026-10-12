#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SOMNIS: pretrained GSSC sleep staging for MSA and PD selected PSG.

This script preserves the original call to the pretrained Greifswald Sleep
Stage Classifier, without cohort-specific fine-tuning. It accepts group-specific
selected EEGLAB .set recordings rather than the resampled NPZ feature input.

Example from repository root:
  python preprocessing/gssc_sleep_staging.py \\
    --input-root data/cut --output-root data/sleepstage/GSSC --groups MSA PD

Input:   data/cut/MSA/MSA_001_selected.set
Output:  data/sleepstage/GSSC/MSA/MSA_001_selected_gssc.csv
         (Stage column W, N1, N2, N3, R)
The filename is intentionally kept compatible with the archived MSA feature
extraction script, which expects MSA_<ID>_selected_gssc.csv.

Requires the third-party GSSC package and its published pretrained weights.
Refer to its upstream installation / model-download instructions.
"""

from __future__ import annotations

import argparse
import gc
import logging
from pathlib import Path

import mne
import numpy as np
import pandas as pd

LOG = logging.getLogger("somnis.gssc_sleep_staging")

# These match the channel configurations in the original GSSC MSA script.
GSSC_SCHEMES = [
    ("V1-A", ["F3-A2", "F4-A1", "C3-A2", "C4-A1", "O1-A2", "O2-A1"],
     ["LOC-A2", "ROC-A2"]),
    ("V2-M", ["F3-M2", "F4-M1", "C3-M2", "C4-M1", "O1-M2", "O2-M1"],
     ["E1-M2", "E2-M2"]),
    ("V3-Fp", ["Fp1-M2", "Fp2-M1", "C3-M2", "C4-M1", "O1-M2", "O2-M1"],
     ["E1-M2", "E2-M2"]),
]
STAGE_LABELS = {0: "W", 1: "N1", 2: "N2", 3: "N3", 4: "R"}


def arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run pretrained GSSC on selected MSA/PD EEGLAB PSG files."
    )
    parser.add_argument("--input-root", type=Path, required=True,
                        help="Parent directory containing MSA/ and PD/ selected .set files")
    parser.add_argument("--output-root", type=Path, required=True,
                        help="Parent directory containing MSA/ and PD/ staging CSVs")
    parser.add_argument("--groups", nargs="+", default=["MSA", "PD"])
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--chunk-n", type=int, default=100,
                        help="Number of epochs per inference chunk (original: 100)")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--strict-epoch-count", action="store_true",
                        help="Fail if GSSC and 30-s feature epoch counts disagree")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def detect_channels(raw: mne.io.BaseRaw) -> tuple[list[str], list[str], str]:
    available = {}
    for ch in raw.ch_names:
        canonical = ch.strip().upper()
        if canonical in available:
            raise ValueError(f"Duplicate channel name after normalization: {ch}")
        available[canonical] = ch
    for label, eeg, eog in GSSC_SCHEMES:
        required = eeg + eog
        if all(name.upper() in available for name in required):
            return ([available[name.upper()] for name in eeg],
                    [available[name.upper()] for name in eog], label)
    raise ValueError(f"No complete GSSC EEG+EOG configuration. Channels: {raw.ch_names}")


def choose_device(mode: str) -> bool:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("GSSC inference requires PyTorch. See requirements.") from exc
    cuda_available = torch.cuda.is_available()
    if mode == "cuda" and not cuda_available:
        raise RuntimeError("--device cuda was selected but CUDA is unavailable")
    return cuda_available if mode == "auto" else mode == "cuda"


def run_one(set_path: Path, output_path: Path, inferer,
            strict_epoch_count: bool, overwrite: bool) -> str:
    if output_path.exists() and not overwrite:
        LOG.info("SKIP existing: %s", output_path)
        return "skipped"

    raw = mne.io.read_raw_eeglab(set_path, preload=True, verbose="ERROR")
    try:
        eeg_names, eog_names, scheme = detect_channels(raw)
        stages, times, probabilities = inferer.mne_infer(
            raw, eeg=eeg_names, eog=eog_names,
            eeg_drop=True, eog_drop=True, filter=True,
        )
        stages = np.asarray(stages, dtype=int).reshape(-1)
        times = np.asarray(times).reshape(-1)
        probabilities = np.asarray(probabilities)
        if len(stages) == 0:
            raise ValueError("GSSC returned no sleep-stage predictions")
        if len(times) != len(stages):
            raise ValueError("GSSC returned mismatched times and stage counts")
        if probabilities.ndim != 2 or probabilities.shape != (len(stages), 5):
            raise ValueError(
                f"Expected probabilities (n_epochs,5), received {probabilities.shape}"
            )
        if not all(stage in STAGE_LABELS for stage in stages):
            raise ValueError("GSSC produced unrecognized sleep-stage codes")
        if not np.isfinite(probabilities).all() or not np.isfinite(times).all():
            raise ValueError("GSSC outputs contain non-finite values")

        # Feature extraction uses floor(n_samples / (30 * fs)) from the SAME
        # selected .set file after resampling; compare with this expectation.
        expected_epochs = int(np.floor(raw.n_times / (30.0 * raw.info["sfreq"])))
        if len(stages) != expected_epochs:
            message = (f"GSSC epoch count={len(stages)} != expected 30-s epoch "
                       f"count={expected_epochs} for {set_path.name}; "
                       "check epoch alignment and selected-file versions")
            if strict_epoch_count:
                raise ValueError(message)
            LOG.warning(message)

        frame = pd.DataFrame({
            "Time_Sec": times,
            "Epoch_30s_Idx": np.arange(len(stages), dtype=int),
            "Stage": [STAGE_LABELS[int(code)] for code in stages],
            "Stage_Code": stages,
            "Confidence": probabilities.max(axis=1),
        })
        output_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = output_path.with_name(output_path.stem + ".partial.csv")
        try:
            frame.to_csv(tmp_path, index=False)
            tmp_path.replace(output_path)
        finally:
            tmp_path.unlink(missing_ok=True)
        LOG.info("OK %-27s scheme=%s epochs=%s -> %s",
                 set_path.name, scheme, len(stages), output_path)
        return "ok"
    finally:
        raw.close()


def main(argv: list[str] | None = None) -> int:
    args = arguments(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")
    if args.chunk_n < 1:
        raise ValueError("--chunk-n must be >= 1")

    # Enumerate files before loading GSSC weights, including in dry-run mode.
    jobs = []
    for group in args.groups:
        input_dir = args.input_root / group
        if not input_dir.is_dir():
            LOG.warning("Missing group directory: %s", input_dir)
            continue
        for set_path in sorted(input_dir.glob(f"{group}_*_selected.set")):
            csv_path = args.output_root / group / f"{set_path.stem}_gssc.csv"
            jobs.append((set_path, csv_path))
    LOG.info("Matched %d files from groups %s", len(jobs), args.groups)
    if not jobs:
        LOG.error("No matching selected .set files found")
        return 2
    if args.dry_run:
        for src, dst in jobs:
            LOG.info("DRY RUN %s -> %s", src, dst)
        return 0
    pending = [(src, dst) for src, dst in jobs if args.overwrite or not dst.exists()]
    if not pending:
        LOG.info("All %d GSSC outputs already exist", len(jobs))
        return 0

    # Import after --help and --dry-run so project layout can be tested even if
    # GSSC/PyTorch are not yet installed.
    from gssc.infer import EEGInfer
    use_cuda = choose_device(args.device)
    LOG.info("Loading pretrained GSSC on %s", "CUDA" if use_cuda else "CPU")
    inferer = EEGInfer(use_cuda=use_cuda, chunk_n=args.chunk_n)
    results = {"ok": 0, "skipped": 0, "failed": 0}
    for src, dst in jobs:
        try:
            state = run_one(src, dst, inferer, args.strict_epoch_count, args.overwrite)
            results[state] += 1
        except Exception:
            results["failed"] += 1
            LOG.exception("FAILED: %s", src)
        finally:
            gc.collect()
            if use_cuda:
                import torch
                torch.cuda.empty_cache()
    LOG.info("Done: OK=%d SKIP=%d FAILED=%d", results["ok"], results["skipped"], results["failed"])
    return 1 if results["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
