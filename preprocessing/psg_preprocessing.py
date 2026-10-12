#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SOMNIS: group-agnostic PSG preprocessing (MSA / PD).

Reproduces the *nominal* processing parameters in the original MSA script:
  EEG 0.3-45 Hz; EOG 0.1-15 Hz; chin EMG 10-70 Hz;
  zero-phase FIR; 50-Hz FIR notch; 100-Hz resampling.

The input files are previously selected/cropped EEGLAB recordings, **not**
uncut whole-night raw files. This program does not decide which intervals to
keep, does not run automatic sleep staging, and does not perform artifact
rejection. Input from MNE is normally in SI volts for EEG-type EEGLAB channels;
conversion to microvolts here is explicit rather than amplitude-guessing.

Run from the repository root:
  python preprocessing/psg_preprocessing.py \\
    --input-root data/cut --output-root data/preprocess --groups MSA PD

Expected input:  data/cut/MSA/MSA_001_selected.set
                 data/cut/PD/PD_001_selected.set
Output:          data/preprocess/MSA/MSA_001_processed.npz
                 data/preprocess/PD/PD_001_processed.npz

Output NPZ fields (compatible with SOMNIS feature extraction):
  data          shape=(10, n_samples), amplitudes in microvolts
  sfreq         scalar; default 100.0
  ch_names      ordered names: F3 F4 C3 C4 O1 O2 L R Ch1 Ch2
  subject_id    e.g. MSA_001
  units         "uV"
  processing_time, original_sfreq, input_data_unit, channel_scheme

Note: if published features were produced using a legacy amplitude heuristic,
validate this corrected, explicit units handling against the archived output
before claiming bit-for-bit reproduction of the published feature values.
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime, timezone
from pathlib import Path

import mne
import numpy as np

LOG = logging.getLogger("somnis.psg_preprocessing")

# Ordered aliases. Prefer complete acquisition scheme V1 or V2 where possible.
# These are the channel types and order required by the 1390-feature pipeline.
CHANNEL_SCHEMES = {
    "V1-A": {
        "F3": "F3-A2", "F4": "F4-A1", "C3": "C3-A2", "C4": "C4-A1",
        "O1": "O1-A2", "O2": "O2-A1", "L": "LOC-A2", "R": "ROC-A2",
        "Ch1": "Chin1-Chin2", "Ch2": "Chin3-Chin2",
    },
    "V2-M": {
        "F3": "F3-M2", "F4": "F4-M1", "C3": "C3-M2", "C4": "C4-M1",
        "O1": "O1-M2", "O2": "O2-M1", "L": "E1-M2", "R": "E2-M2",
        "Ch1": "Chin 1-Chin 2", "Ch2": "Chin 3-Chin 2",
    },
}

CHANNEL_ALIASES = {
    name: [scheme[name] for scheme in CHANNEL_SCHEMES.values()]
    for name in CHANNEL_SCHEMES["V1-A"]
}
STANDARD_CHANNELS = ["F3", "F4", "C3", "C4", "O1", "O2", "L", "R", "Ch1", "Ch2"]
CHANNEL_TYPES = {
    name: ("eeg" if name in STANDARD_CHANNELS[:6] else
           "eog" if name in ("L", "R") else "emg")
    for name in STANDARD_CHANNELS
}
FILTER_BANDS = {"eeg": (0.3, 45.0), "eog": (0.1, 15.0), "emg": (10.0, 70.0)}


def arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess MSA and PD selected EEGLAB PSG recordings."
    )
    parser.add_argument("--input-root", type=Path, required=True,
                        help="Parent directory containing MSA/ and PD/ .set files")
    parser.add_argument("--output-root", type=Path, required=True,
                        help="Parent directory for group-specific processed NPZ files")
    parser.add_argument("--groups", nargs="+", default=["MSA", "PD"],
                        help="Group subdirectory names; default: MSA PD")
    parser.add_argument("--sampling-rate", type=float, default=100.,
                        help="Output sampling frequency (Hz); original study: 100")
    parser.add_argument("--notch-frequency", type=float, default=50.,
                        help="Power-line notch (Hz); original study: 50")
    parser.add_argument("--mne-data-unit", choices=["V", "uV"], default="V",
                        help="Unit of raw.get_data() before output. MNE EEGLAB EEG is "
                             "normally volts. Set uV only if independently verified.")
    parser.add_argument("--allow-mixed-aliases", action="store_true",
                        help="Permit mixed channel alias schemes when all ten channels exist")
    parser.add_argument("--overwrite", action="store_true",
                        help="Replace existing output NPZ files")
    parser.add_argument("--dry-run", action="store_true",
                        help="List input files without reading or processing")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def configure_logging(verbose: bool = False) -> None:
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")


def select_channels(raw: mne.io.BaseRaw, allow_mixed: bool) -> tuple[list[str], str]:
    """Match 10 channels in the correct canonical order; never silently drop one."""
    available = {}
    for name in raw.ch_names:
        key = name.strip().upper()
        if key in available:
            raise ValueError(f"Ambiguous duplicate channel name: {name}")
        available[key] = name

    # Complete V1/V2 match preferred over per-channel matching.
    for scheme_name, scheme in CHANNEL_SCHEMES.items():
        if all(alias.upper() in available for alias in scheme.values()):
            return [available[scheme[ch].upper()] for ch in STANDARD_CHANNELS], scheme_name

    if not allow_mixed:
        raise ValueError(
            "No complete V1-A or V2-M channel scheme found. "
            "Provide all ten required channels, or use --allow-mixed-aliases "
            "for data with a documented mixed reference montage. "
            f"Present channel names: {raw.ch_names}"
        )

    selected = []
    for name in STANDARD_CHANNELS:
        candidates = [available[alias.upper()] for alias in CHANNEL_ALIASES[name]
                      if alias.upper() in available]
        if len(candidates) != 1:
            raise ValueError(
                f"Channel {name} requires exactly one alias, found: {candidates}. "
                "Inspect the montage before proceeding."
            )
        selected.append(candidates[0])
    return selected, "mixed"


def validate_filter_settings(original_sfreq: float, output_sfreq: float,
                             notch_frequency: float) -> None:
    if original_sfreq <= 2 * max(70., notch_frequency):
        raise ValueError(f"Input sampling rate {original_sfreq:g} Hz is too low "
                         "for 70-Hz EMG high cutoff and requested notch")
    if output_sfreq <= 0:
        raise ValueError("Output sampling rate must be positive")
    # The resampling anti-aliasing filter suppresses frequencies above the
    # new Nyquist (50 Hz by default), regardless of the earlier 10-70-Hz EMG passband.
    if output_sfreq / 2 < 45:
        raise ValueError("Output Nyquist must exceed EEG gamma upper bound 45 Hz")


def preprocess_one(set_path: Path, output_path: Path, subject_id: str,
                   args: argparse.Namespace) -> str:
    if output_path.exists() and not args.overwrite:
        LOG.info("SKIP existing: %s", output_path)
        return "skipped"

    raw = mne.io.read_raw_eeglab(set_path, preload=True, verbose="ERROR")
    try:
        original_sfreq = float(raw.info["sfreq"])
        validate_filter_settings(original_sfreq, args.sampling_rate,
                                 args.notch_frequency)
        selected_raw, scheme = select_channels(raw, args.allow_mixed_aliases)
        # The original script picked the channels and renamed to F3, ..., Ch2.
        raw.pick(selected_raw)
        raw.rename_channels(dict(zip(selected_raw, STANDARD_CHANNELS)))
        raw.reorder_channels(STANDARD_CHANNELS)
        raw.set_channel_types(CHANNEL_TYPES, verbose="ERROR")

        # Same filter sequence as original script: band-pass -> notch -> resample.
        for modality, (lo, hi) in FILTER_BANDS.items():
            picks = [ch for ch in STANDARD_CHANNELS if CHANNEL_TYPES[ch] == modality]
            raw.filter(l_freq=lo, h_freq=hi, picks=picks, method="fir",
                       fir_design="firwin", phase="zero", verbose="ERROR")

        raw.notch_filter(freqs=[args.notch_frequency], picks=STANDARD_CHANNELS,
                         method="fir", fir_design="firwin", phase="zero",
                         verbose="ERROR")
        raw.resample(args.sampling_rate, npad="auto", verbose="ERROR")
        data = raw.get_data()
        if args.mne_data_unit == "V":
            data = data * 1e6  # Explicit V -> microvolts for the downstream features.
        if not np.isfinite(data).all():
            raise ValueError("Non-finite samples after filtering; refusing to write NPZ")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Write atomically to avoid retaining half-written outputs after a failure.
        tmp_path = output_path.with_name(output_path.stem + ".partial.npz")
        try:
            np.savez_compressed(
                tmp_path, data=data, sfreq=float(raw.info["sfreq"]),
                ch_names=np.asarray(STANDARD_CHANNELS), subject_id=subject_id,
                processing_time=datetime.now(timezone.utc).isoformat(),
                original_sfreq=original_sfreq, input_data_unit=args.mne_data_unit,
                units="uV", channel_scheme=scheme,
            )
            tmp_path.replace(output_path)
        finally:
            tmp_path.unlink(missing_ok=True)
        LOG.info("OK %-16s scheme=%s shape=%s -> %s", subject_id, scheme,
                 tuple(data.shape), output_path)
        return "ok"
    finally:
        raw.close()


def main(argv: list[str] | None = None) -> int:
    args = arguments(argv)
    configure_logging(args.verbose)
    summary = {"ok": 0, "skipped": 0, "failed": 0}
    total = 0
    for group in args.groups:
        input_dir = args.input_root / group
        if not input_dir.is_dir():
            LOG.warning("Missing group input directory: %s", input_dir)
            continue
        pattern = f"{group}_*_selected.set"
        paths = sorted(input_dir.glob(pattern))
        LOG.info("Group %s: found %s input files", group, len(paths))
        for set_path in paths:
            total += 1
            subject_id = set_path.stem.removesuffix("_selected")
            output_path = args.output_root / group / f"{subject_id}_processed.npz"
            if args.dry_run:
                LOG.info("DRY RUN %s -> %s", set_path, output_path)
                continue
            try:
                state = preprocess_one(set_path, output_path, subject_id, args)
                summary[state] += 1
            except Exception:
                summary["failed"] += 1
                LOG.exception("FAILED: %s", set_path)
    LOG.info("Done: files=%d, processed=%d, skipped=%d, failed=%d",
             total, summary["ok"], summary["skipped"], summary["failed"])
    if total == 0:
        LOG.error("No matching .set files found. Check input-root/groups/naming.")
        return 2
    return 1 if summary["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
