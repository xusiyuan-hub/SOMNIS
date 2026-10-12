#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SOMNIS stage-resolved PSG feature extraction (MSA and PD).

Input:   <preprocess-root>/<GROUP>/<GROUP>_<ID>_processed.npz
         <staging-root>/<GROUP>/<GROUP>_<ID>_selected_gssc.csv
Output:  <output-root>/epoch/<GROUP>/<GROUP>_<ID>_features.npz
         <output-root>/stages/<GROUP>.npz

The original per-epoch feature definitions and ordering are retained:
6 EEG * 33 + 2 EOG * 24 + 2 EMG * 16 = 278 features/epoch.
Each feature is then averaged *within* each of W,N1,N2,N3,R (5*278=1390).
No per-subject imputation or scaling occurs here.

Run from the project root:
  python features/extract_stage_features.py \\
    --preprocess-root data/preprocess \\
    --staging-root data/sleepstage/GSSC \\
    --output-root data/features --groups MSA PD

Reproducibility: --invalid-policy legacy-zero reproduces the original rule of
using zeros for failed per-channel feature calculations. The recommended
--invalid-policy nan (default) instead preserves failures as missing values;
this may change numerical values relative to already-published results.
"""
from __future__ import annotations

import argparse
import logging
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.stats import iqr, kurtosis, skew

LOG = logging.getLogger("somnis.features")
STAGES = ("W", "N1", "N2", "N3", "R")
EEG_CHANNELS = ("F3", "F4", "C3", "C4", "O1", "O2")
EOG_CHANNELS = ("L", "R")
EMG_CHANNELS = ("Ch1", "Ch2")
ALL_CHANNELS = EEG_CHANNELS + EOG_CHANNELS + EMG_CHANNELS
CHANNEL_MODALITY = {ch: "EEG" for ch in EEG_CHANNELS}
CHANNEL_MODALITY.update({ch: "EOG" for ch in EOG_CHANNELS})
CHANNEL_MODALITY.update({ch: "EMG" for ch in EMG_CHANNELS})
TIME_FEATURES = ("std", "iqr", "skew", "kurt", "nzc", "hmob", "hcomp")
FREQ_FEATURES = ("delta", "theta", "alpha", "sigma", "beta", "gamma",
                 "r_delta", "r_theta", "r_alpha", "r_sigma", "r_beta", "r_gamma",
                 "dt", "ds", "db", "dg", "at", "abspow")
NONLINEAR_FEATURES = ("perm_entropy", "svd_entropy", "sample_entropy", "dfa",
                      "petrosian", "katz", "higuchi", "lziv")
EEG_FEATURES = TIME_FEATURES + FREQ_FEATURES + NONLINEAR_FEATURES
EOG_FEATURES = TIME_FEATURES + ("delta", "theta", "alpha", "sigma",
                "r_delta", "r_theta", "r_alpha", "r_sigma", "abspow") + NONLINEAR_FEATURES
EMG_FEATURES = TIME_FEATURES + ("abspow",) + NONLINEAR_FEATURES
FEATURES_BY_TYPE = {"EEG": EEG_FEATURES, "EOG": EOG_FEATURES, "EMG": EMG_FEATURES}
EPOCH_FEATURE_NAMES = tuple(f"{ch}_{ft}" for ch in ALL_CHANNELS
                            for ft in FEATURES_BY_TYPE[CHANNEL_MODALITY[ch]])
STAGE_FEATURE_NAMES = tuple(f"{stage}_{name}" for stage in STAGES
                            for name in EPOCH_FEATURE_NAMES)
assert len(EPOCH_FEATURE_NAMES) == 278 and len(STAGE_FEATURE_NAMES) == 1390


def feature_type(name: str) -> str:
    # Handles both F3_sample_entropy and W_F3_sample_entropy without destroying
    # the underscore in sample_entropy, perm_entropy, or svd_entropy.
    parts = name.split("_", 2)
    if parts[0] in STAGES and len(parts) == 3 and parts[1] in ALL_CHANNELS:
        base = parts[2]
    else:
        base = name.split("_", 1)[1]
    if base in TIME_FEATURES:
        return "Time"
    if base in FREQ_FEATURES:
        return "Frequency"
    if base in NONLINEAR_FEATURES:
        return "Nonlinear"
    raise ValueError(f"Unknown feature type for {name!r}")


def bandpowers(psd: np.ndarray, freqs: np.ndarray, bands: list[tuple]) -> dict:
    """Original inclusive band edges and trapezoidal PSD integration."""
    delta_f = float(freqs[1] - freqs[0]) if len(freqs) > 1 else 1.0
    return {name: float(np.trapezoid(psd[(freqs >= low) & (freqs <= high)], dx=delta_f))
            for low, high, name in bands}


def epoch_channel_features(signal: np.ndarray, sfreq: float, modality: str) -> dict:
    """Original time, Welch spectral, and AntroPy calculations, without filtering."""
    import antropy as ant  # import lazily: --help / aggregation do not require AntroPy

    f: dict[str, float] = {}
    f["std"] = float(np.std(signal, ddof=1))
    f["iqr"] = float(iqr(signal, rng=(25, 75)))
    f["skew"] = float(skew(signal))
    f["kurt"] = float(kurtosis(signal))
    f["nzc"] = float(ant.num_zerocross(signal))
    f["hmob"], f["hcomp"] = [float(x) for x in ant.hjorth_params(signal)]
    # scipy.signal.welch defaults match the archived code.
    freqs, psd = welch(signal, sfreq, nperseg=max(int(4 * sfreq), 128))
    step = float(freqs[1] - freqs[0]) if len(freqs) > 1 else 1.0
    if modality in ("EEG", "EOG"):
        bands = [(0.5, 4, "delta"), (4, 8, "theta"), (8, 12, "alpha"),
                 (12, 16 if modality == "EEG" else 15, "sigma")]
        if modality == "EEG":
            bands += [(16, 30, "beta"), (30, 45, "gamma")]
        f.update(bandpowers(psd, freqs, bands))
        broad_hi = 45 if modality == "EEG" else 15
        mask = (freqs >= 0.5) & (freqs <= min(broad_hi, sfreq / 2 - 0.1))
        f["abspow"] = float(np.trapezoid(psd[mask], dx=step))
        relative = ["delta", "theta", "alpha", "sigma"]
        if modality == "EEG":
            relative += ["beta", "gamma"]
        for name in relative:
            f["r_" + name] = f[name] / f["abspow"] if f["abspow"] > 0 else 0.0
        if modality == "EEG":
            for ratio, num, den in [("dt", "delta", "theta"),
                                    ("ds", "delta", "sigma"),
                                    ("db", "delta", "beta"),
                                    ("dg", "delta", "gamma"),
                                    ("at", "alpha", "theta")]:
                f[ratio] = f[num] / f[den] if f[den] > 0 else 0.0
    else:
        # With sfreq=100 Hz, frequencies >= Nyquist (50 Hz) are not recoverable.
        mask = (freqs >= 10) & (freqs <= min(70, sfreq / 2 - 0.1))
        f["abspow"] = float(np.trapezoid(psd[mask], dx=step))
    f["perm_entropy"] = float(ant.perm_entropy(signal, normalize=True))
    f["svd_entropy"] = float(ant.svd_entropy(signal, normalize=True))
    f["sample_entropy"] = float(ant.sample_entropy(signal))
    f["dfa"] = float(ant.detrended_fluctuation(signal))
    f["petrosian"] = float(ant.petrosian_fd(signal))
    f["katz"] = float(ant.katz_fd(signal))
    f["higuchi"] = float(ant.higuchi_fd(signal))
    f["lziv"] = float(ant.lziv_complexity(signal > np.mean(signal), normalize=True))
    return f


def check_epoch_archive(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as d:
        values = np.asarray(d["features"], dtype=float)
        names = np.asarray(d["feature_names"]).astype(str)
        if "epoch_seconds" in d.files and float(d["epoch_seconds"]) != 30:
            raise ValueError(f"{path} is not a 30-second epoch archive")
    if values.ndim != 2 or values.shape[1] != 278 or len(values) < 1:
        raise ValueError(f"Invalid epoch feature matrix {path}: {values.shape}")
    if not np.array_equal(names, EPOCH_FEATURE_NAMES):
        raise ValueError(f"Wrong feature names/order in {path}")
    return values, names


def extract_epoch_archive(src: Path, dst: Path, invalid_policy: str = "nan", overwrite: bool = False) -> None:
    """Extract one participant's 30-s features; preserve every epoch row."""
    if dst.exists() and not overwrite:
        check_epoch_archive(dst)
        with np.load(dst, allow_pickle=False) as cached:
            required = {"invalid_policy", "source_size", "source_mtime_ns"}
            if not required.issubset(cached.files):
                raise ValueError(f"Epoch cache {dst} lacks provenance; use --overwrite-epochs")
            if (str(cached["invalid_policy"]) != invalid_policy or
                int(cached["source_size"]) != src.stat().st_size or
                int(cached["source_mtime_ns"]) != src.stat().st_mtime_ns):
                raise ValueError(f"Epoch cache {dst} was built from different settings/input; "
                                 "pass --overwrite-epochs")
        return
    with np.load(src, allow_pickle=False) as d:
        data = np.asarray(d["data"], dtype=float)
        sfreq = float(d["sfreq"])
        names = [str(x) for x in d["ch_names"]]
    if data.ndim != 2 or len(names) != data.shape[0]:
        raise ValueError(f"Bad channel/data dimensions in {src}")
    if sfreq <= 0:
        raise ValueError(f"Invalid sampling rate {sfreq} in {src}")
    if len(set(names)) != len(names) or any(ch not in names for ch in ALL_CHANNELS):
        raise ValueError(f"Missing or duplicate PSG channels in {src}")
    ordered = data[[names.index(ch) for ch in ALL_CHANNELS]]
    samples_per_epoch = int(30 * sfreq)
    count = ordered.shape[1] // samples_per_epoch
    if count == 0:
        raise ValueError(f"No complete 30-second epoch in {src}")
    if ordered.shape[1] % samples_per_epoch:
        LOG.warning("%s: dropped %d trailing samples", src.name,
                    ordered.shape[1] % samples_per_epoch)
    X = np.full((count, 278), np.nan, dtype=float)
    invalid_channels = 0
    for ep in range(count):
        start = ep * samples_per_epoch
        offset = 0
        for ch_index, channel in enumerate(ALL_CHANNELS):
            modality = CHANNEL_MODALITY[channel]
            features = FEATURES_BY_TYPE[modality]
            window = ordered[ch_index, start:start + samples_per_epoch]
            if window.size != samples_per_epoch:
                raise AssertionError("Window extraction misalignment")
            if not np.isfinite(window).all() or np.all(window == 0):
                invalid_channels += 1
                offset += len(features)
                continue
            try:
                values = epoch_channel_features(window, sfreq, modality)
                v = np.array([values[name] for name in features], dtype=float)
                v[~np.isfinite(v)] = np.nan
                X[ep, offset:offset + len(features)] = v
            except Exception as exc:
                invalid_channels += 1
                if invalid_policy == "legacy-zero":
                    X[ep, offset:offset + len(features)] = 0.0
                LOG.warning("%s epoch=%d channel=%s: %s", src.name, ep, channel, exc)
            offset += len(features)
    # In the archived script, a feature column that *never* appeared was added
    # as a column of zeros; retain this edge case under legacy-zero.
    if invalid_policy == "legacy-zero":
        empty_cols = np.isnan(X).all(axis=0)
        X[:, empty_cols] = 0.0
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.stem + ".partial.npz")
    try:
        np.savez_compressed(tmp, features=X, feature_names=np.asarray(EPOCH_FEATURE_NAMES),
                            sfreq=sfreq, epoch_seconds=30, n_epochs=count,
                            invalid_channel_epochs=invalid_channels,
                            invalid_policy=invalid_policy, source_size=src.stat().st_size,
                            source_mtime_ns=src.stat().st_mtime_ns)
        tmp.replace(dst)
    finally:
        tmp.unlink(missing_ok=True)
    LOG.info("Extracted %s: epochs=%d, shape=%s, invalid channel epochs=%d",
             src.name, count, X.shape, invalid_channels)


def average_all_epochs(features: np.ndarray) -> np.ndarray:
    """Mean per feature over ALL epochs, regardless of stage; output (278,)."""
    if features.ndim != 2 or features.shape[1] != 278 or len(features) == 0:
        raise ValueError("Expected non-empty (epochs,278) matrix")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = np.nanmean(features, axis=0)
    return result


def read_stages(stage_file: Path, n_epochs: int) -> np.ndarray:
    """Fail instead of silently clipping a misaligned hypnogram."""
    df = pd.read_csv(stage_file)
    if "Stage" not in df.columns:
        raise ValueError(f"No Stage column in {stage_file}")
    if len(df) != n_epochs:
        raise ValueError(f"{stage_file}: GSSC epochs={len(df)} but feature epochs={n_epochs}")
    if "Epoch_30s_Idx" in df.columns:
        idx = pd.to_numeric(df["Epoch_30s_Idx"], errors="raise").to_numpy()
        if not np.array_equal(idx, np.arange(n_epochs)):
            raise ValueError(f"Nonconsecutive epoch indices in {stage_file}")
    if "Time_Sec" in df.columns and n_epochs > 1:
        ts = pd.to_numeric(df["Time_Sec"], errors="raise").to_numpy(dtype=float)
        if not np.isfinite(ts).all() or not np.allclose(np.diff(ts), 30.0, atol=0.2):
            raise ValueError(f"GSSC stage times not separated by 30 seconds: {stage_file}")
    # -1 for unrecognized / undefined; never convert another label into W.
    return df["Stage"].astype(str).str.strip().str.upper().to_numpy()


def average_by_stage(features: np.ndarray, stage_labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(stage_labels) != len(features):
        raise ValueError("GSSC labels and feature epochs do not match")
    out = np.full((5, 278), np.nan, dtype=float)
    counts = np.zeros(5, dtype=int)
    for i, stage in enumerate(STAGES):
        selected = features[stage_labels == stage]
        counts[i] = len(selected)
        if counts[i]:
            out[i] = average_all_epochs(selected)
    return out, counts


def list_processed_files(preprocess_root: Path, group: str) -> list[tuple[str, Path]]:
    if not (preprocess_root / group).is_dir():
        raise FileNotFoundError(preprocess_root / group)
    jobs = []
    pattern = re.compile(rf"^{re.escape(group)}_(.+)_processed\.npz$", re.I)
    for p in sorted((preprocess_root / group).glob("*_processed.npz")):
        match = pattern.match(p.name)
        if match:
            jobs.append((f"{group}_{match.group(1)}", p))
    if not jobs:
        raise FileNotFoundError(f"No processed NPZ files found for {group}")
    return jobs


def run_stage_group(preprocess_root: Path, staging_root: Path, output_root: Path,
                    group: str, overwrite_epochs: bool, invalid_policy: str,
                    fail_fast: bool = False) -> tuple[int, int]:
    X_all: list[np.ndarray] = []
    IDs: list[str] = []
    failed = 0
    for subject_id, src in list_processed_files(preprocess_root, group):
        epoch_path = output_root / "epoch" / group / f"{subject_id}_features.npz"
        hypno_path = staging_root / group / f"{subject_id}_selected_gssc.csv"
        try:
            if not hypno_path.is_file():
                raise FileNotFoundError(hypno_path)
            extract_epoch_archive(src, epoch_path, invalid_policy, overwrite_epochs)
            features, _ = check_epoch_archive(epoch_path)
            labels = read_stages(hypno_path, len(features))
            stage_values, counts = average_by_stage(features, labels)
            X_all.append(stage_values.reshape(-1))
            IDs.append(subject_id)
            LOG.info("Stage average %-20s counts=%s", subject_id, counts.tolist())
        except Exception:
            failed += 1
            LOG.exception("Failed stage aggregation for %s", subject_id)
            if fail_fast:
                raise
    if not IDs:
        raise RuntimeError(f"No successful participants for group {group}")
    out = output_root / "stages" / f"{group}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, X=np.vstack(X_all), feature_names=np.asarray(STAGE_FEATURE_NAMES),
                        feature_types=np.asarray([feature_type(name) for name in STAGE_FEATURE_NAMES]),
                        subject_ids=np.asarray(IDs), representation="stage-resolved")
    LOG.info("Wrote %s: X=%s; failed=%d", out, (len(IDs), 1390), failed)
    return len(IDs), failed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--preprocess-root", type=Path, required=True)
    p.add_argument("--staging-root", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--groups", nargs="+", default=["MSA", "PD"])
    p.add_argument("--invalid-policy", choices=("nan", "legacy-zero"), default="nan",
                   help="When a channel feature computation fails: NaN (recommended) or 0 (archived)")
    p.add_argument("--overwrite-epochs", action="store_true",
                   help="Recompute per-epoch archives instead of using existing cached files")
    p.add_argument("--fail-fast", action="store_true")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")
    failed = 0
    for group in args.groups:
        _, errors = run_stage_group(args.preprocess_root, args.staging_root,
                                    args.output_root, group, args.overwrite_epochs,
                                    args.invalid_policy, args.fail_fast)
        failed += errors
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
