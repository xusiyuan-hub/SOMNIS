#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build locked SOMNIS discovery/independent NPZ datasets for BOTH feature modes.

This replaces prepare_stages_dataset.py and prepare_wholenight_dataset.py.
It NEVER randomly re-splits patients. Instead, it uses researcher-provided,
PRE-EXISTING text lists that document the original locked subject allocation.
An unlisted subject is excluded, as in the archived analysis.

Required split files in --label-root:
  earlyMSA_train.txt   earlyMSA_test.txt
  earlyPD_train.txt    earlyPD_test.txt
Each line contains one ID, either 001 or MSA_001 (same for PD). Do not
publish actual patient IDs without institutional permission.

Input NPZs from add_modality_labels.py:
  --input-root/stages/MSA_ch.npz and PD_ch.npz  (1390 columns)
  --input-root/wholenight/MSA_wostages_ch.npz and PD_wostages_ch.npz (278)
Outputs in --output-root:
  stages/earlyMSA.npz, stages/earlyPD.npz
  wholenight/earlyMSA_wostages.npz, wholenight/earlyPD_wostages.npz
Each contains X, feature_names, feature_types, channels, subject_ids,
and dataset with 'train'/'test' labels expected by the SOMNIS model.

Example:
  python features/prepare_dataset.py \\
    --input-root data/datasets/all --label-root data/labels \\
    --output-root data/datasets/early --groups MSA PD --mode both \\
    --expect-original-cohort
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

from add_modality_labels import strings, normalize_subject_id

LOG = logging.getLogger("somnis.features.prepare_dataset")
PUBLISHED_COUNTS = {"MSA": (77, 33), "PD": (64, 28)}


def read_id_file(path: Path, group: str) -> set[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing original split file: {path}")
    content = path.read_text(encoding="utf-8-sig").splitlines()
    ids = [normalize_subject_id(line.strip(), group) for line in content if line.strip()]
    if not ids:
        raise ValueError(f"Empty split file: {path}")
    if len(set(ids)) != len(ids):
        raise ValueError(f"Duplicate ID in split file: {path}")
    return set(ids)


def load_fixed_splits(label_root: Path, group: str) -> tuple[set[str], set[str]]:
    train = read_id_file(label_root / f"early{group}_train.txt", group)
    test = read_id_file(label_root / f"early{group}_test.txt", group)
    both = train & test
    if both:
        raise ValueError(f"LEAKAGE: {group} IDs appear in BOTH splits: {sorted(both)[:5]}")
    return train, test


def load_source(path: Path, group: str, mode: str) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as d:
        fields = ("X", "feature_names", "feature_types", "channels", "subject_ids")
        absent = set(fields) - set(d.files)
        if absent:
            raise ValueError(f"Missing fields {sorted(absent)} in {path}")
        X = np.asarray(d["X"], dtype=float)
        names = strings(d["feature_names"])
        types = strings(d["feature_types"])
        channels = np.char.upper(strings(d["channels"]))
        ids = np.asarray([normalize_subject_id(x, group) for x in strings(d["subject_ids"])])
    expected = 1390 if mode == "stages" else 278
    if X.ndim != 2 or X.shape[1] != expected:
        raise ValueError(f"{path}: expected (n,{expected}), got {X.shape}")
    if not (len(names) == len(types) == len(channels) == expected):
        raise ValueError(f"{path}: metadata length mismatch")
    if len(ids) != len(X) or len(set(ids)) != len(ids):
        raise ValueError(f"{path}: duplicated or mismatched subject IDs")
    if not set(np.unique(channels)).issubset({"EEG", "EOG", "EMG"}):
        raise ValueError(f"{path}: invalid modality labels")
    if np.any(np.isinf(X)):
        raise ValueError(f"{path}: infinite features cannot be imputed")
    return dict(X=X, feature_names=names, feature_types=types,
                channels=channels, subject_ids=ids)


def create_subset(source: dict, group: str, train_ids: set[str], test_ids: set[str],
                  require_all: bool) -> dict:
    observed = set(source["subject_ids"])
    requested = train_ids | test_ids
    missing = requested - observed
    if missing and require_all:
        raise ValueError(f"{group}: {len(missing)} IDs in lists are not in feature NPZ: "
                         f"{sorted(missing)[:10]}")
    if missing:
        LOG.warning("%s: %d split-list IDs absent from features: %s", group, len(missing),
                    sorted(missing)[:10])
    ids = source["subject_ids"]
    train = np.isin(ids, list(train_ids))
    test = np.isin(ids, list(test_ids))
    keep = train | test
    dataset = np.where(train[keep], "train", "test")
    kept_ids = ids[keep]
    if not len(kept_ids) or not np.any(dataset == "train") or not np.any(dataset == "test"):
        raise ValueError(f"{group}: both train and test subsets must have samples")
    return {"X": source["X"][keep],
            "subject_ids": kept_ids,
            "dataset": np.asarray(dataset, dtype=str),
            "feature_names": source["feature_names"],
            "feature_types": source["feature_types"],
            "channels": source["channels"]}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-root", type=Path, required=True)
    p.add_argument("--label-root", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--groups", nargs="+", default=["MSA", "PD"])
    p.add_argument("--mode", choices=["stages", "wholenight", "both"], default="both")
    p.add_argument("--expect-original-cohort", action="store_true",
                   help="Assert study's exact counts: MSA 77 train/33 test; PD 64/28")
    p.add_argument("--allow-missing-ids", action="store_true",
                   help="Permit IDs listed in train/test not present in feature matrix (not recommended)")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    groups = args.groups
    modes = ("stages", "wholenight") if args.mode == "both" else (args.mode,)
    if len(set(groups)) != len(groups):
        raise ValueError("Repeated group names")
    splits = {group: load_fixed_splits(args.label_root, group) for group in groups}
    if len(groups) > 1:
        for i, group in enumerate(groups):
            for other in groups[i+1:]:
                if (splits[group][0] | splits[group][1]) & (splits[other][0] | splits[other][1]):
                    raise ValueError("Subject IDs overlap between diagnostic groups")
    results: dict[tuple[str, str], dict] = {}
    destinations: dict[tuple[str, str], Path] = {}
    for mode in modes:
        for group in groups:
            name = f"{group}_ch.npz" if mode == "stages" else f"{group}_wostages_ch.npz"
            source = load_source(args.input_root / mode / name, group, mode)
            train_ids, test_ids = splits[group]
            data = create_subset(source, group, train_ids, test_ids,
                                 require_all=not args.allow_missing_ids)
            n_train = int((data["dataset"] == "train").sum())
            n_test = int((data["dataset"] == "test").sum())
            if args.expect_original_cohort and group in PUBLISHED_COUNTS:
                if (n_train, n_test) != PUBLISHED_COUNTS[group]:
                    raise ValueError(f"{mode}/{group}: expected {PUBLISHED_COUNTS[group]}, "
                                     f"got {(n_train, n_test)}")
            outfile = (f"early{group}.npz" if mode == "stages" else
                       f"early{group}_wostages.npz")
            dest = args.output_root / mode / outfile
            if dest.exists() and not args.overwrite:
                raise FileExistsError(f"Output exists: {dest}; use --overwrite")
            results[(mode, group)] = data
            destinations[(mode, group)] = dest
            LOG.info("%s/%s: train=%d test=%d, X=%s", mode, group,
                     n_train, n_test, data["X"].shape)
    # Block silent mismatch of column meanings even when dimensions are equal.
    for mode in modes:
        for group in groups[1:]:
            baseline = results[(mode, groups[0])]
            trial = results[(mode, group)]
            if not np.array_equal(baseline["feature_names"], trial["feature_names"]):
                raise ValueError(f"Feature NAME/order mismatch: {groups[0]} vs {group}, {mode}")
            if not np.array_equal(baseline["channels"], trial["channels"]):
                raise ValueError(f"Channel-modality mapping mismatch: {groups[0]} vs {group}, {mode}")
    # When doing both feature representations, the identical subjects MUST be used.
    if len(modes) == 2:
        for group in groups:
            stage = results[("stages", group)]
            night = results[("wholenight", group)]
            if (set(zip(stage["subject_ids"], stage["dataset"])) !=
                    set(zip(night["subject_ids"], night["dataset"]))):
                raise ValueError(f"{group}: stage and whole-night cohort allocations differ")
    for key, data in results.items():
        dest = destinations[key]
        dest.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(dest, **data)
        LOG.info("Wrote %s", dest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
