# -*- coding: utf-8 -*-
"""SOMNIS leave-one-sleep-stage-out ablation with locked original partitions.

Uses exactly the frozen models/somnis_stacking.py routines. Every condition
refits and retunes the reduced data independently; no stages are zeroed out.

    python analyses/sleep_stage_ablation.py --input-root data/datasets/early \
        --output-root results/ablation --expect-original-cohort

The five 30-second sleep-stage blocks are named W, N1, N2, N3, and R.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from analyses.modality_ablation import analysis_cli

STAGES = ('W', 'N1', 'N2', 'N3', 'R')
STAGE_CONDITIONS = ('Full', '-W', '-N1', '-N2', '-N3', '-R')


def get_stage_labels(feature_names):
    """Parse stage from NAME PREFIX (before first underscore), never from position."""
    names = np.asarray(feature_names, dtype=str)
    labels = np.asarray([name.split('_', 1)[0] for name in names], dtype=str)
    bad = np.flatnonzero(~np.isin(labels, STAGES))
    if len(bad):
        raise ValueError('Feature names must start with W_, N1_, N2_, N3_, or R_; '
                         f'first invalid: {names[int(bad[0])]!r}')
    counts = {stage: int(np.sum(labels == stage)) for stage in STAGES}
    if counts != {s: 278 for s in STAGES}:
        raise ValueError(f'Expected exactly 278 features per stage, got {counts}. '
                         'Do not infer stages from row/column position.')
    return labels


def cli(argv=None):
    return analysis_cli('sleep_stage', STAGE_CONDITIONS, argv)


if __name__ == '__main__':
    raise SystemExit(cli())
