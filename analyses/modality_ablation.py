# -*- coding: utf-8 -*-
"""SOMNIS: leave-one-modality-out ablation (stage-resolved PSG features).

Run from the repository root:
  python analyses/modality_ablation.py --input-root data/datasets/early \
      --output-root results/ablation --expect-original-cohort

Conditions: Full, -EEG, -EOG, -EMG. Each condition is re-tuned and fit
independently with the fixed SOMNIS nested CV procedures, and the validation
cohort is never used for tuning or threshold selection.

For historical modality-ablation compatibility, EEG/EOG/EMG always use seed
indices 1/2/3, even when an earlier modality is removed. In the Full model
this gives exactly the same seed offsets as models/somnis_stacking.py.

Never publish generated subject-level prediction CSVs or the clinical NPZs.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from models import somnis_stacking as somnis

MODALITIES = ('EEG', 'EOG', 'EMG')
SEED_INDICES = {'EEG': 1, 'EOG': 2, 'EMG': 3}
MODALITY_CONDITIONS = ('Full', '-EEG', '-EOG', '-EMG')
METRICS = ('AUC', 'ACC', 'BAC', 'Sens', 'Spec', 'PPV', 'NPV')
DATASETS = ('Discovery set (outer-fold OOF)', 'Independent validation set')
REQUIRED_OUTPUTS = ('model_performance_with_bootstrap_CI.csv',
                    'predicted_probabilities_for_dca.csv',
                    'final_discovery_threshold_operating_points.csv')


def _digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def configure_model(seed: int, bootstrap: int, smoke_test: bool = False):
    """Set runtime globals on the UNMODIFIED fixed main-model functions."""
    somnis.RANDOM_SEED = seed
    somnis.BOOTSTRAP_SEED = seed
    somnis.N_BOOTSTRAP = bootstrap
    if smoke_test:
        # This is an explicitly nonpublication integration test only.
        somnis.N_OUTER_FOLDS = 2
        somnis.N_META_TUNE_FOLDS = 2
        somnis.N_BASE_TUNE_FOLDS = 2
        somnis.N_L0_OOF_FOLDS = 2
        somnis.XGB_PARAM_GRID = [{'n_estimators': 8, 'max_depth': 2,
                                  'learning_rate': 0.1}]
        somnis.SVM_PARAM_GRID = [{'C': 0.1, 'kernel': 'linear', 'class_weight': None}]
        print('*** SMOKE TEST ONLY. THESE RESULTS ARE NOT PUBLICATION-VALID. ***',
              flush=True)


def load_study(msa_file: Path, pd_file: Path, expect_original_cohort: bool):
    somnis.EXPECT_ORIGINAL_COHORT = expect_original_cohort
    info = somnis.validate_input_feature_schema(msa_file, pd_file, 'stages')
    somnis.MSA_FILE_PATH, somnis.PD_FILE_PATH = msa_file, pd_file
    data = somnis.load_data()
    data['channels'] = info['MSA']['modalities']
    data['feature_names'] = info['MSA']['features']
    return data


def _make_l1_with_stable_modality_seeds(X_fit, y_fit, X_holdout,
                                        feature_groups, group_names, seed,
                                        stage_label):
    """Keep original modality-ablation seed offsets, call fixed SOMNIS base routines."""
    y_fit = np.asarray(y_fit, dtype=int)
    l1_fit = np.zeros((len(y_fit), len(group_names)), dtype=float)
    l1_holdout = np.zeros((len(X_holdout), len(group_names)), dtype=float)
    selected_params = {}
    for j, (mask, name) in enumerate(zip(feature_groups, group_names)):
        if name not in SEED_INDICES:
            raise ValueError(f'Unexpected modality {name}')
        offset = SEED_INDICES[name]
        xf = X_fit[:, mask]
        xv = X_holdout[:, mask]
        l1_fit[:, j] = somnis.strict_oof_probabilities_for_one_modality(
            X_group=xf, y=y_fit, seed=seed + 100_000 * offset,
            stage_label=stage_label, modality_name=name)
        params, _ = somnis.tune_one_base_model(
            X=xf, y=y_fit, seed=seed + 200_000 * offset,
            stage_label=stage_label, modality_name=name)
        selected_params[name] = params
        l1_holdout[:, j] = somnis.fit_xgb_and_predict(xf, y_fit, xv, params)
    return l1_fit, l1_holdout, selected_params


def _tune_meta_with_stable_modality_seeds(X, y, feature_groups, group_names,
                                           seed, stage_label):
    """Same meta CV and AUC grid as frozen model, but preserve per-modality seed."""
    splits = somnis.make_stratified_splits(y, somnis.N_META_TUNE_FOLDS, seed,
                                          f'{stage_label} / strict meta SVM tuning')
    fold_aucs = [[] for _ in somnis.SVM_PARAM_GRID]
    candidate_oof = np.full((len(y), len(somnis.SVM_PARAM_GRID)), np.nan)
    for fold, (tr, va) in enumerate(splits, start=1):
        z_train, z_val, _ = _make_l1_with_stable_modality_seeds(
            X[tr], y[tr], X[va], feature_groups, group_names,
            seed=seed + 1_000_000 * fold,
            stage_label=f'{stage_label} | meta_fold_{fold}')
        for k, params in enumerate(somnis.SVM_PARAM_GRID):
            p = somnis.fit_svm_and_predict(z_train, y[tr], z_val, params)
            candidate_oof[va, k] = p
            fold_aucs[k].append(somnis.safe_auc(y[va], p))
    if not np.isfinite(candidate_oof).all():
        raise RuntimeError('Incomplete meta OOF probabilities')
    scores = [float(np.nanmean(a)) for a in fold_aucs]
    winner, best = None, -np.inf
    for index, score in enumerate(scores):
        if score > best + 1e-12:
            winner, best = index, score
    if winner is None:
        raise RuntimeError('No meta-SVM candidate could be selected')
    return somnis.SVM_PARAM_GRID[winner].copy(), scores[winner], candidate_oof[:, winner].copy()


def _prepare_condition(data: dict, analysis: str, condition: str):
    """Return filtered X and modality masks. Filters are based on NPZ metadata."""
    if analysis == 'modality':
        if condition not in MODALITY_CONDITIONS:
            raise ValueError(f'Unknown modality condition {condition}')
        channels = np.asarray(data['channels'])
        keep = np.ones(len(channels), dtype=bool) if condition == 'Full' else (
            channels != condition[1:])
        names = [m for m in MODALITIES if m != condition[1:]]
    elif analysis == 'sleep_stage':
        from analyses.sleep_stage_ablation import STAGE_CONDITIONS, get_stage_labels
        if condition not in STAGE_CONDITIONS:
            raise ValueError(f'Unknown sleep stage condition {condition}')
        stages = get_stage_labels(data['feature_names'])
        keep = np.ones(len(stages), dtype=bool) if condition == 'Full' else (
            stages != condition[1:])
        channels = np.asarray(data['channels'])
        names = list(MODALITIES)
    else:
        raise ValueError('analysis must be modality or sleep_stage')
    Xd, Xi = data['X_discovery'][:, keep], data['X_independent'][:, keep]
    filtered_channels = channels[keep]
    groups = [filtered_channels == m for m in names]
    if any(not np.any(g) for g in groups):
        raise RuntimeError(f'{condition}: empty retained modality')
    return Xd, Xi, groups, names, keep


def _metric_summary_row(dataset, result, ci, threshold, rule, condition, n_features,
                        n_modality_inputs):
    row = {'Condition': condition, 'Dataset': dataset,
           'N_features': int(n_features), 'N_modality_inputs': int(n_modality_inputs),
           'TN': result['TN'], 'FP': result['FP'], 'FN': result['FN'], 'TP': result['TP'],
           'Threshold': threshold, 'Threshold_selection_rule': rule}
    for key in METRICS:
        row[key] = result[key]
        row[f'{key}_CI_lower'] = ci[key][0]
        row[f'{key}_CI_upper'] = ci[key][1]
    return row


def run_condition(data: dict, analysis: str, condition: str, folder: Path,
                  bootstrap: int, seed: int, skip_condition_figures: bool = False):
    """Complete independent re-tuning of one ablation condition.

    The ONLY fitting/tuning functions used are the fixed main model functions,
    except the stable-modality-seed wrapper used by historical modality ablation.
    """
    folder.mkdir(parents=True, exist_ok=True)
    Xd, Xi, groups, names, feature_keep = _prepare_condition(data, analysis, condition)
    yd, yi = data['y_discovery'], data['y_independent']
    use_stable = analysis == 'modality'
    tune = (_tune_meta_with_stable_modality_seeds if use_stable
            else somnis.tune_meta_svm_strict)
    build = (_make_l1_with_stable_modality_seeds if use_stable
             else somnis.build_strict_l1_features_and_holdout)
    print(f'[{analysis}/{condition}] {Xd.shape[1]} features; '
          f'{len(groups)} Level-1 modalities; Discovery n={len(yd)}; '
          f'Independent n={len(yi)}', flush=True)
    pd.DataFrame({'Modality': names,
                  'N_features': [int(np.sum(g)) for g in groups]}).to_csv(
        folder / 'features_by_retained_modality.csv', index=False)

    outer_splits = somnis.make_stratified_splits(
        yd, somnis.N_OUTER_FOLDS, seed, f'{condition} outer CV')
    outer_prob = np.full(len(yd), np.nan)
    outer_labels = np.full(len(yd), -1, dtype=int)
    outer_id = np.full(len(yd), -1, dtype=int)
    outer_thresholds = np.full(len(yd), np.nan)

    for fold, (tr, va) in enumerate(outer_splits, start=1):
        # All meta tuning and cut-off selection are restricted to outer training.
        best_svm, _, oof_train = tune(
            X=Xd[tr], y=yd[tr], feature_groups=groups, group_names=names,
            seed=seed + 10_000_000 * fold,
            stage_label=f'outer_fold_{fold}')
        threshold, _ = somnis.select_threshold_from_training_oof(yd[tr], oof_train)
        l1_tr, l1_va, _ = build(
            X_fit=Xd[tr], y_fit=yd[tr], X_holdout=Xd[va],
            feature_groups=groups, group_names=names,
            seed=seed + 20_000_000 * fold,
            stage_label=f'outer_fold_{fold}_final_fit')
        p = somnis.fit_svm_and_predict(l1_tr, yd[tr], l1_va, best_svm)
        outer_prob[va] = p
        outer_labels[va] = (p >= threshold).astype(int)
        outer_id[va] = fold
        outer_thresholds[va] = threshold
        print(f'  outer fold {fold}/{somnis.N_OUTER_FOLDS}: '
              f'AUC={somnis.safe_auc(yd[va], p):.4f}, threshold={threshold:.4f}',
              flush=True)
        del l1_tr, l1_va, oof_train
        gc.collect()

    if not np.isfinite(outer_prob).all() or (outer_labels < 0).any():
        raise RuntimeError('Incomplete Discovery outer-fold predictions')
    discovery_metrics = somnis.calculate_metrics(yd, outer_prob,
                                                 predicted_labels=outer_labels)

    final_svm, final_cv_auc, final_oof = tune(
        X=Xd, y=yd, feature_groups=groups, group_names=names,
        seed=seed + 900_000_000, stage_label='final_discovery_training')
    final_threshold, final_rule = somnis.select_threshold_from_training_oof(yd, final_oof)
    threshold_audit = somnis._roc_threshold_table(yd, final_oof)
    threshold_audit['Selected'] = np.isclose(threshold_audit['Threshold'],
                                             final_threshold)
    threshold_audit['SelectedRule'] = np.where(threshold_audit['Selected'], final_rule, '')
    threshold_audit.to_csv(folder / 'final_discovery_threshold_operating_points.csv',
                           index=False, encoding='utf-8-sig')

    l1_tr, l1_test, base_params = build(
        X_fit=Xd, y_fit=yd, X_holdout=Xi, feature_groups=groups, group_names=names,
        seed=seed + 950_000_000, stage_label='final_discovery_fit')
    p_test = somnis.fit_svm_and_predict(l1_tr, yd, l1_test, final_svm)
    independent_metrics = somnis.calculate_metrics(yi, p_test,
                                                   threshold=final_threshold)

    discovery_ci = somnis.bootstrap_confidence_intervals(
        yd, outer_prob, outer_labels, n_bootstrap=bootstrap, random_seed=seed)
    independent_ci = somnis.bootstrap_confidence_intervals(
        yi, p_test, independent_metrics['predicted_labels'],
        n_bootstrap=bootstrap, random_seed=seed + 1)

    rows = [
        _metric_summary_row(DATASETS[0], discovery_metrics, discovery_ci,
                            'fold-specific', 'selected separately from each outer-training partition',
                            condition, Xd.shape[1], len(names)),
        _metric_summary_row(DATASETS[1], independent_metrics, independent_ci,
                            final_threshold, final_rule, condition, Xd.shape[1], len(names))]
    summary = pd.DataFrame(rows)
    summary.to_csv(folder / 'model_performance_with_bootstrap_CI.csv',
                   index=False, encoding='utf-8-sig')
    pred = pd.concat([
        pd.DataFrame({'Dataset': DATASETS[0],
                      'SubjectID': somnis.normalize_subject_ids(data['discovery_ids']),
                      'TrueLabel': yd, 'MSA_probability': outer_prob,
                      'PredictedLabel': outer_labels,
                      'AppliedThreshold': outer_thresholds, 'OuterFold': outer_id}),
        pd.DataFrame({'Dataset': DATASETS[1],
                      'SubjectID': somnis.normalize_subject_ids(data['independent_ids']),
                      'TrueLabel': yi, 'MSA_probability': p_test,
                      'PredictedLabel': independent_metrics['predicted_labels'],
                      'AppliedThreshold': final_threshold, 'OuterFold': np.nan})],
        ignore_index=True)
    pred.to_csv(folder / 'predicted_probabilities_for_dca.csv',
                index=False, encoding='utf-8-sig')

    if not skip_condition_figures:
        somnis.OUTPUT_DIR = folder
        somnis.save_clinical_performance_figure(
            y_discovery=yd, discovery_oof_probability=outer_prob,
            discovery_oof_label=outer_labels, y_independent=yi,
            independent_probability=p_test,
            independent_label=independent_metrics['predicted_labels'],
            discovery_metrics=discovery_metrics,
            independent_metrics=independent_metrics,
            discovery_ci=discovery_ci, independent_ci=independent_ci,
            independent_threshold=final_threshold)
    metadata = {'analysis': analysis, 'condition': condition, 'model': 'SOMNIS',
                'model_file': 'models/somnis_stacking.py', 'seed': seed,
                'bootstrap': bootstrap, 'feature_count': int(Xd.shape[1]),
                'retained_modalities': names,
                'final_svm_params': final_svm, 'final_base_params': base_params,
                'final_meta_cv_auc': float(final_cv_auc),
                'final_threshold': float(final_threshold),
                'threshold_rule': final_rule,
                'warning': 'Participant-level predictions are private; do not commit.',
                'all_conditions_refitted': True}
    (folder / 'condition_metadata.json').write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'Saved {analysis}/{condition}: {folder}', flush=True)
    return summary


def condition_slug(condition):
    return 'full' if condition == 'Full' else 'minus_' + condition[1:]


def _run_signature(args, analysis, condition, msa_file, pd_file) -> str:
    """Use actual input bytes and source/parameter identity; reject stale caches."""
    record = {'analysis': analysis, 'condition': condition,
              'msa_sha256': _digest(msa_file), 'pd_sha256': _digest(pd_file),
              'main_sha256': _digest(ROOT / 'models' / 'somnis_stacking.py'),
              'driver_sha256': _digest(Path(__file__).resolve()),
              'stage_driver_sha256': (_digest(ROOT / 'analyses' / 'sleep_stage_ablation.py')
                                      if analysis == 'sleep_stage' else None),
              'seed': args.seed, 'bootstrap': args.bootstrap,
              'smoke_test': args.smoke_test,
              'skip_condition_figures': args.skip_condition_figures}
    return hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()


def aggregate_summary(base_dir: Path, conditions, kind: str) -> pd.DataFrame:
    rows = []
    for condition in conditions:
        p = base_dir / condition_slug(condition) / 'model_performance_with_bootstrap_CI.csv'
        if not p.is_file():
            continue
        df = pd.read_csv(p)
        if len(df) != 2 or set(df['Dataset']) != set(DATASETS):
            raise RuntimeError(f'Incomplete condition metrics: {p}')
        if 'Condition' in df.columns:
            if not (df['Condition'] == condition).all():
                raise ValueError(f'Incorrect Condition in {p}')
        else:
            df.insert(0, 'Condition', condition)
        rows.append(df)
    if not rows:
        raise RuntimeError(f'No completed {kind} conditions to aggregate')
    summary = pd.concat(rows, ignore_index=True)
    csv_name = ('modality_ablation_summary.csv' if kind == 'modality' else
                'sleep_stage_ablation_summary.csv')
    summary.to_csv(base_dir / csv_name, index=False, encoding='utf-8-sig')
    full = summary.loc[summary.Condition == 'Full']
    if len(full) == 2:
        drop_rows = []
        for _, row in summary.iterrows():
            baseline = full.loc[full.Dataset == row.Dataset]
            if baseline.empty:
                continue
            for metric in METRICS:
                drop_rows.append({'Condition': row.Condition, 'Dataset': row.Dataset,
                                  'Metric': metric, 'Full_value': float(baseline.iloc[0][metric]),
                                  'Ablation_value': float(row[metric]),
                                  'Performance_drop_vs_full': float(baseline.iloc[0][metric] - row[metric])})
        drop_name = ('modality_ablation_performance_drop_vs_full.csv' if kind == 'modality'
                     else 'sleep_stage_ablation_performance_drop_vs_full.csv')
        pd.DataFrame(drop_rows).to_csv(base_dir / drop_name, index=False,
                                      encoding='utf-8-sig')
    return summary


def analysis_cli(analysis: str, choices: tuple[str, ...], argv=None):
    """Orchestrate separate child processes and safely resume matching runs."""
    parser = argparse.ArgumentParser(description=(
        'SOMNIS ' + ('leave-one-modality-out' if analysis == 'modality'
                    else 'leave-one-sleep-stage-out') + ' ablation'))
    parser.add_argument('--input-root', type=Path, default=Path('data/datasets/early'))
    parser.add_argument('--msa-file', type=Path)
    parser.add_argument('--pd-file', type=Path)
    parser.add_argument('--output-root', type=Path, default=Path('results/ablation'))
    parser.add_argument('--conditions', default=','.join(choices),
                        help='Comma-separated subset, e.g. Full,-EEG; cumulative summary is regenerated')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--bootstrap', type=int, default=2000)
    parser.add_argument('--expect-original-cohort', action='store_true')
    parser.add_argument('--force', action='store_true', help='Re-run even if cached success matches')
    parser.add_argument('--check-only', action='store_true', help='Validate input and conditions; no training')
    parser.add_argument('--skip-condition-figures', action='store_true',
                        help='Skip per-condition four-panel PNG/PDF')
    parser.add_argument('--smoke-test', action='store_true',
                        help='NONPUBLICATION: 2-fold miniature CV and tiny grid for integration testing')
    parser.add_argument('--_child-condition', choices=choices, default=None,
                        help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    args.conditions = [t.strip() for t in args.conditions.split(',') if t.strip()]
    if not args.conditions or any(t not in choices for t in args.conditions):
        parser.error(f'--conditions must be comma-separated choices from: {choices}')
    if len(set(args.conditions)) != len(args.conditions):
        parser.error('--conditions contains duplicates')
    if (args.msa_file is None) != (args.pd_file is None):
        parser.error('--msa-file and --pd-file must both be provided')
    if args.bootstrap < 1:
        parser.error('--bootstrap must be positive')
    msa_file = args.msa_file or args.input_root / 'stages' / 'earlyMSA.npz'
    pd_file = args.pd_file or args.input_root / 'stages' / 'earlyPD.npz'
    configure_model(args.seed, args.bootstrap, args.smoke_test)
    data = load_study(msa_file, pd_file, args.expect_original_cohort)
    # All conditions checked before expensive training begins.
    for cond in (args.conditions if args._child_condition is None else [args._child_condition]):
        xd, xi, group_masks, names, keep = _prepare_condition(data, analysis, cond)
        print(f'OK {analysis}/{cond}: {xd.shape[1]} features, {len(names)} base learners',
              flush=True)
    if args.check_only:
        return 0
    root = args.output_root / ('modality' if analysis == 'modality' else 'sleep_stage')
    root.mkdir(parents=True, exist_ok=True)
    scripts = {'modality': ROOT / 'analyses' / 'modality_ablation.py',
               'sleep_stage': ROOT / 'analyses' / 'sleep_stage_ablation.py'}
    conditions = ([args._child_condition] if args._child_condition is not None
                  else args.conditions)
    for cond in conditions:
        folder = root / condition_slug(cond)
        folder.mkdir(parents=True, exist_ok=True)
        signature = _run_signature(args, analysis, cond, msa_file, pd_file)
        success = folder / '_SUCCESS.json'
        complete = all((folder / file).is_file() for file in REQUIRED_OUTPUTS)
        cached = False
        if complete and success.is_file() and not args.force:
            try:
                meta = json.loads(success.read_text(encoding='utf-8'))
                cached = meta.get('signature') == signature
            except (ValueError, OSError):
                cached = False
        if cached:
            print(f'SKIP verified cached {analysis}/{cond}', flush=True)
            continue
        if args._child_condition is None:
            cmd = [sys.executable, str(scripts[analysis]),
                   '--input-root', str(args.input_root),
                   '--output-root', str(args.output_root),
                   '--seed', str(args.seed), '--bootstrap', str(args.bootstrap),
                   f'--_child-condition={cond}']
            if args.msa_file:
                cmd += ['--msa-file', str(args.msa_file), '--pd-file', str(args.pd_file)]
            if args.expect_original_cohort:
                cmd += ['--expect-original-cohort']
            if args.force:
                cmd += ['--force']
            if args.smoke_test:
                cmd += ['--smoke-test']
            if args.skip_condition_figures:
                cmd += ['--skip-condition-figures']
            print('Starting child process for', cond, flush=True)
            subprocess.run(cmd, cwd=str(ROOT), check=True)
        else:
            if success.is_file():
                success.unlink()  # stale markers never imply success
            run_condition(data, analysis, cond, folder, args.bootstrap, args.seed,
                          args.skip_condition_figures)
            success.write_text(json.dumps({'signature': signature, 'condition': cond,
                                           'analysis': analysis}, indent=2),
                               encoding='utf-8')
    if args._child_condition is None:
        # Aggregate all successfully completed conditions from this analysis,
        # but never silently include stale/mismatched cached output.
        valid = []
        for cond in choices:
            folder = root / condition_slug(cond)
            success = folder / '_SUCCESS.json'
            if not success.is_file() or not all((folder / f).is_file() for f in REQUIRED_OUTPUTS):
                continue
            expected = _run_signature(args, analysis, cond, msa_file, pd_file)
            try:
                if json.loads(success.read_text(encoding='utf-8')).get('signature') == expected:
                    valid.append(cond)
            except (ValueError, OSError):
                pass
        if valid:
            summary = aggregate_summary(root, valid, analysis)
            print(f'Updated summary with {len(valid)}/{len(choices)} conditions: '
                  f'{root} ({len(summary)} dataset rows)', flush=True)
            if len(valid) != len(choices):
                print('NOTE: Partial summary. Finish all conditions before plotting '
                      'complete-paper ablation figures.', flush=True)
    return 0


def cli(argv=None):
    return analysis_cli('modality', MODALITY_CONDITIONS, argv)


if __name__ == '__main__':
    raise SystemExit(cli())
