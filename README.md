# SOMNIS

**Research code for sleep-polysomnography-based differentiation of early multiple system atrophy (MSA) and Parkinson's disease (PD).**

This repository contains the preprocessing, automated sleep-stage scoring, feature engineering, model development, ablation studies, model interpretation, and visualization code used in the SOMNIS research workflow.

> **Research software, not a clinical diagnostic device.** This repository does not provide patient PSG recordings, private subject identifiers, or the original participant split lists. The provided scripts require appropriately authorized study data. The complete pipeline has not yet been independently validated as a turnkey reproduction of the published numerical results.

## Study design and modeling overview

The early-stage analysis cohort comprises **202 participants** (110 MSA and 92 PD). Participants were assigned to an existing, locked **Discovery set** (77 MSA and 64 PD; *n* = 141) and **held-out independent validation partition** (33 MSA and 28 PD; *n* = 61). The held-out partition belongs to the same study cohort; it is **not an external-center validation dataset**. Labels are `MSA = 1` and `PD = 0`.

The primary analysis is based on **30-second PSG epochs** and five sleep stages: wake (`W`), N1, N2, N3, and REM (`R` in the feature files).

| Feature representation | EEG | EOG | Chin EMG | Total |
| --- | ---: | ---: | ---: | ---: |
| One 30-second epoch | 198 | 48 | 32 | **278** |
| Stage-resolved, five stages | 990 | 240 | 160 | **1,390** |
| Whole-night, all epochs pooled | 198 | 48 | 32 | **278** |

- **Stage-resolved:** each epoch-level feature is averaged separately within `W`, `N1`, `N2`, `N3`, and `R` for each participant. A stage without valid observations may produce missing feature values.
- **Whole-night:** the same 278 epoch-level features are averaged **over all available epochs, regardless of sleep stage**. This is **not** an unweighted average of the five stage-specific means.
- **SOMNIS stacking:** one modality-specific XGBoost learner for each of EEG, EOG, and EMG produces probabilities; a support vector machine (SVM) combines their three outputs.
- **Comparators:** one XGBoost model on concatenated stage-resolved features; EEG-, EOG-, or EMG-only XGBoost; and a separate whole-night representation evaluated with the stacking pipeline.
- **Additional analyses:** leave-one-modality-out, leave-one-sleep-stage-out, XGBoost gain, three-input SVM SHAP, and effect-size replication of Discovery-selected features.

### Validation and thresholding

The Discovery set uses five outer stratified cross-validation folds, with training-only hyperparameter tuning and strict cross-fitted out-of-fold (OOF) predictions for stacking. Imputation and scaling are fitted only on the corresponding training subset. The held-out validation set is evaluated after final model development on the Discovery set.

The included model scripts select an operating threshold **using training-derived OOF predictions only**: among ROC thresholds satisfying **sensitivity >= 0.75** and **specificity >= 0.60**, choose the one with the highest balanced accuracy (BAC); if no threshold qualifies, choose the global maximum-BAC threshold. The code reports AUC, accuracy, BAC, sensitivity, specificity, PPV, and NPV with class-stratified, participant-level bootstrap 95% confidence intervals (2,000 replicates by default).

**Reproducibility note:** the 0.75/0.60 rule above describes the **code currently in this repository**. Match the final manuscript's stated operating-point rule and results before citing it as an exact reproduction of the paper.

## Repository structure

```text
SOMNIS/
├── README.md
├── .gitignore
├── preprocessing/
│   ├── psg_preprocessing.py
│   └── gssc_sleep_staging.py
├── features/
│   ├── extract_stage_features.py
│   ├── extract_whole_night_features.py
│   ├── add_modality_labels.py
│   └── prepare_dataset.py
├── models/
│   ├── somnis_stacking.py
│   ├── concatenated_xgboost.py
│   └── single_modality_xgboost.py
├── analyses/
│   ├── model_interpretation.py
│   ├── modality_ablation.py
│   └── sleep_stage_ablation.py
├── visualization/
│   ├── plot_model_interpretation.py
│   └── plot_ablation_results.py
└── environments/
    ├── feature_extraction.yml
    ├── gssc.yml
    └── modeling.yml
```

## Software environments

The original workflow ran in **three separate environments**; the two environments named `yasa_env` on the research machines were *not* identical.

| Processing stage | Original platform | Python | Example key dependencies |
| --- | --- | --- | --- |
| PSG preprocessing and feature extraction | Linux server | 3.12.12 | NumPy 2.3.5, SciPy 1.16.3, MNE 1.11.0, AntroPy 0.1.9 |
| GSSC sleep staging | Linux server | 3.9.25 | GSSC 0.0.9, MNE 1.8.0, PyTorch 2.8.0 |
| Training, ablations, interpretation and plots | Windows workstation | 3.10.19 | scikit-learn 1.7.2, XGBoost 3.1.3, SHAP 0.49.1 |

Example environment creation from the repository root:

```bash
conda env create -f environments/feature_extraction.yml
conda env create -f environments/gssc.yml
conda env create -f environments/modeling.yml
```

Activate only the relevant environment for each operation:

```bash
conda activate somnis-features
# preprocessing and features/*

conda activate somnis-gssc
# GSSC sleep staging

conda activate somnis-modeling
# models/*, analyses/*, visualization/*
```

The YAML files are **reconstructed core dependency specifications**, not exact transitive lockfiles. GSSC requires its upstream pretrained model files; PyTorch/CUDA installation may depend on platform and GPU availability. Verify installation and package compatibility on the target machine. No GSSC model weights are redistributed here.

## Required data and local folder layout

The input PSG files are **previously selected/cropped EEGLAB `.set` recordings**. Selection of valid recording intervals and other manual/clinical inclusion decisions are not performed by this pipeline.

A typical *local-only* directory layout (ignored by Git) is:

```text
data/
├── cut/
│   ├── MSA/MSA_001_selected.set
│   └── PD/PD_001_selected.set
├── preprocess/                  # 10-channel 100-Hz NPZ recordings
├── sleepstage/GSSC/              # per-recording GSSC stage CSVs
├── features/                     # epoch, stage and whole-night feature arrays
├── datasets/
│   ├── all/                      # modality-labeled feature arrays
│   └── early/
│       ├── stages/
│       │   ├── earlyMSA.npz
│       │   └── earlyPD.npz
│       └── wholenight/
│           ├── earlyMSA_wostages.npz
│           └── earlyPD_wostages.npz
└── labels/                       # original restricted split lists; NOT public
    ├── earlyMSA_train.txt
    ├── earlyMSA_test.txt
    ├── earlyPD_train.txt
    └── earlyPD_test.txt
```

`MSA_001` and `PD_001` are *illustrative filename patterns, not example patient data*. Do not substitute artificial split lists when attempting to reproduce the original study.

Final model NPZ inputs contain:

- `X`: participant-by-feature matrix (`1390` or `278` columns).
- `feature_names`: ordered feature names, including `W_` / `N1_` / `N2_` / `N3_` / `R_` prefixes for stage-resolved data.
- `channels`: per-column modality labels `EEG`, `EOG`, or `EMG` (despite the historical key name).
- `subject_ids`: participant identifiers (private).
- `dataset`: pre-existing `train` or `test` partition labels.
- `feature_types`: metadata generated by the feature-processing pipeline.

**Important:** do not change feature ordering or use an updated patient split when comparing against the archived manuscript results.

## Reproducing the workflow

Run commands **from the repository root**. All paths below are illustrative *local paths*; create and populate the private `data/` directory with authorized material first.

### 1. Preprocess PSG — `somnis-features`

```bash
python preprocessing/psg_preprocessing.py --input-root data/cut --output-root data/preprocess --groups MSA PD
```

Processing uses six EEG (`F3`, `F4`, `C3`, `C4`, `O1`, `O2`), two EOG (`L`, `R`) and two chin EMG (`Ch1`, `Ch2`) channels, with modality-specific filters, 50-Hz notch filtering and 100-Hz output sampling. MNE voltage-to-microvolt conversion is explicitly controlled by `--mne-data-unit` (default `V`). **Verify channel mappings, units, epoch boundaries, and the exact archived preprocessing settings before claiming numerical equivalence.**

### 2. Automatic sleep staging — `somnis-gssc`

```bash
python preprocessing/gssc_sleep_staging.py --input-root data/cut --output-root data/sleepstage/GSSC --groups MSA PD --device auto --strict-epoch-count
```

GSSC runs on the selected `.set` recordings, *not* on the processed NPZ. Confirm that stage timestamps and 30-second epoch indexing align with the feature windows.

### 3. Extract features — `somnis-features`

```bash
python features/extract_stage_features.py --preprocess-root data/preprocess --staging-root data/sleepstage/GSSC --output-root data/features --groups MSA PD --invalid-policy legacy-zero
python features/extract_whole_night_features.py --preprocess-root data/preprocess --output-root data/features --groups MSA PD --invalid-policy legacy-zero
python features/add_modality_labels.py --input-root data/features --output-root data/datasets/all --groups MSA PD --mode both
python features/prepare_dataset.py --input-root data/datasets/all --label-root data/labels --output-root data/datasets/early --groups MSA PD --mode both --expect-original-cohort
```

The `legacy-zero` mode is shown for comparison with the original failed-feature handling policy. The default `--invalid-policy nan` instead retains invalid calculations as missing values; **changing this policy may change results**. The stage and whole-night scripts share epoch feature caches: if you change extraction settings, recompute cached epoch files with `--overwrite-epochs` consistently.

`prepare_dataset.py` requires the four original restricted ID lists and **does not randomly re-split** the cohort. `--expect-original-cohort` asserts the stated Discovery and validation group counts.

### 4. Train models — `somnis-modeling`

First check the input structure without training:

```bash
python models/somnis_stacking.py --input-root data/datasets/early --mode both --check-only --expect-original-cohort
```

Then run the models:

```bash
python models/somnis_stacking.py --input-root data/datasets/early --output-root results/models --mode both --expect-original-cohort
python models/concatenated_xgboost.py --input-root data/datasets/early --output-root results/models --expect-original-cohort
python models/single_modality_xgboost.py --input-root data/datasets/early --output-root results/models --modality all --expect-original-cohort
```

`somnis_stacking.py --mode both` runs **separate** stage-resolved and whole-night models. It does not concatenate the 1,390- and 278-feature representations. The concatenated and single-modality comparator scripts use **stage-resolved features only**.

The reference stage-resolved stacking predictions are saved locally to:

```text
results/models/somnis/stages/predicted_probabilities_for_dca.csv
```

This file contains participant-level identifiers and predictions and must remain private.

### 5. Model interpretation and ablations — `somnis-modeling`

Run interpretation against the fixed main-model predictions:

```bash
python analyses/model_interpretation.py --input-root data/datasets/early --output-root results/interpretation --expect-original-cohort --reference-predictions results/models/somnis/stages/predicted_probabilities_for_dca.csv
```

The interpretation module reconstructs the **final Discovery-trained model** using the frozen main-model routines. It checks consistency against the reference predictions when that option is supplied. It does **not** rerun the outer CV. Top-10 XGBoost gain features per modality are selected in Discovery; the same locked 30 features are evaluated for Hedges' *g* in both partitions. The SVM meta-model is interpreted with three-input interventional SHAP.

Run leave-one-out ablations (each condition is independently tuned and trained):

```bash
python analyses/modality_ablation.py --input-root data/datasets/early --output-root results/ablation --expect-original-cohort
python analyses/sleep_stage_ablation.py --input-root data/datasets/early --output-root results/ablation --expect-original-cohort
```

Modality conditions: `Full`, `-EEG`, `-EOG`, `-EMG`. Sleep-stage conditions: `Full`, `-W`, `-N1`, `-N2`, `-N3`, `-R`. Completed runs can be resumed when their input and script signatures match; Full is independently run within each ablation family. Compare Full results with the main model before using the figures in a manuscript.

### 6. Generate plots without retraining — `somnis-modeling`

```bash
python visualization/plot_model_interpretation.py --analysis-dir results/interpretation/stages --which all
python visualization/plot_ablation_results.py --input-root results/ablation --output-dir results/ablation/visualization --which both
```

The second command requires completed modality **and** sleep-stage ablation summaries. Plots can be exported as PNG, PDF, and SVG (depending on the figure).

## Outputs and reproducibility checks

Output directories are ignored by Git. Main models generate a performance summary, bootstrap confidence intervals, subject-level predictions, classification threshold audit, and performance figures. Analyses generate feature-importance/effect-size tables and ablation summaries.

Before presenting the results as a reproduction of the associated paper, verify on the **original locked study data** that:

1. PSG preprocessing, stage alignment, channel mappings, and per-feature numeric values agree with the archival workflow.
2. The exact participant allocation and feature-name ordering match the original run.
3. Main and comparator predictions, operating thresholds, performance metrics, and confidence intervals match the manuscript.
4. Full-model predictions match the `Full` control of each ablation analysis.
5. Gain, SHAP, and effect-size results reproduce the submitted figures and supplementary tables.

Developer smoke-test options (if available) change the number of folds or candidate grids and **must not** be used to report research performance.

## Data availability and privacy

The PSG recordings and individual-level derived datasets originate from a clinical research cohort and are **not distributed in this repository**. Access to original data and participant split lists is subject to institutional approvals and applicable privacy rules; availability must be confirmed by the study investigators. **Do not commit** raw recordings, subject lists, participant-level feature matrices, prediction CSVs, protected metadata, or identifiers. An ignore rule cannot remove files that were already committed to Git history; inspect repository history before making it public.

The `.gitignore` intentionally excludes CSV and binary data by default. If an aggregate or synthetic example is approved for public release, add that specific file deliberately, document its provenance, and perform a privacy review first.

## Citation and license

This repository accompanies the SOMNIS study. The manuscript citation, author list, and DOI should be added after the bibliographic details are confirmed. **No open-source license is declared here**; permission to reuse or redistribute the code is not implied. The research team should approve an appropriate `LICENSE` before public release.

## Disclaimer

For scientific research and reproducibility only. No individual-patient diagnostic recommendations or clinical decisions should be based on this code without independent validation and regulatory review where applicable.
