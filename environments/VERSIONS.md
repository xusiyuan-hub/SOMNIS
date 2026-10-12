# SOMNIS environment versions and installation

These files describe **three different environments**, not one merged environment.
Versions below were read from the study team's environments on October 12, 2026.
They are **key direct/package versions only**, not full resolved dependency lockfiles.

| Pipeline | Machine | Conda environment observed | Python |
|---|---|---|---|
| Preprocessing + feature extraction | Linux server | `yasa_env` | `3.12.12` |
| GSSC sleep staging | Linux server | `gssc` | `3.9.25` |
| SOMNIS models, ablations, interpretation and figures | Windows PC | `yasa_env` | `3.10.19` |

Although two machines both used an environment named `yasa_env`, **they are different environments** with different Python/library versions.
These scripts intentionally separate the three software stacks.

## Installation

From the SOMNIS repository root, choose **one** environment at a time:

```bash
conda env create -f environments/feature_extraction.yml
conda activate somnis-features
```

```bash
conda env create -f environments/gssc.yml
conda activate somnis-gssc
```

```bash
conda env create -f environments/modeling.yml
conda activate somnis-modeling
```

The `.txt` files contain the corresponding pinned **Python packages** if you prefer an existing Python environment:

```bash
python -m pip install -r environments/feature_extraction.txt
```

Note: installing a `.txt` alone does **not** choose the Python version. The YAML files do.
Conda does not guarantee that an identical build is available on every OS; an explicit full conda export from each originating platform is more rigorous.

### PyTorch / GSSC note

The original GSSC environment had `gssc==0.0.9`, `torch==2.8.0`, and no installed `torchaudio` package (not needed by the supplied staging script).
GSSC upstream: https://github.com/jshanna100/gssc
PyPI release: https://pypi.org/project/gssc/0.0.9/
For a GPU installation, follow the official PyTorch installation guidance for the target CUDA driver/platform before installing other dependencies, and verify the resolved torch version and `torch.cuda.is_available()`.
A pip installation of `torch==2.8.0` does not, by itself, establish identical GPU/CUDA behavior to the study server.
GSSC model/checkpoint selection should be recorded and validated separately.

### A version discrepancy to resolve in the manuscript

The currently uploaded Main Manuscript reports **Python 3.13.9**, scikit-learn 1.7.2, XGBoost 3.1.3 and SHAP 0.49.1 in the modeling Methods/Key Resources Table. The examined **local modeling Python was 3.10.19**, while these package versions match. Confirm which interpreter actually generated the final published results (e.g. from historical logs) and correct the manuscript if needed. Do not change the environment merely to make the version text agree.

## Save the actual original environment snapshots (recommended)

On the Linux server, run these separately (do not change installed software):

```bash
conda activate yasa_env
python -m pip freeze > requirements_feature_original.txt
conda env export --no-builds > environment_feature_original.yml

conda activate gssc
python -m pip freeze > requirements_gssc_original.txt
conda env export --no-builds > environment_gssc_original.yml
```

On the Windows machine, you already created `requirements_model_original.txt` in the modeling environment. A fuller platform snapshot can also be exported:

```powershell
conda activate yasa_env
conda env export --no-builds > environment_model_original.yml
```

Review exported files for private local paths, credentials, institution-only package indexes, file-URL links and usernames before posting anything publicly. Retain original exports for audit, and use the curated `.yml` / `.txt` files as easy-to-read installation recipes.

## Validation after installing an environment

```bash
python --version
python -m pip check
python -m pip list --format=freeze
```

GSSC optional GPU test:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

These version manifests cannot prove numerical equivalence by themselves. Test the stage/whole-night extraction against archived features and the model predictions against archived performance outputs before claiming full reproduction.
