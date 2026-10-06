# Truncated diffusion samplers under power-law spectra: experiments

Code for the numerical experiments of the paper. Each experiment is one
self-contained script. All three run on a CPU; no GPU is needed.

| Script | What it tests | Data |
|---|---|---|
| `01_power_law_spectra.py` | Whether patch covariance spectra follow a power law (Assumption 1), with held-out model selection and bootstrap intervals over images | CIFAR-10/100 (downloaded); optional extra image or audio datasets |
| `02_gaussian_truncation.py` | Theorems 1–2 on Gaussians with an exact power-law spectrum: the two-sided truncation bound, rules for choosing K, and the scaling of the required K in ε and η | Synthetic, none needed |
| `03_gmm_patches.py` | Gaussian-mixture models of CIFAR patches: fit, Assumption 1 per component, truncation in a shared basis against the Gaussian prediction, and per-component bases (Proposition 4) | CIFAR-10/100 (downloaded) |

Total variation is always estimated by Monte Carlo from exact densities,
using TV(P, Q) = E_{(P+Q)/2} |tanh(½ log dP/dQ)|. No learned classifier is used.

## Setup

```bash
pip install -r requirements.txt
```

`torch`/`torchvision` are needed only to download CIFAR, `scikit-learn` only for
the CIFAR split in experiment 1, and `librosa`/`soundfile` only for optional
audio datasets.

## Running

Every script has a `--profile` option. `smoke` is a quick code test, not a result.

```bash
# Experiment 1: spectra (pilot = 8x8 and 16x16 patches; research = larger protocol)
python 01_power_law_spectra.py --profile research --cifar cifar10 cifar100

# Experiment 2: synthetic Gaussians (research uses d = 65536)
python 02_gaussian_truncation.py --profile research

# Experiment 3: patch mixtures (the configuration used in the paper)
python 03_gmm_patches.py --profile paper
```

Useful options:

- `--output DIR` sets the output root (default `results/`); `--data-dir DIR`
  sets where CIFAR is stored (default `data/`).
- Experiment 1 accepts extra datasets: `--image-root div2k=/path/to/DIV2K_HR`,
  `--audio-root maestro=/path/to/maestro`, and `--manifest NAME=file.csv` with
  columns `path,split,group` to fix splits and keep each image, recording or
  artist in one split. `--cifar none` skips CIFAR.
- Experiment 3 accepts `--datasets cifar10` to run one dataset, and
  `--data synthetic` to test the code without downloading CIFAR.

Each run writes its configuration, environment, CSV tables and figures to
`results/<experiment>/<profile>_<hash>/`, where the hash identifies the
configuration, and a ZIP of the outputs next to it.

## Outputs

**Experiment 1:** `spectrum_summary.csv` (selected model, power-law exponent and
bootstrap interval per dataset and patch size), `all_fit_candidates.csv`
(every model and rank interval with validation and test errors),
`*_sample_stability.csv`, and spectra as `.npz`.

**Experiment 2:** `truncation_curves.csv` (TV with confidence interval, Wiener
tail, bounds, and a flag for any possible bound violation), `selector_calibration.csv`,
`accuracy_scaling.csv` and `resolution_scaling.csv` (measured against predicted
slopes; slopes need at least three points in the asymptotic range).

**Experiment 3:** `gmm_fit.csv`, `em_trace.csv`, `component_spectra.csv`,
`component_assumption1_summary.csv`, `shared_basis_curves.csv`,
`shared_basis_required_K.csv`, `shared_basis_selectors.csv`,
`nongaussianity.csv`, `component_basis.csv`, `component_basis_summary.csv`, and
the fitted mixture as `.npz`.

## Runtime

Experiment 3 with `--profile paper` takes about 45–60 minutes per dataset on a
CPU, mostly EM for M = 16 and 32. Experiments 1 and 2 with `pilot` take a
few minutes; `research` takes considerably longer.
