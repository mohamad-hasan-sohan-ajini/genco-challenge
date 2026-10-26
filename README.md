# GENCO challenge

This repository studies how ageing-related changes in transmission-line resistance and
reactance affect the errors produced by a pretrained GENCO power-flow model. The
workflow creates an aged IEEE 118-bus network, generates comparable datasets for the
original and aged networks, and compares their prediction-error distributions.

## 1. Paper

[Read the paper (PDF)](GridFM__to_estimate_Aged_grid_Admittance_Matrix.pdf).

## 2. How to run the code

### 1. Install GridFM Datakit and GraphKit

Python 3.10, 3.11, or 3.12 is recommended. From the repository root, create a virtual
environment and install the two GridFM packages:

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install gridfm-datakit gridfm-graphkit jupyterlab tensorboard
```

Set up the Julia packages used by GridFM Datakit's PowerModels solver:

```bash
gridfm_datakit setup_pm
```

### 2. Create the aged network

The remaining commands assume the working directory is `src`:

```bash
cd src
```

Run `create_aged_network.py` to download the original PGLib IEEE 118-bus case and
create an aged version. For example:

```bash
../venv/bin/python create_aged_network.py \
  --weibull-shape 2.5 \
  --weibull-scale 40 \
  --max-age 60 \
  --max-degradation 0.5 \
  --degradation-exponent 2 \
  --seed 0 \
  --output grids/case118_aged.m
```

Every eligible line (in-service, non-transformer branch) receives an age sampled
from a Weibull distribution conditioned on ages between 0 and 60 years, using
inverse-CDF sampling rather than clipping. The fractional degradation is
`D = 0.5 * (age / 60)^2`, and both R and X are multiplied by the same `1 + D`.
All five parameters shown above are configurable; the seed makes the sampled ages
reproducible. Transformer branches, out-of-service branches, and generator data
remain unchanged. The command creates:

- `grids/case118_aged_seed0_k2p5_scale40.m`: the aged network;
- `grids/case118_aged_seed0_k2p5_scale40.json`: distribution settings, sampled ages, degradations,
  and each eligible line's original and aged values;
- `grids/case118_aged_seed0_k2p5_scale40_age_distribution.png`: a histogram of the
  sampled ages of eligible bus-to-bus lines;
- `grids/case118_aged_seed0_k2p5_scale40_weibull_distribution.pdf`: the theoretical
  right-truncated Weibull age probability density, normalized over 0–60 years;
- `grids/case118_original.m`: the unmodified source network.

The script appends the seed, Weibull shape (`k`) and scale to the base output
filename. Decimal points become `p` so the matching MATPOWER function name is
valid. Set `network.name` in the aged YAML configuration to the generated stem,
for example `case118_aged_seed0_k2p5_scale40`.

Select another IEEE system with `--case` (14, 30, 57, or 118; default: 118).
For example, from `src`:

```bash
../venv/bin/python create_aged_network.py --case 57 --seed 0
```

This selects `case57_ieee` and writes `grids/case57_original.m`,
`grids/case57_aged_seed0_k2p5_scale40.m`, its JSON report, and its age histogram.
Without `--output`, the base filename is `grids/case<case>_aged.m`.
An explicit `--output` still overrides that base path.

### 3. Generate the original and aged datasets

The two configurations use the same data-generation procedure but load different
MATPOWER networks. Generate both datasets from `src`:

```bash
../venv/bin/gridfm_datakit generate configs/case118.yaml
../venv/bin/gridfm_datakit generate configs/case118_aged.yaml
```

Dataset size, random seed, perturbations, solver settings, and output directories are
controlled by the corresponding YAML files. For the seed-0 examples, set
`settings.seed: 0` and `settings.data_dir: "./datakit_out/case118/seed0"` in both
IEEE-118 configs, and set the aged `network.name` to
`case118_aged_seed0_k2p5_scale40`. For IEEE-57, use the `case57` configs and
`./datakit_out/case57/seed0`. Datakit adds the network name and `raw` to
`settings.data_dir`. When changing the seed, update the directory and aged
network filename as well; YAML paths are literal, not templates.

### 4. Compare model errors

Open `1_error_dist.ipynb` and run all cells to evaluate the original and aged datasets
and plot their error distributions:

```bash
../venv/bin/jupyter lab 1_error_dist.ipynb
```

Before running the notebook, make sure its `DATASET_ROOTS` mapping points to the
dataset directories produced by the two YAML configurations. The notebook downloads
the pretrained checkpoint and normalizer, evaluates both datasets, saves prediction
tables under `results/local_predictions`, and plots the original-versus-aged error
distributions.

### 5. Run impedance estimation without the notebook

From the repository root:

```bash
venv/bin/python src/aged_impedance_estimation.py --case 57 --seed 0
```

The standalone script runs Section 7's projected Adam experiment and its plots;
it excludes Section 8. `--case` accepts 14, 30, 57, or 118 (default: 118), and
`--seed` selects the experiment seed (default: 0). The example reads from
`src/datakit_out/case57/seed0/case57_original/` and
`src/datakit_out/case57/seed0/case57_aged_seed0_k2p5_scale40/`.
Changing the case or seed updates all paths automatically. Network generation
must use the same seed for the original and aged datasets.

Estimates, prediction tables, metrics, and title-free PDF plots at 300 DPI go to
`src/results/case57/seed0/`. TensorBoard events go to
`src/tb/case57/seed0/impedance-multiplier-denoising-TIMESTAMP/`. Paths are resolved
relative to the script, so it can also be launched from another working directory.
The script retains the notebook's 250 epochs, learning rate, multiplier bounds,
and in-service/non-transformer learnability mask. Estimation scenarios are also
used for the final PF comparison; those plots are not held-out evaluation.

IEEE-118 retains `gridfm/genco-pf-transfer/finetune/case118_250000`. IEEE-14/30/57
load matching config/checkpoint/normalizer bundles from `gridfm/genco-pf-datakit`,
using the Base model with training seed 0 independently of the experiment seed.
Older configs are converted in memory for the installed Graphkit version.
Existing IEEE-118 outputs in the old seed-only directories are not moved.

### 6. Replot saved results for the paper

```bash
venv/bin/python src/paper_plots.py --case 57 --seed 0
```

`--case` accepts 14, 30, 57, or 118 (default: 118); the default seed is 0.
This loads the saved estimator and prediction tables from
`src/results/case57/seed0/` without retraining. Original-network inference is
only needed if its cached nominal PF prediction table is missing. Legacy
IEEE-118 results under `src/results/local_predictions/seed0/` remain supported
when the corresponding new case-specific directory is absent.
The first plotting method makes a two-panel Vm/Va error histogram with nominal
impedance in red, aged impedance in green, and estimated impedance in blue.
Both panels have legends. It saves
`estimated_impedance_pf_error_histograms_replot.pdf` beside the original plot,
at 300 DPI with no titles. Add further plotting-method calls in `main()`.

The script also plots original PF observations with nominal impedance (orange)
against aged PF observations with aged impedance (green), with
Vm/Va panels and a legend on each. This is saved as
`nominal_vs_aged_pf_error_histograms_replot.pdf`. On its first run, the original
PF baseline is evaluated with the saved GENCO checkpoint and cached as
`case57_original_pf_ground_truth_predictions.parquet` in the seed's results
directory; subsequent runs reuse that table.

All three figures and their printed statistics use the same case names and
colours: nominal impedance + aged PF (red), aged impedance + aged PF (green),
estimated impedance + aged PF (blue), and nominal impedance + nominal PF (orange).

`plot_tensorboard_impedance_rmse(case)` also reads seeds 0–4 from
`src/tb/case57/seedXX/` for this example
and plots the logged reactance and resistance mismatch RMSE against epoch, with
seed curves and an upper-right legend in each panel. It selects the latest
run containing both tags for each seed and uses all raw logged points, without
TensorBoard smoothing or downsampling. Seeds with missing logs or checkpoints
are reported and skipped. Separate figures are saved under `src/results/case57/`
as `reactance_mismatch_rmse_seeds0-4_replot.pdf` and
`resistance_mismatch_rmse_seeds0-4_replot.pdf` when all five seeds are present;
otherwise their filenames list the included seeds. Legacy IEEE-118 logs under
`src/tb/seedXX/` are also supported.
