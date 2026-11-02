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

- `grids/case118_aged_seed0_k2p5_s40_d50_a60_g2.m`: the aged network;
- `grids/case118_aged_seed0_k2p5_s40_d50_a60_g2.json`: distribution settings, sampled ages, degradations,
  and each eligible line's original and aged values;
- `grids/case118_aged_seed0_k2p5_s40_d50_a60_g2_age_distribution.png`: a histogram of the
  sampled ages of eligible bus-to-bus lines;
- `grids/case118_aged_seed0_k2p5_s40_d50_a60_g2_weibull_distribution.pdf`: the theoretical
  right-truncated Weibull age probability density, normalized over 0–60 years;
- `grids/case118_original.m`: the unmodified source network.

The script appends compact parameter tags: `seed` for the random seed, `k` for
Weibull shape, `s` for scale in years, `d` for maximum degradation in percent,
`a` for maximum age in years, and `g` for the degradation exponent. For example,
`--max-degradation 0.05` gives `d5`, while `0.5` gives `d50`. Decimal points become
`p` so the matching MATPOWER function name is valid. Set `network.name` in the
aged YAML configuration to the generated stem, for example
`case118_aged_seed0_k2p5_s40_d50_a60_g2`.

Existing network files and dataset paths are not renamed automatically. When using
a new filename, update the aged YAML's `network.name` and the aged dataset path in
the evaluation/estimation scripts. Different degradation parameters now produce
different names, which also separates Datakit's `_corrected.m` caches.

Select another IEEE system with `--case` (14, 30, 57, or 118; default: 118).
For example, from `src`:

```bash
../venv/bin/python create_aged_network.py --case 57 --seed 0
```

This selects `case57_ieee` and writes `grids/case57_original.m`,
`grids/case57_aged_seed0_k2p5_s40_d50_a60_g2.m`, its JSON report, and its age histogram.
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
`case118_aged_seed0_k2p5_s40_d50_a60_g2`. For IEEE-57, use the `case57` configs and
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

#### Prepare the complete official HF test dataset

From `src`, run:

```bash
python prepare_HF_test_dataset.py
```

From the repository root, use
`venv/bin/python src/prepare_HF_test_dataset.py`. The script downloads the complete
`gridfm/pf_small_case57_ieee` raw dataset into
`src/data/hf_full/case57_ieee/raw/`, reusing it when already complete. It checks
the published scenario count against all three raw tables, so the earlier
partitions-0–4 download cannot accidentally be used as the full dataset.
`--data-dir` selects another source-cache parent containing `case57_ieee/`;
`--local-files-only` requires already-cached HF data and model configuration.

The script loads the official
`gridfm/genco-pf-datakit/case57_ieee/base/seed0/case57_ieee_base_seed0.yaml`
configuration and its pretrained normalizer. It calls GraphKit's
`LitGridHeteroDataModule.setup("fit")` directly, including the preliminary seeded
shuffle and the official split by `load_scenario_idx`. The required settings are
`seed=0`, `test_ratio=0.1`, `val_ratio=0.1`, and
`split_by_load_scenario_idx=True`. Every graph in the resulting test split is
exported, with no additional sampling. The ratios apply to load-scenario groups,
so the test graph count need not be exactly 10% of the total graph count.

The standalone export is saved to `src/data/case57_test/` by default;
`--output-dir` can select another location. It contains the selected raw Parquet
tables, processed GraphKit graphs, configuration, pretrained normalizer stats,
a manifest, and the full train/validation/test scenario-ID and load-group-ID
lists for auditing. It prints the test count and verifies zero pairwise overlap
for both scenario IDs and load-scenario IDs. A completed export is verified and
reused on subsequent runs without accessing HF or the original dataset.

Processed graph dictionaries are copied byte for byte before normalization or PF
masking. Each graph retains its original `scenario_id`, node and edge stores,
features, and targets. Export filenames `processed/data_index_0.pt`, etc. use
local sequential indices to support ordinary GraphKit indexing; they do not
renumber the `scenario_id` inside a graph. `scenario_splits.pt["test"]` maps these
local indices to the original IDs. Only test graphs and test raw rows are copied;
the train/validation audit files contain IDs, not graphs.

From `src`, independently load the exported dataset with the official GraphKit
dataset class through the provided helper:

```python
from prepare_HF_test_dataset import load_test_dataset

dataset = load_test_dataset("data/case57_test")
graph = dataset[0]
print(len(dataset), graph.scenario_id.item())
```

This reads only the exported files and applies the bundled training normalizer
at access time. Optional `data_normalizer=` and `transform=` arguments allow
evaluation with another already-loaded model bundle and its PF task transforms.
The export also works as a local Datakit dataset root with
`error_evaluation.load_data_source()`; its processed cache prevents reprocessing.

#### Compare nominal, aged, and HF data with the first evaluator

From `src`, run:

```bash
python error_evaluation.py --case 57 --seed 0
```

Or, from the repository root:

```bash
venv/bin/python src/error_evaluation.py --case 57 --seed 0
```

This evaluates all four aged datasets under `src/datakit_out/case57/seed0/`:

- `case57_aged_seed0_k2p5_s40_d10_a60_g2_normal`
- `case57_aged_seed0_k2p5_s40_d10_a60_g2_all_out`
- `case57_aged_seed0_k2p5_s40_d50_a60_g2_normal`
- `case57_aged_seed0_k2p5_s40_d50_a60_g2_all_out`

Each comparison includes its nominal counterpart (`case57_normal` for `_normal`,
`case57_all_out` for `_all_out`) and partitions 0–4 of `gridfm/pf_small_case57_ieee`.
`--case` selects one IEEE
system (14, 30, 57, or 118), while `--seed` selects the local data folder and the
seed in the aged dataset names. Every dataset uses the corresponding
`gridfm/genco-pf-datakit/caseN_ieee/base/seed0` model and its normalizer.

The first evaluator retains its limit of 1,000 evaluated scenarios per dataset.
Use `--max-samples 5000` for up to 5,000, or a smaller positive value for a quick
check. Use `--ageing case57_aged_seed0_k2p5_s40_d10_a60_g2_normal` to evaluate
only that aged dataset. The model is loaded once, and nominal and HF predictions
are reused across comparisons. Each scenario supplies its own grid admittance and PF
inputs and targets, with the same four signed-error panels and printed mean/std
statistics as before.

Each PDF is saved as
`src/results/error_evaluation/case57/seed0/<aged-dataset-name>/pf_error_histograms.pdf`.
The four ageing cases have separate folders. Repeating the same case/seed/ageing
overwrites its existing PDF. `--output-dir` overrides the base directory while
retaining one subfolder per aged dataset. Paths are relative to the script, so either command works from any
working directory when the script path is provided correctly. The old
`case57_original` and `k2p5_scale40` dataset directories are no longer used by
this CLI.

#### Compare local perturbation and ageing conditions with an HF baseline

From the repository root, run:

```bash
venv/bin/python src/error_evaluation_2.py --case 57 --seed 0
```

`error_evaluation_2.py` reads existing Datakit datasets from
`src/datakit_out/case57/seed0/`. `--case` accepts 14, 30, 57, or 118 (default: 57),
and `--seed` selects the data folder and the seed in aged dataset names (default:
0). It loads the matching `gridfm/genco-pf-datakit/case57_ieee/base/seed0` model,
configuration, and normalizer; the model training seed stays 0 for every data seed.
It first evaluates the entire prepared official HF test split from
`src/data/case57_test/` (21,694 scenarios in the current case57 export), including
all of its original scenario IDs. Prepare it with `prepare_HF_test_dataset.py`
before evaluation. The default root follows `--case` as `data/caseN_test/`;
`--hf-test-dir` can select another matching prepared test export. The HF test split
always uses the official seed-0 split, independently of the local data seed.
The script prints its resolved path and sample count, verifies the split audit
and processed graph inventory before inference, and checks the evaluated scenario
IDs against the complete official test list afterward. It uses the exported
dataset directly, without downloading or processing the original HF dataset.
All nine local datasets must already exist, with `bus_data.parquet`, `gen_data.parquet`,
and `branch_data.parquet` under their `raw` directories. These can be partitioned
Parquet directories or individual files.

The script evaluates each dataset once and writes these fourteen comparisons to
separate PDFs (local names below omit the common `case57_` prefix):

| First condition | Second condition |
| --- | --- |
| `HF` | `normal` (all perturbations active) |
| `normal` | `one_out_topo` |
| `normal` | `one_out_gen` |
| `normal` | `one_out_adm` |
| `normal` | `all_out` |
| `all_out` | `one_out_topo` |
| `all_out` | `one_out_gen` |
| `all_out` | `one_out_adm` |
| `normal` | `aged_seed0_k2p5_s40_d50_a60_g2_normal` |
| `normal` | `aged_seed0_k2p5_s40_d10_a60_g2_normal` |
| `aged_seed0_k2p5_s40_d10_a60_g2_normal` | `aged_seed0_k2p5_s40_d50_a60_g2_normal` |
| `all_out` | `aged_seed0_k2p5_s40_d50_a60_g2_all_out` |
| `all_out` | `aged_seed0_k2p5_s40_d10_a60_g2_all_out` |
| `aged_seed0_k2p5_s40_d10_a60_g2_all_out` | `aged_seed0_k2p5_s40_d50_a60_g2_all_out` |

A fifteenth PDF, `all_conditions.pdf`, overlays the HF baseline followed by all
nine local conditions. Each PDF has
four panels: signed prediction errors (`prediction - actual`) for voltage
magnitude (p.u.), voltage angle (degrees), active power (MW), and reactive power
(MVAr). Vm/Va include all buses; P/Q use the PF task's masked targets, as in
`error_evaluation.py`. Colors identify the same dataset consistently across PDFs,
and legends use full dataset names in the top-left corner of every panel.

Outputs go to `src/results/error_evaluation_2/case57/seed0/`. Pairwise PDF names
are `<first_dataset>_vs_<second_dataset>.pdf`. The dedicated HF-versus-normal
comparison is `HF_vs_case57_normal.pdf`. Each dataset's predictions, targets,
and signed errors are also saved as `<dataset>_errors.parquet`, including
`HF_errors.parquet` for the baseline. The script evaluates all saved local
scenarios by default, including scenarios whose IDs contain solver gaps.
`--max-samples` limits only the local datasets; it never truncates the HF test set.
The distributions compare each condition's available scenarios; datasets need not
have equal sample counts or identical ground truths. Histogram bins span the
0.5th–99.5th percentiles across the curves in each panel, with fractions normalized
within the displayed range. Printed counts, means, and population standard
deviations use all finite errors, including tails.

After evaluation, the script prints one summary table per quantity and saves
`Vm_error_summary.csv`, `Va_error_summary.csv`, `P_error_summary.csv`, and
`Q_error_summary.csv` in the same output directory. Each table has these columns:
`exp name`, `number of samples`, `topo pert`, `gen pert`, `adm pert`, `mean error`,
`mean error/error std`, and `skewness around zero`. Perturbation columns contain
the actual type from the dataset's `raw/args.log` (`none` means disabled);
missing settings are reported as `unknown`. When the log contains several
generation entries, the last entry is used.
`number of samples` is the number of distinct evaluated PF scenario IDs in each
dataset, not the number of bus/error observations. It is the same across that
dataset's four summary rows, even though the quantities have different target counts.
The `HF` baseline is the first row in every summary table. Its perturbation types
are reported as `unknown` because the downloaded raw tables do not include a
generation-settings log.

Statistics use all finite errors, including histogram tails, separately for each
quantity so their physical units are not mixed. The ratio is the signed mean
divided by the population standard deviation (`ddof=0`). Skewness around zero is
the uncentered moment ratio `mean(error**3) / mean(error**2)**1.5`, without
subtracting the mean. A zero standard deviation gives an undefined mean/std
ratio; an all-zero error series gives undefined skewness. Undefined values print
as `NaN` and are saved as empty CSV cells.

To limit the local datasets while still evaluating the entire HF test set, select
a separate output folder:

```bash
venv/bin/python src/error_evaluation_2.py --case 57 --seed 0 \
  --max-samples 2 --output-dir /tmp/error_evaluation_2_check
```

### 5. Run impedance estimation without the notebook

From the repository root:

```bash
venv/bin/python src/aged_impedance_estimation.py \
  --case 57 \
  --seed 0 \
  --ageing case57_aged_seed0_k2p5_s40_d10_a60_g2_normal \
  --epochs 250
```

The standalone script runs Section 7's projected Adam experiment and impedance
plots; it excludes Section 8. `--case` accepts 14, 30, 57, or 118 (default: 57),
and `--seed` selects the experiment seed (default: 0). `--ageing` is the exact
aged Datakit case directory name. `--epochs` controls the number of optimization
epochs and defaults to 250. The example reads aged PF data from
`src/datakit_out/case57/seed0/case57_aged_seed0_k2p5_s40_d10_a60_g2_normal/`
and nominal line parameters from `src/grids/case57_original.m`.

Estimates and title-free impedance PDF plots at 300 DPI go to
`src/results/case57/seed0/case57_aged_seed0_k2p5_s40_d10_a60_g2_normal/`.
Signed PF error analysis belongs to
`src/error_evaluation.py`. TensorBoard events go to
`src/tb/case57/seed0/k2p5_s40_d10_a60_g2_normal/impedance-multiplier-denoising-TIMESTAMP/`.
Each ageing type has its own folder, with a timestamped subfolder for each new run.
Paths are resolved
relative to the script, so it can also be launched from another working directory.
The script retains the notebook's 250 epochs, learning rate, multiplier bounds,
and in-service/non-transformer learnability mask.

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
