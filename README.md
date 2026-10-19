# GENCO challenge

This repository studies how ageing-related changes in transmission-line resistance and
reactance affect the errors produced by a pretrained GENCO power-flow model. The
workflow creates an aged IEEE 118-bus network, generates comparable datasets for the
original and aged networks, and compares their prediction-error distributions.

## 1. Install GridFM Datakit and GraphKit

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

## 2. Create the aged network

The remaining commands assume the working directory is `src`:

```bash
cd src
```

Run `create_aged_network.py` to download the original PGLib IEEE 118-bus case and
create an aged version. For example:

```bash
../venv/bin/python create_aged_network.py \
  --aged-lines-percent 10 \
  --min-res 0.8 \
  --max-res 1.2 \
  --min-react 0.9 \
  --max-react 1.1 \
  --seed 42 \
  --output grids/case118_aged.m
```

For every selected line, the script samples independent resistance and reactance
multipliers from the requested uniform ranges. The seed makes both line selection and
the sampled multipliers reproducible. The command creates:

- `grids/case118_aged.m`: the aged network;
- `grids/case118_aged.json`: the affected lines and their original and aged values;
- `grids/case118_original.m`: the unmodified source network.

The output names above match the network names used by the YAML files in
`src/configs`.

## 3. Generate the original and aged datasets

The two configurations use the same data-generation procedure but load different
MATPOWER networks. Generate both datasets from `src`:

```bash
../venv/bin/gridfm_datakit generate configs/case118.yaml
../venv/bin/gridfm_datakit generate configs/case118_aged.yaml
```

Dataset size, random seed, perturbations, solver settings, and output directories are
controlled by the corresponding YAML files. The current configurations write the
original and aged data beneath `datakit_out` and `datakit_out_aged`, respectively.

## 4. Compare model errors

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
