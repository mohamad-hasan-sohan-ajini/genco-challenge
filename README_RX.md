# Joint resistance and reactance experiment

Run these commands from the repository root, in order:

```bash
# Create logrx networks and generate nominal/aged PF data for all four cases, seed 0.
bash src/run_all_rx_data_generation.sh

# Fit independent R and X multipliers with frozen GENCO, using PF MSE only.
bash src/run_all_rx_estimation.sh --epochs 60

# Baseline distributions, recovery/deployment analysis, and statistical summaries.
bash src/run_all_rx_analysis.sh
```

Every stage defaults to IEEE-14, 30, 57, 118 and **seed 0 only**. Add
`--all-seeds` to each command for seeds 0–4, or use `--seeds 0 1` for a subset.
Execution is seed-first: every selected case for seed 0, then seed 1, etc.
All eligible scenarios are evaluated; there is no sample-limit option.

To inspect the networks and YAML configurations before generating data:

```bash
bash src/run_all_rx_data_generation.sh --prepare-only
```

Generation reuses completed datasets with matching generation records. It
refuses to append to unidentified or incomplete datasets; `--overwrite`
explicitly regenerates RX data. Datakit and its Julia/PowerModels dependencies
must already be installed as described in the main README.

## Ageing model and isolation

For each eligible in-service, non-transformer branch, sample one age `a` from
the existing right-truncated Weibull distribution (`k=2.5`, scale 40 years,
maximum age 60 years), and compute:

```text
D(a) = 0.5 * ln(1 + a/60) / ln(2)
R_aged = R_nominal * (1 + D(a))
X_aged = X_nominal * (1 + D(a))
```

The same age and degradation multiplier apply to R and X of a line. Transformer
parameters, out-of-service branches, and generator attachments are preserved.
The estimator learns R and X independently with multipliers bounded to [1,1.5];
it does not enforce the true shared multiplier. KL regularization is disabled
to match the existing PF-only resistance experiment.

Inputs and outputs use separate directories:

| Artifact | Directory |
|---|---|
| Nominal and logarithmically aged R/X networks | `src/grids_rx/` |
| Generated YAML configurations | `src/configs_rx/` |
| Nominal/aged PF datasets | `src/datakit_out_rx/caseN/seedS/` |
| Final estimator checkpoints | `src/results_rx/caseN/seedS/AGEING/pf_only/run-*/` |
| TensorBoard runs | `src/tb_rx/caseN/seedS/k2p5_s40_d50_a60_logrx/pf_only/` |
| Baseline PF errors | `src/results_rx/new_rx_error_distribution/` |
| Recovery and deployment analysis | `src/results_rx/new_rx_impedance_error_reduction/` |
| Statistical summaries | `src/results_rx/new_rx_statistical_tests/` |

Network names end in `logrx`, distinguishing them from resistance-only `logr`.
Model checkpoint training seed remains 0, independently of the experiment seed.
The original `new_` scripts and their experiment files keep their own paths.

## Analysis counterparts

- `new_rx_error_distribution.py`: nominal matched, aged matched, and aged-PF/
  nominal-impedance distributions; complete Parquet and NumPy error exports.
- `new_rx_impedance_error_reduction.py`: initial/final learning diagnostics,
  true R/X recovery, mean rows per case, learning curves, R/X parity plots,
  and aged PF predictions using the final estimated R **and X**. It validates
  that the checkpoint belongs to the selected final TensorBoard epoch.
- `new_rx_statistical_tests.py`: signed error mean, skewness about zero, and
  all six same-seed Wasserstein-1 comparisons for four conditions, independently
  for Vm, Va, P and Q. It writes per-seed and mean/std tables. With one seed,
  the sample std is undefined (`NaN`), rather than zero.

Each script accepts `--case`, `--seeds`, and `--all-seeds`. PDF names start with
`caseXX-`. Legends remain at the top: matched nominal green, matched aged
orange, mismatched red, and estimated impedance blue. Statistical/report
filenames include the selected seed set so seed-0 and five-seed reports coexist.

## Reporting the experiment in the paper

The experiment tests whether joint estimation recovers values close to the
actual aged R/X and improves the deployed PF error distributions. Its main
recovery CSV reports nominal and estimated MAE/RMSE for R and X separately,
plus combined complex-impedance RMSE:

```text
RMSE_Z = sqrt(mean((R_est - R_aged)^2 + (X_est - X_aged)^2))
P = 100 * (RMSE_nominal - RMSE_estimated) / RMSE_nominal
```

Recovery uses **actual aged MATPOWER parameters** on eligible lines.
TensorBoard's diagnostic reference is the admittance averaged over perturbed
scenarios, so its initial/final values are kept in separate `logged_*` columns.
The TensorBoard combined R/X RMSE averages concatenated R and X residuals;
it is not the same numerical scale as `RMSE_Z` above.

Use the measured `resistance_rmse_reduction_pct`,
`reactance_rmse_reduction_pct`, and `impedance_rmse_reduction_pct` to fill in
the paper's percentages. For example: “Joint estimation reduced resistance
RMSE by P_R%, reactance RMSE by P_X%, and impedance RMSE by P_Z%.” Compare
these values and PF statistics alongside the resistance-only results, stating
that the RX experiment ages both parameters. The percentages and whether
they support the proposed claim must come from the completed experiment;
the scripts do not assume an improvement or fabricate P.
