#!/usr/bin/env bash

# Generate matched nominal and aged datasets for every supported IEEE network.
# Run from src. Seeds 1-4 complement the existing seed-0 datasets.

set -euo pipefail

config_dir="$(mktemp -d /tmp/all-network-seeds-XXXXXX)"

for case_number in 14 30 57 118; do
  for seed in 1 2 3 4; do
    ageing="case${case_number}_aged_seed${seed}_k2p5_s40_d50_a60_logr"
    aged_config="${config_dir}/case${case_number}_aged_d50_seed${seed}.yaml"
    nominal_config="${config_dir}/case${case_number}_nominal_seed${seed}.yaml"

    echo "Generating IEEE-${case_number} aged network, seed ${seed}"

    ../venv/bin/python create_aged_network.py \
      --case "${case_number}" \
      --seed "${seed}" \
      --weibull-shape 2.5 \
      --weibull-scale 40 \
      --max-age 60 \
      --max-degradation 0.5

    sed \
      -e "s|case${case_number}_aged_seed0_k2p5_s40_d50_a60_logr|${ageing}|" \
      -e "s|./datakit_out/case${case_number}/seed0|./datakit_out/case${case_number}/seed${seed}|" \
      -e "s|^  seed: 0|  seed: ${seed}|" \
      "configs/case${case_number}_aged_d50.yaml" > "${aged_config}"

    # Keep every stochastic setting identical to the aged configuration while
    # selecting the original network. Reusing the seed matches the PF scenario,
    # topology, generation, and admittance-perturbation random streams.
    sed \
      -e "s|./datakit_out/case${case_number}/seed0|./datakit_out/case${case_number}/seed${seed}|" \
      -e "s|^  seed: 0|  seed: ${seed}|" \
      "configs/case${case_number}.yaml" > "${nominal_config}"

    echo "Generating IEEE-${case_number} nominal Datakit dataset, seed ${seed}"
    ../venv/bin/gridfm_datakit generate "${nominal_config}"

    echo "Generating IEEE-${case_number} aged Datakit dataset, seed ${seed}"
    ../venv/bin/gridfm_datakit generate "${aged_config}"
  done
done

echo "Finished generating 16 matched nominal/aged dataset pairs (32 datasets)."
echo "Temporary generated configs: ${config_dir}"
