"""RX baseline error distributions for all cases; seed 0 by default, --all-seeds for 0-4."""

import rx_common as rx


def main(argv=None):
    parser = rx.parser(__doc__)
    parser.add_argument("--recompute", action="store_true", help="Re-evaluate existing complete RX baselines")
    options = parser.parse_args(argv)
    cases, seeds = rx.selected(options)
    models = {}
    for seed in seeds:
        for case in cases:
            print(f"RX baseline: case{case}, seed{seed}; all eligible scenarios")
            if case not in models:
                models[case] = rx.pf.load_huggingface_model(rx.pf.model_spec(case))
            tables = rx.baseline_tables(case, seed, models[case], options.recompute)
            directory = rx.BASELINES / f"case{case}" / f"seed{seed}"
            rx.plot_groups(tables, directory, case, seed)
            nominal, aged = rx.dataset_roots(case, seed)
            rx.write_manifest(directory / f"case{case}-seed{seed}-baseline_manifest.json",
                              case=case, seed=seed, ageing=rx.aged_name(case, seed),
                              nominal_dataset=str(nominal), aged_dataset=str(aged),
                              all_eligible_scenarios=True)


if __name__ == "__main__":
    main()
