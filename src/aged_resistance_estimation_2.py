"""Resistance estimation with optional aging-distribution KL regularization.

This entry point uses ``aged_impedance_estimation_2.py`` while optimizing only
resistance. Reactance is fixed at its nominal value throughout training. KL
regularization is off by default and can be enabled with ``--kl-weight``.
"""

import aged_impedance_estimation_2 as impedance_estimation


def main(argv=None):
    return impedance_estimation.main(argv, resistance_only=True)


if __name__ == "__main__":
    main()
