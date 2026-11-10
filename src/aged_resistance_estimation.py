"""Estimate aged line resistance while keeping reactance fixed at nominal values.

This is the resistance-only entry point for the shared GENCO-based estimation
pipeline in ``aged_impedance_estimation.py``. Its command-line arguments are the
same, but only resistance multipliers are optimized.
"""

import aged_impedance_estimation as impedance_estimation


def main(argv=None):
    return impedance_estimation.main(argv, resistance_only=True)


if __name__ == "__main__":
    main()
