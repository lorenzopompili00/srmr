#!/usr/bin/env python
"""Mode-by-mode mismatches of the SEOBNRv5HM merger-ringdown as a standalone model.

All the input values at the attachment time (amplitude, its derivative, phase and
frequency) are read from NR, as are the remnant mass and spin; only the free
coefficients of the ansatz come from the SEOBNRv5 fits.

Example::

    python scripts/eval_v5.py --data data/nr_mrd_N2.h5 --out data/v5_mismatches.csv
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from srmr import config as cfg
from srmr.cli import WINDOW_HELP, parse_modes, parse_window
from srmr.dataset import Dataset
from srmr.evaluate import evaluate_dataset
from srmr.rd_model import V5Ansatz


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    """Per-mode quantiles of the mismatch and of the amplitude/phase errors."""
    return df.groupby(["ell", "m"]).agg(
        n=("mismatch", "size"),
        t_end=("t_end", "first"),
        mm_median=("mismatch", "median"),
        mm_p90=("mismatch", lambda x: np.quantile(x, 0.9)),
        mm_max=("mismatch", "max"),
        amp_err_median=("amp_error", "median"),
        phase_err_median=("phase_error", "median"),
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="data/nr_mrd_N2.h5")
    p.add_argument("--out", default="data/v5_mismatches.csv")
    p.add_argument("--modes", nargs="+", default=None, help="default: v5 minus (5,5), (3,2), (4,3)")
    p.add_argument("--t-end", nargs="+", default=None, help=WINDOW_HELP)
    p.add_argument("--optimize-phase", action="store_true",
                   help="maximise the overlap over a constant phase offset")
    p.add_argument("--no-enforce-peak", action="store_true",
                   help="drop the SEOBNRv5 (2,2) amplitude-peak safeguard")
    p.add_argument("--min-amp-ratio", type=float, default=1e-3,
                   help="ignore modes weaker than this fraction of the (2,2) at attachment")
    args = p.parse_args()

    ds = Dataset(args.data)
    ansatz = V5Ansatz(enforce_peak=not args.no_enforce_peak)
    df = evaluate_dataset(
        ds,
        ansatz,
        modes=parse_modes(args.modes, cfg.DEFAULT_MODES),
        optimize_phase=args.optimize_phase,
        t_end=parse_window(args.t_end),
    )

    df.to_csv(args.out, index=False)
    pd.set_option("display.float_format", lambda x: f"{x:.3e}")
    print(f"{len(ds)} simulations, phase optimisation: {args.optimize_phase}\n")
    print("All simulations:")
    print(summarise(df).to_string())
    print(f"\nModes with |h_lm| / |h_22| > {args.min_amp_ratio:g} at the attachment time:")
    print(summarise(df[df.amp_ratio > args.min_amp_ratio]).to_string())
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
