#!/usr/bin/env python
"""Quantify the NR uncertainty of the merger-ringdown, mode by mode.

Compares two extractions of the *same* simulations -- different extrapolation
order, or different resolution -- in exactly the way a standalone ringdown model
is compared against NR: each waveform is anchored at its own ``|h_22|`` peak, and
the constant phase offset (which is arbitrary and differs between runs) is
removed at the attachment time.

The resulting mismatch is the level below which improving the model is
meaningless.

Example::

    python scripts/nr_uncertainty.py --ref data/nr_mrd_N2.h5 \\
        --alt data/nr_mrd_N3.h5 data/nr_mrd_N2_lowres.h5 --out data/nr_uncertainty.csv
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
from srmr.mismatch import amplitude_error, mismatch, phase_error


def compare(ref: Dataset, alt: Dataset, label: str, modes, window) -> pd.DataFrame:
    """Mode-by-mode differences between two extractions of the same simulations."""
    # Keep both row indices explicitly: the two datasets need not hold the same
    # simulations (a few have no lower resolution available).
    common = ref.params.reset_index(names="i").merge(
        alt.params.reset_index(names="j")[["name", "j"]], on="name"
    )
    if common.name.duplicated().any():
        raise ValueError("simulation names are not unique; cannot pair the datasets")

    rows = []
    amp0_22 = ref.mode((2, 2)).iv.amp0.values
    for lm in modes:
        a, b = ref.mode(lm, t_end=window.get(lm)), alt.mode(lm, t_end=window.get(lm))
        for i, j in zip(common.i, common.j):
            # Both waveforms are anchored at their own peak; the overall phase is
            # arbitrary, so compare phases relative to the attachment time.
            pa, pb = a.phase[i] - a.phase[i][0], b.phase[j] - b.phase[j][0]
            rows.append(
                dict(
                    comparison=label,
                    name=ref.params.name[i],
                    ell=lm[0],
                    m=lm[1],
                    t_end=a.tau[-1],
                    q=ref.params.q[i],
                    chi=ref.params.chi[i],
                    amp_ratio=a.iv.amp0[i] / amp0_22[i],
                    mismatch=mismatch(a.amp[i], pa, b.amp[j], pb, ref.dt),
                    amp_error=amplitude_error(b.amp[j], a.amp[i]),
                    phase_error=phase_error(pb, pa),
                    d_t_peak22=ref.params.t_peak22[i] - alt.params.t_peak22[j],
                    d_Mf=ref.params.Mf[i] - alt.params.Mf[j],
                    d_af=ref.params.af[i] - alt.params.af[j],
                )
            )
    return pd.DataFrame(rows)


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby(["comparison", "ell", "m"]).agg(
        n=("mismatch", "size"),
        mm_median=("mismatch", "median"),
        mm_p90=("mismatch", lambda x: np.quantile(x, 0.9)),
        amp_err_median=("amp_error", "median"),
        phase_err_median=("phase_error", "median"),
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ref", default="data/nr_mrd_N2.h5", help="reference dataset")
    p.add_argument("--alt", nargs="+", default=["data/nr_mrd_N3.h5", "data/nr_mrd_N2_lowres.h5"])
    p.add_argument("--out", default="data/nr_uncertainty.csv")
    p.add_argument("--modes", nargs="+", default=None)
    p.add_argument("--t-end", nargs="+", default=None, help=WINDOW_HELP)
    p.add_argument("--min-amp-ratio", type=float, default=1e-3)
    args = p.parse_args()

    ref = Dataset(args.ref)
    modes, window = parse_modes(args.modes, cfg.DEFAULT_MODES), parse_window(args.t_end)
    label = lambda path: os.path.basename(path).replace("nr_mrd_", "").replace(".h5", "")

    df = pd.concat(
        [compare(ref, Dataset(a), f"{label(args.ref)} vs {label(a)}", modes, window)
         for a in args.alt],
        ignore_index=True,
    )
    df.to_csv(args.out, index=False)

    pd.set_option("display.float_format", lambda x: f"{x:.3e}")
    print(f"\nModes with |h_lm| / |h_22| > {args.min_amp_ratio:g} at the attachment time:")
    print(summarise(df[df.amp_ratio > args.min_amp_ratio]).to_string())
    print("\nMedian |shift| of the (2,2) peak time and of the remnant, per comparison:")
    print(
        df.groupby("comparison")[["d_t_peak22", "d_Mf", "d_af"]]
        .apply(lambda g: g.abs().median())
        .to_string()
    )
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
