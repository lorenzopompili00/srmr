#!/usr/bin/env python
"""Refit the free coefficients of the ansatz case by case, and compare with SEOBNRv5.

The mismatch obtained with the refitted coefficients is the floor set by the
*functional form* alone; the difference with respect to :mod:`scripts.eval_v5` is
what the polynomial interpolation across parameter space costs.

Example::

    python scripts/refit_v5.py --data data/nr_mrd_N2.h5 --out data/v5_refit.csv
"""

from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from srmr import config as cfg
from srmr.cli import WINDOW_HELP, parse_modes, parse_window
from srmr.dataset import Dataset
from srmr.evaluate import attachment
from srmr.fit import fit_coefficients
from srmr.mismatch import mismatch
from srmr.qnm_utils import mu, spherical_from_spheroidal
from srmr.rd_model import V5Ansatz, evaluate

_DS: Dataset | None = None


def _init(path: str) -> None:
    global _DS
    _DS = Dataset(path)


def _refit_one(job, ansatz, weight_phase=True):
    """Refit one (simulation, mode) and return the coefficients and the mismatch."""
    i, lm, t_end = job
    ds = _DS
    nr = ds.mode(lm, t_end=t_end)
    par, tau = ds.params.iloc[i], nr.tau
    p0 = ansatz.coefficients(lm, par.nu, par.chi)

    if lm not in cfg.MIXED_MODES:
        att = attachment(ds, i, lm)
        coeffs = fit_coefficients(ansatz, tau, nr.amp[i], nr.phase[i], att, p0,
                                  weight_phase_by_amplitude=weight_phase)
        amp, phase = evaluate(tau, att, ansatz, coeffs)
    else:
        ell, m = lm
        mm_lm = cfg.MIXED_MODES[lm]
        sph = ds.mode(lm, "spheroidal", t_end=t_end)
        part = ds.mode(lm, "partner", t_end=t_end)
        att_s, att_p = attachment(ds, i, lm, "spheroidal"), attachment(ds, i, lm, "partner")
        coeffs = fit_coefficients(ansatz, tau, sph.amp[i], sph.phase[i], att_s, p0,
                                  weight_phase_by_amplitude=weight_phase)
        coeffs_p = fit_coefficients(
            ansatz, tau, part.amp[i], part.phase[i], att_p,
            ansatz.coefficients(mm_lm, par.nu, par.chi),
            weight_phase_by_amplitude=weight_phase,
        )
        amp_s, phase_s = evaluate(tau, att_s, ansatz, coeffs)
        amp_p, phase_p = evaluate(tau, att_p, ansatz, coeffs_p)
        h = spherical_from_spheroidal(
            ell, m, par.af,
            amp_s * np.exp(1j * phase_s),
            amp_p * np.exp(1j * phase_p) / np.conj(mu(m, m, m, par.af)),
        )
        amp, phase = np.abs(h), np.unwrap(np.angle(h))

    return dict(
        name=par["name"], ell=lm[0], m=lm[1], t_end=tau[-1],
        q=par.q, nu=par.nu, chi=par.chi,
        amp_ratio=nr.iv.amp0[i] / ds.mode((2, 2)).iv.amp0[i],
        mismatch=mismatch(amp, phase, nr.amp[i], nr.phase[i], ds.dt),
        **coeffs,
        **{f"{k}_v5": v for k, v in p0.items()},
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="data/nr_mrd_N2.h5")
    p.add_argument("--out", default="data/v5_refit.csv")
    p.add_argument("--compare-to", default="data/v5_mismatches.csv",
                   help="output of eval_v5.py, to contrast with the refitted mismatches")
    p.add_argument("--modes", nargs="+", default=None)
    p.add_argument("--t-end", nargs="+", default=None, help=WINDOW_HELP)
    p.add_argument("--v5-phase-weighting", action="store_true",
                   help="fit the phase with uniform weights, as SEOBNRv5 did, instead of "
                        "weighting by the amplitude (which is what the mismatch asks for)")
    p.add_argument("--nproc", type=int, default=32)
    p.add_argument("--min-amp-ratio", type=float, default=1e-3)
    args = p.parse_args()

    ds = Dataset(args.data)
    window = parse_window(args.t_end)
    modes = parse_modes(args.modes, cfg.DEFAULT_MODES)
    jobs = [(i, lm, window.get(lm)) for lm in modes for i in range(len(ds))]
    worker = partial(_refit_one, ansatz=V5Ansatz(),
                     weight_phase=not args.v5_phase_weighting)

    with ProcessPoolExecutor(args.nproc, initializer=_init, initargs=(args.data,)) as pool:
        rows = list(tqdm(pool.map(worker, jobs, chunksize=8), total=len(jobs)))
    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)

    keep = df[df.amp_ratio > args.min_amp_ratio]
    summary = keep.groupby(["ell", "m"]).agg(
        n=("mismatch", "size"), ansatz_floor=("mismatch", "median")
    )

    # The comparison file only covers the modes it was generated for, so a mode
    # asked for here may simply be absent from it.  Say so rather than silently
    # dropping it from the table.
    if os.path.exists(args.compare_to):
        v5 = pd.read_csv(args.compare_to)
        merged = keep.merge(v5[["name", "ell", "m", "mismatch"]], on=["name", "ell", "m"],
                            suffixes=("_refit", "_v5"))
        if len(merged):
            v5_median = merged.groupby(["ell", "m"]).mismatch_v5.median()
            summary["v5_fits"] = v5_median
            summary["ratio"] = summary.v5_fits / summary.ansatz_floor
        missing = sorted(set(summary.index) - set(merged.groupby(["ell", "m"]).groups))
        if missing:
            print(f"\nnote: {args.compare_to} has no rows for {missing}; rerun "
                  f"eval_v5.py with the same --modes to fill in the comparison.")
    else:
        print(f"\nnote: {args.compare_to} not found; showing the ansatz floor only.")

    pd.set_option("display.float_format", lambda x: f"{x:.3e}")
    print(f"\nModes with |h_lm| / |h_22| > {args.min_amp_ratio:g}, median mismatch:")
    print(summary.to_string())
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
