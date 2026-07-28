#!/usr/bin/env python
"""Histograms of the mode-by-mode merger-ringdown mismatches, on a log x axis.

Shows, per mode, the SEOBNRv5 model against two references: the floor set by the
ansatz alone (coefficients refitted case by case) and the NR uncertainty
(resolution and extrapolation order).  Modes whose amplitude at the attachment
time is a negligible fraction of the (2,2) are dropped -- they are numerical
noise, not signal.

Example::

    python scripts/plot_mismatches.py --out data/mismatches.png
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from srmr import config as cfg
from srmr.cli import parse_modes

# Categorical slots 1, 2, 3 and 7 of the reference palette, in fixed order.
# Identity is carried by line style as well as hue, so the panels stay readable
# without colour.
SERIES = [
    ("SEOBNRv5 fits", "#2a78d6", "-", True),
    ("ansatz floor (refit)", "#eb6834", "-", False),
    ("NR resolution", "#1baf7a", "--", False),
    ("NR extrapolation", "#4a3aa7", ":", False),
]


def _collect(args, modes) -> dict[str, pd.DataFrame]:
    """Load the available inputs into ``{series label: table}``."""
    out = {}
    keep = lambda d: d[(d.amp_ratio > args.min_amp_ratio) & d.mismatch.notna()]
    if os.path.exists(args.model):
        out["SEOBNRv5 fits"] = keep(pd.read_csv(args.model))
    if os.path.exists(args.refit):
        out["ansatz floor (refit)"] = keep(pd.read_csv(args.refit))
    if os.path.exists(args.nr):
        nr = keep(pd.read_csv(args.nr))
        for label, key in [("NR resolution", "lowres"), ("NR extrapolation", "N3")]:
            sub = nr[nr.comparison.str.endswith(key)]
            if len(sub):
                out[label] = sub
    return {k: v[v.set_index(["ell", "m"]).index.isin(modes)] for k, v in out.items()}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="data/v5_mismatches.csv")
    p.add_argument("--refit", default="data/v5_refit.csv")
    p.add_argument("--nr", default="data/nr_uncertainty.csv")
    p.add_argument("--out", default="data/mismatches.png")
    p.add_argument("--modes", nargs="+", default=None)
    p.add_argument("--min-amp-ratio", type=float, default=1e-3)
    args = p.parse_args()

    modes = parse_modes(args.modes, cfg.DEFAULT_MODES)
    data = _collect(args, modes)
    if not data:
        raise SystemExit("no input tables found")

    lo = max(1e-12, min(d.mismatch[d.mismatch > 0].min() for d in data.values()))
    hi = max(d.mismatch.max() for d in data.values())
    bins = np.logspace(np.log10(lo), np.log10(hi), 55)

    ncols = min(len(modes), 2)
    nrows = int(np.ceil(len(modes) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.4 * ncols, 3.5 * nrows),
                             squeeze=False, sharex=True)

    for ax, lm in zip(axes.ravel(), modes):
        for label, color, style, fill in SERIES:
            if label not in data:
                continue
            d = data[label]
            values = d[(d.ell == lm[0]) & (d.m == lm[1])].mismatch.values
            if not len(values):
                continue
            ax.hist(values, bins=bins, histtype="stepfilled" if fill else "step",
                    color=color, linestyle=style, linewidth=1.6,
                    facecolor=color + "33" if fill else "none", edgecolor=color,
                    label=f"{label}  (median {np.median(values):.1e})")
        ax.set_xscale("log")
        ax.set_title(f"$({lm[0]},{lm[1]})$", fontsize=11)
        ax.grid(alpha=0.18, linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.set_ylim(0, ax.get_ylim()[1] * 1.35)  # headroom for the legend
        ax.legend(fontsize=7.5, frameon=False, loc="upper left")

    for ax in axes[-1]:
        ax.set_xlabel("mismatch")
    for ax in axes[:, 0]:
        ax.set_ylabel("number of simulations")
    for ax in axes.ravel()[len(modes):]:
        ax.set_visible(False)

    fig.suptitle(
        "Standalone merger-ringdown mismatch against NR"
        f"   (modes with $|h_{{\\ell m}}|/|h_{{22}}| > {args.min_amp_ratio:g}$"
        " at the attachment time)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(args.out, dpi=140)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
