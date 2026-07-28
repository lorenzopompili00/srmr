#!/usr/bin/env python
"""One simulation: NR vs. the standalone SEOBNRv5 merger-ringdown, mode by mode.

The reference NR extraction is overlaid with the same simulation at a lower
resolution and at a different extrapolation order, so the model error can be read
against the NR uncertainty.  Rows: amplitude, relative amplitude difference,
frequency, phase difference (offset removed at the attachment time).

Example::

    python scripts/plot_case.py --name SXS:BBH:0335 --out data/case.png
    python scripts/plot_case.py --index 200 --modes 2,2 3,3 --out data/case.png
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from srmr import config as cfg
from srmr.cli import parse_modes
from srmr.dataset import Dataset
from srmr.evaluate import predict
from srmr.mismatch import mismatch
from srmr.rd_model import V5Ansatz

# Same entity -> same hue as in plot_mismatches.py; line style carries identity too.
STYLE = {
    "NR": dict(color="#1a1a19", lw=1.8, ls="-", zorder=3),
    "SEOBNRv5": dict(color="#2a78d6", lw=1.4, ls="--", zorder=4),
    "NR resolution": dict(color="#1baf7a", lw=1.1, ls="-", zorder=2),
    "NR extrapolation": dict(color="#4a3aa7", lw=1.1, ls=":", zorder=2),
}


def _curves(ref: Dataset, i: int, lm: tuple[int, int], alts: dict, ansatz):
    """``label -> (amp, phase)`` for the reference, the model and the alternatives."""
    nr = ref.mode(lm)
    out = {"NR": (nr.amp[i], nr.phase[i]), "SEOBNRv5": predict(ref, i, lm, ansatz)}
    for label, (ds, j) in alts.items():
        alt = ds.mode(lm)
        # Each extraction is anchored at its own |h22| peak; the overall phase is
        # arbitrary, so shift it to the reference's value at the attachment time.
        out[label] = (alt.amp[j], alt.phase[j] - alt.phase[j][0] + nr.phase[i][0])
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="data/nr_mrd_N2.h5")
    p.add_argument("--lowres", default="data/nr_mrd_N2_lowres.h5")
    p.add_argument("--extrap", default="data/nr_mrd_N3.h5")
    p.add_argument("--index", type=int, default=0, help="row of the dataset")
    p.add_argument("--name", default=None, help="simulation name, instead of --index")
    p.add_argument("--modes", nargs="+", default=None)
    p.add_argument("--out", default="data/case.png")
    args = p.parse_args()

    ref, ansatz = Dataset(args.data), V5Ansatz()
    modes = parse_modes(args.modes, cfg.DEFAULT_MODES)
    i = args.index
    if args.name is not None:
        match = np.flatnonzero(ref.params.name.values == args.name)
        if not len(match):
            raise SystemExit(f"{args.name} is not in {args.data}")
        i = int(match[0])
    par, name = ref.params.iloc[i], ref.params.name[i]

    alts = {}
    for label, path in [("NR resolution", args.lowres), ("NR extrapolation", args.extrap)]:
        if not os.path.exists(path):
            continue
        ds = Dataset(path)
        match = np.flatnonzero(ds.params.name.values == name)
        if len(match):
            alts[label] = (ds, int(match[0]))

    fig, axes = plt.subplots(4, len(modes), figsize=(3.3 * len(modes), 8.6),
                             squeeze=False, sharex=True,
                             gridspec_kw=dict(height_ratios=[2, 1.2, 2, 1.2]))

    for col, lm in enumerate(modes):
        tau = ref.mode(lm).tau
        curves = _curves(ref, i, lm, alts, ansatz)
        amp_ref, phase_ref = curves["NR"]
        scale = np.nanmax(amp_ref)

        for label, (amp, phase) in curves.items():
            style = STYLE[label]
            axes[0, col].semilogy(tau, amp, label=label, **style)
            axes[2, col].plot(tau, -np.gradient(phase, ref.dt), **style)
            if label == "NR":
                continue
            axes[1, col].semilogy(tau, np.abs(amp - amp_ref) / scale, **style)
            axes[3, col].semilogy(
                tau, np.abs((phase - phase[0]) - (phase_ref - phase_ref[0])), **style
            )

        # The model residual vanishes identically at tau=0, so let the axes span a
        # fixed number of decades below the largest residual instead of auto-scaling.
        for row in (1, 3):
            top = max(np.nanmax(line.get_ydata()) for line in axes[row, col].lines)
            axes[row, col].set_ylim(top * 1e-5, top * 3)
        # The frequency is meaningless once the amplitude reaches the NR noise floor;
        # keep the axis on the range spanned by the reference waveform.
        omega_ref = -np.gradient(phase_ref, ref.dt)
        lo, hi = np.nanmin(omega_ref), np.nanmax(omega_ref)
        axes[2, col].set_ylim(lo - 0.1 * (hi - lo), hi + 0.1 * (hi - lo))

        mm = mismatch(*curves["SEOBNRv5"], amp_ref, phase_ref, ref.dt)
        axes[0, col].set_title(f"$({lm[0]},{lm[1]})$   $\\mathcal{{M}}$ = {mm:.1e}", fontsize=10)
        axes[3, col].set_xlabel(r"$t - t_{\rm match}\ [M]$")
        for ax in axes[:, col]:
            ax.grid(alpha=0.18, linewidth=0.6)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)

    axes[0, 0].set_ylabel(r"$|h_{\ell m}|$")
    axes[1, 0].set_ylabel(r"$\Delta |h_{\ell m}| \,/\, \max |h_{\ell m}|$")
    axes[2, 0].set_ylabel(r"$\omega_{\ell m}$")
    axes[3, 0].set_ylabel(r"$|\Delta \phi_{\ell m}|$  [rad]")
    axes[0, 0].legend(fontsize=7.5, frameon=False)

    fig.suptitle(
        f"{name}:  q={par.q:.2f}, $\\chi_1$={par.chi1:.2f}, $\\chi_2$={par.chi2:.2f}"
        f",  $M_f$={par.Mf:.3f}, $a_f$={par.af:.3f}",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(args.out, dpi=140)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
