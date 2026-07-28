#!/usr/bin/env python
"""Overview plots of the NR merger-ringdown dataset itself -- no model involved.

Two figures:

* ``<prefix>_params.png`` -- what the catalogue covers: spins, mass ratio, remnant,
  and how strong each mode is relative to the (2,2) at the attachment time.
* ``<prefix>_modes.png``  -- every waveform, mode by mode: the raw amplitude, then
  the two quantities a model actually has to reproduce, ``A~`` and ``phi~``, i.e.
  what is left once the ``n = 0`` QNM is divided out.

Example::

    python scripts/plot_dataset.py --out-prefix data/dataset
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from srmr import config as cfg
from srmr.cli import parse_modes
from srmr.dataset import Dataset
from srmr.qnm_utils import qnm_omega_tau

#: Mass ratio is a magnitude, so it gets one hue light->dark.
QRAMP = LinearSegmentedColormap.from_list("q", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab"])
#: Modes are identities, so they get a fixed categorical order, never cycled.
MODE_COLOURS = ["#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7",
                "#eda100", "#e87ba4", "#008300", "#e34948"]
#: With more than ~4 overlapping series, colour alone stops being enough.
MODE_STYLES = ["-", "--", ":", "-."]


def _tidy(ax):
    ax.grid(alpha=0.18, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def plot_parameters(ds: Dataset, modes, out: str, min_amp_ratio: float) -> None:
    p = ds.params
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.4))
    norm = Normalize(1, p.q.max())

    ax = axes[0, 0]
    sc = ax.scatter(p.chi1, p.chi2, c=p.q, cmap=QRAMP, norm=norm, s=18,
                    edgecolor="#1a1a19", linewidth=0.25)
    ax.set_xlabel(r"$\chi_1$")
    ax.set_ylabel(r"$\chi_2$")
    ax.set_title(f"spins, coloured by mass ratio  ({len(p)} simulations)", fontsize=10)
    fig.colorbar(sc, ax=ax, label="$q = m_1/m_2$")

    ax = axes[0, 1]
    ax.hist(p.q, bins=np.logspace(0, np.log10(p.q.max()), 40), color="#2a78d6",
            edgecolor="#1a1a19", linewidth=0.4)
    ax.set_xscale("log")
    ax.set_xlabel("$q$")
    ax.set_ylabel("simulations")
    ax.set_title(f"mass-ratio coverage: $q$ = {p.q.min():.2f} to {p.q.max():.1f}", fontsize=10)

    ax = axes[1, 0]
    sc = ax.scatter(p.af, p.Mf, c=p.q, cmap=QRAMP, norm=norm, s=18,
                    edgecolor="#1a1a19", linewidth=0.25)
    ax.set_xlabel("$a_f$")
    ax.set_ylabel("$M_f$")
    ax.set_title("remnant, from the NR metadata", fontsize=10)
    fig.colorbar(sc, ax=ax, label="$q$")

    ax = axes[1, 1]
    amp0_22 = ds.mode((2, 2)).iv.amp0.values
    bins = np.logspace(-9, 0.3, 55)
    # (2,2) is trivially 1 here, and its spike squashes everything else.
    for k, lm in enumerate([m for m in modes if m != (2, 2)]):
        ratio = ds.mode(lm).iv.amp0.values / amp0_22
        ax.hist(ratio, bins=bins, histtype="step", color=MODE_COLOURS[k + 1],
                linestyle=MODE_STYLES[k % len(MODE_STYLES)], linewidth=1.6,
                label=f"$({lm[0]},{lm[1]})$   {int((ratio > min_amp_ratio).sum())} usable")
    ax.axvline(min_amp_ratio, color="#1a1a19", linestyle=":", linewidth=1.2)
    ax.annotate(f"cut at {min_amp_ratio:g}", (min_amp_ratio, ax.get_ylim()[1] * 0.92),
                fontsize=8, ha="right", rotation=90, va="top", color="#1a1a19")
    ax.set_xscale("log")
    ax.set_xlabel(r"$|h_{\ell m}| / |h_{22}|$ at the attachment time")
    ax.set_ylabel("simulations")
    ax.set_title("mode hierarchy; odd $m$ vanishes at $q=1$, $\\chi_1=\\chi_2$", fontsize=10)
    ax.legend(fontsize=8, frameon=False)

    for ax in axes.ravel():
        _tidy(ax)
    fig.suptitle("NR merger-ringdown dataset: parameter-space coverage", fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    print(f"Wrote {out}")


def plot_modes(ds: Dataset, modes, out: str, min_amp_ratio: float, max_curves: int) -> None:
    p = ds.params
    norm = Normalize(1, p.q.max())
    amp0_22 = ds.mode((2, 2)).iv.amp0.values

    fig, axes = plt.subplots(3, len(modes), figsize=(3.3 * len(modes), 8.6), squeeze=False)
    for col, lm in enumerate(modes):
        nr = ds.mode(lm)
        keep = np.flatnonzero(
            (nr.iv.amp0.values / amp0_22 > min_amp_ratio) & np.isfinite(nr.amp).all(axis=1)
        )
        if max_curves and len(keep) > max_curves:
            keep = keep[np.linspace(0, len(keep) - 1, max_curves).round().astype(int)]
        # keep the visual density roughly constant however many curves are drawn
        alpha = float(np.clip(70 / max(len(keep), 1), 0.06, 0.40))

        amp_tilde, phi_tilde = [], []
        for i in keep:
            par = p.iloc[i]
            colour = QRAMP(norm(par.q))
            omega_qnm, tau_qnm = qnm_omega_tau(lm[0], lm[1], par.af, par.Mf)
            tau, amp, phase = nr.tau, nr.amp[i], nr.phase[i]
            style = dict(color=colour, lw=0.5, alpha=alpha)

            axes[0, col].semilogy(tau, amp / par.nu, **style)
            # what a model must actually produce: the QNM divided out
            amp_tilde.append(amp / par.nu * np.exp(tau / tau_qnm))
            phi_tilde.append(phase - phase[0] + omega_qnm * tau)
            axes[1, col].plot(tau, amp_tilde[-1], **style)
            axes[2, col].plot(tau, phi_tilde[-1], **style)

        axes[0, col].set_title(f"$({lm[0]},{lm[1]})$   {len(keep)} simulations", fontsize=10)
        axes[2, col].set_xlabel(r"$t - t_{\rm match}$  [M]")
        axes[1, col].set_yscale("log")
        # A handful of weak modes go to NR noise at late times; without clipping,
        # those few curves set the axis and hide the other several hundred.
        for row, data, qs, pad in ((1, amp_tilde, (0.5, 99.5), 1.6),
                                   (2, phi_tilde, (0.5, 98.0), 1.15)):
            lo, hi = np.nanpercentile(np.concatenate(data), qs)
            if row == 1:
                axes[row, col].set_ylim(lo / pad, hi * pad)
            else:
                axes[row, col].set_ylim(lo - abs(lo) * 0.15, hi * pad)
        for row in range(3):
            _tidy(axes[row, col])

    axes[0, 0].set_ylabel(r"$|h_{\ell m}| / \nu$")
    axes[1, 0].set_ylabel(r"$\tilde{A} = |h_{\ell m}|\, e^{\tau/\tau_{\rm qnm}} / \nu$")
    axes[2, 0].set_ylabel(r"$\tilde{\phi} = \phi_{\ell m} - \phi_0 + \omega_{\rm qnm}\tau$")

    sm = plt.cm.ScalarMappable(cmap=QRAMP, norm=norm)
    fig.colorbar(sm, ax=axes, label="$q = m_1/m_2$", fraction=0.02, pad=0.01)
    fig.suptitle(
        "Every waveform: raw amplitude (top), and the two quantities the model must "
        "supply once the $n=0$ QNM is divided out",
        fontsize=11,
    )
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"Wrote {out}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="data/nr_mrd_N2.h5")
    p.add_argument("--out-prefix", default="data/dataset")
    p.add_argument("--modes", nargs="+", default=None)
    p.add_argument("--min-amp-ratio", type=float, default=1e-3)
    p.add_argument("--max-curves", type=int, default=None,
                   help="subsample to this many curves per panel; default is all of them")
    args = p.parse_args()

    ds = Dataset(args.data)
    modes = parse_modes(args.modes, cfg.DEFAULT_MODES)
    plot_parameters(ds, modes, f"{args.out_prefix}_params.png", args.min_amp_ratio)
    plot_modes(ds, modes, f"{args.out_prefix}_modes.png", args.min_amp_ratio, args.max_curves)


if __name__ == "__main__":
    main()
