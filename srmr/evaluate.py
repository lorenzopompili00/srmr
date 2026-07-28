"""Evaluate a standalone ringdown model against the NR dataset, mode by mode.

"Standalone" means that everything the model needs at the attachment time --
amplitude, its derivative, phase and frequency -- is read from the NR waveform,
and the remnant mass and spin are read from the NR metadata.  Only the shape of
the merger-ringdown after the attachment time is predicted.

The mode-mixed modes (3,2) and (4,3) are handled exactly as in SEOBNRv5HM: the
spheroidal ``S h_{ell m 0}`` mode and the ``(m, m)`` spherical mode are modelled
separately and recombined into the spherical ``(ell, m)`` mode, which is what is
compared against NR.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as cfg
from .dataset import Dataset
from .mismatch import amplitude_error, mismatch, phase_error
from .qnm_utils import mu, spherical_from_spheroidal
from .rd_model import Attachment, evaluate


def attachment(ds: Dataset, i: int, lm: tuple[int, int], kind: str = "spherical") -> Attachment:
    """Attachment values of simulation ``i`` for mode ``lm``.

    For ``kind='partner'`` the QNM is that of the ``(m, m)`` mode, since the
    stored series is the ``(m, m)`` spherical mode resampled on the ``lm`` grid.
    """
    row = ds.mode(lm, kind).iv.iloc[i]
    par = ds.params.iloc[i]
    qnm_lm = (lm[1], lm[1]) if kind == "partner" else lm
    return Attachment.from_nr(
        qnm_lm, par.nu, par.af, par.Mf, row.amp0, row.damp0, row.phi0, row.omega0
    )


def predict(
    ds: Dataset, i: int, lm: tuple[int, int], ansatz, t_end: float | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Model prediction for the spherical mode ``lm`` of simulation ``i``.

    Returns ``(amplitude, phase)`` on ``ds.mode(lm, t_end=t_end).tau``.
    """
    par = ds.params.iloc[i]
    tau = ds.mode(lm, t_end=t_end).tau

    if lm not in cfg.MIXED_MODES:
        att = attachment(ds, i, lm)
        return evaluate(tau, att, ansatz, ansatz.coefficients(lm, par.nu, par.chi))

    ell, m = lm
    mm = cfg.MIXED_MODES[lm]
    amp_s, phase_s = evaluate(
        tau,
        attachment(ds, i, lm, "spheroidal"),
        ansatz,
        ansatz.coefficients(lm, par.nu, par.chi),
    )
    amp_p, phase_p = evaluate(
        tau,
        attachment(ds, i, lm, "partner"),
        ansatz,
        ansatz.coefficients(mm, par.nu, par.chi),
    )
    # (m,m) spherical -> (m,m) spheroidal, then recombine into the (ell,m) spherical mode.
    s_h_mm0 = amp_p * np.exp(1j * phase_p) / np.conj(mu(m, m, m, par.af))
    s_h_lm0 = amp_s * np.exp(1j * phase_s)
    h = spherical_from_spheroidal(ell, m, par.af, s_h_lm0, s_h_mm0)
    return np.abs(h), np.unwrap(np.angle(h))


def evaluate_dataset(
    ds: Dataset,
    ansatz,
    modes: list[tuple[int, int]] | None = None,
    optimize_phase: bool = False,
    t_end: dict[tuple[int, int], float] | None = None,
) -> pd.DataFrame:
    """Mode-by-mode mismatches of a standalone ringdown model over the dataset.

    Args:
        ds: the NR dataset.
        ansatz: the ringdown ansatz, e.g. :class:`~srmr.rd_model.V5Ansatz`.
        modes: modes to evaluate; defaults to :data:`srmr.config.DEFAULT_MODES`.
        optimize_phase: maximise the overlap over a constant phase offset.
        t_end: per-mode evaluation window [M]; defaults to the whole stored one.

    Returns a long-format table with one row per (simulation, mode).
    """
    rows = []
    t_end = t_end or {}
    amp0_22 = ds.mode((2, 2)).iv.amp0.values
    for lm in modes or cfg.DEFAULT_MODES:
        nr = ds.mode(lm, t_end=t_end.get(lm))
        for i in range(len(ds)):
            amp, phase = predict(ds, i, lm, ansatz, t_end.get(lm))
            rows.append(
                dict(
                    name=ds.params.name[i],
                    ell=lm[0],
                    m=lm[1],
                    t_end=nr.tau[-1],
                    q=ds.params.q[i],
                    nu=ds.params.nu[i],
                    chi=ds.params.chi[i],
                    chi1=ds.params.chi1[i],
                    chi2=ds.params.chi2[i],
                    # amplitude of this mode relative to the (2,2) at attachment;
                    # the odd-m modes vanish for equal masses and equal spins.
                    amp_ratio=nr.iv.amp0[i] / amp0_22[i],
                    mismatch=mismatch(
                        amp, phase, nr.amp[i], nr.phase[i], ds.dt, optimize_phase
                    ),
                    amp_error=amplitude_error(amp, nr.amp[i]),
                    phase_error=phase_error(phase, nr.phase[i]),
                )
            )
    return pd.DataFrame(rows)
