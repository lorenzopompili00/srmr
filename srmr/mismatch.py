"""Time-domain, mode-by-mode mismatch between a ringdown model and NR.

This is the flat-weighted overlap used to build the SEOBNRv5HM merger-ringdown
fits (``pyseobnr-toolkit``'s ``MR_pipeline``):

.. math::

    \\mathcal{M} = 1 - \\frac{\\left|\\int h_1 h_2^* \\, dt\\right|}
                            {\\sqrt{\\int |h_1|^2 dt \\int |h_2|^2 dt}} .

Taking the modulus of the numerator maximises over a constant phase offset.
Since a standalone ringdown model is anchored to the NR phase at the attachment
time, the *unoptimised* version (``optimize_phase=False``, which keeps only the
real part) is the more meaningful figure of merit; both are reported.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import simpson


def _overlap(h1: np.ndarray, h2: np.ndarray, dt: float) -> complex:
    return simpson(h1 * h2.conjugate(), dx=dt)


def mismatch(
    amp1: np.ndarray,
    phase1: np.ndarray,
    amp2: np.ndarray,
    phase2: np.ndarray,
    dt: float,
    optimize_phase: bool = False,
) -> float:
    """Mismatch between two modes given as amplitude and phase on the same grid.

    Samples where either waveform is not finite are dropped.

    Args:
        amp1, phase1: amplitude and unwrapped phase of the first mode.
        amp2, phase2: amplitude and unwrapped phase of the second mode.
        dt: (uniform) time step of the grid [M].
        optimize_phase: maximise over a constant phase offset.
    """
    good = np.isfinite(amp1) & np.isfinite(phase1) & np.isfinite(amp2) & np.isfinite(phase2)
    h1 = amp1[good] * np.exp(1j * phase1[good])
    h2 = amp2[good] * np.exp(1j * phase2[good])
    num = _overlap(h1, h2, dt)
    num = abs(num) if optimize_phase else num.real
    norm = np.sqrt(_overlap(h1, h1, dt).real * _overlap(h2, h2, dt).real)
    return float(1 - num / norm)


def amplitude_error(amp1: np.ndarray, amp2: np.ndarray) -> float:
    """Relative L2 error between two amplitudes."""
    good = np.isfinite(amp1) & np.isfinite(amp2)
    return float(np.linalg.norm(amp1[good] - amp2[good]) / np.linalg.norm(amp2[good]))


def phase_error(phase1: np.ndarray, phase2: np.ndarray) -> float:
    """Maximum absolute phase difference [rad], after removing the offset at t=0."""
    good = np.isfinite(phase1) & np.isfinite(phase2)
    diff = (phase1[good] - phase1[good][0]) - (phase2[good] - phase2[good][0])
    return float(np.max(np.abs(diff)))
