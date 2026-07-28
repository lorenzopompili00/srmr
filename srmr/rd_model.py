"""Standalone merger-ringdown waveform model.

The model is written entirely in terms of the *rescaled* amplitude and phase,
obtained by factoring the ``n = 0`` QNM out of the mode.  With
``tau = t - t_match >= 0``,

.. math::

    |h_{\\ell m}|(\\tau) = \\nu\\, \\tilde A(\\tau)\\, e^{-\\tau/\\tau_{\\rm qnm}}, \\qquad
    \\arg h_{\\ell m}(\\tau) = \\phi_{\\rm match} - \\omega_{\\rm qnm}\\tau
                              + \\tilde\\phi(\\tau) ,

so that a mode that rings down exactly at the QNM frequency has
``tilde_A = const`` and ``tilde_phi = 0``.  A model therefore only has to supply
the two functions ``tilde_A(tau)`` and ``tilde_phi(tau)``, subject to the three
constraints imposed by the NR values at the attachment time:

.. math::

    \\tilde A(0) = \\frac{A_0}{\\nu}, \\quad
    \\tilde A'(0) = \\frac{\\dot A_0 + A_0/\\tau_{\\rm qnm}}{\\nu}, \\quad
    \\tilde\\phi'(0) = \\omega_0 + \\omega_{\\rm qnm} ,

with ``tilde_phi(0) = 0``.  ``omega_0 = d arg(h)/dt < 0`` for ``m > 0``, in the
same convention as the NR data (see :mod:`srmr.nr_dataset`).

:class:`V5Ansatz` reproduces the SEOBNRv5HM model, Eqs. (48)-(53) of
arXiv:2303.18039.  Any replacement ansatz only needs the same interface.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .qnm_utils import qnm_omega_tau
from .v5_fits import FREE_COEFFS


@dataclass
class Attachment:
    """Everything a standalone ringdown model needs at the attachment time.

    All quantities are read from NR (or from the remnant metadata); nothing here
    comes from an inspiral model.
    """

    lm: tuple[int, int]  #: the mode these values belong to
    nu: float  #: symmetric mass ratio
    amp0: float  #: ``|h_lm|`` at the attachment time
    damp0: float  #: ``d|h_lm|/dt`` at the attachment time
    phi0: float  #: ``arg(h_lm)`` at the attachment time
    omega0: float  #: ``d arg(h_lm)/dt`` at the attachment time (negative for ``m > 0``)
    omega_qnm: float  #: real part of the ``(ell, m, 0)`` QNM frequency (positive)
    tau_qnm: float  #: damping time of the ``(ell, m, 0)`` QNM (positive)

    @classmethod
    def from_nr(
        cls,
        lm: tuple[int, int],
        nu: float,
        af: float,
        Mf: float,
        amp0: float,
        damp0: float,
        phi0: float,
        omega0: float,
    ) -> "Attachment":
        """Build from NR input values plus the remnant mass and spin."""
        omega_qnm, tau_qnm = qnm_omega_tau(lm[0], lm[1], af, Mf)
        return cls(lm, nu, amp0, damp0, phi0, omega0, omega_qnm, tau_qnm)

    @property
    def amp_tilde0(self) -> float:
        """``tilde_A(0)``."""
        return self.amp0 / self.nu

    @property
    def damp_tilde0(self) -> float:
        """``tilde_A'(0)``."""
        return (self.damp0 + self.amp0 / self.tau_qnm) / self.nu

    @property
    def dphi_tilde0(self) -> float:
        """``tilde_phi'(0)``."""
        return self.omega0 + self.omega_qnm


class V5Ansatz:
    """The SEOBNRv5HM merger-ringdown ansatz.

    ``tilde_A(tau)  = c1c tanh(c1f tau + c2f) + c2c``

    ``tilde_phi(tau) = -d1c log[(1 + d2f exp(-d1f tau)) / (1 + d2f)]``

    ``c1c``, ``c2c`` and ``d1c`` are fixed by the attachment conditions; the free
    coefficients ``c1f, c2f, d1f, d2f`` come from the fits of Appendix D of
    arXiv:2303.18039.

    Args:
        enforce_peak: apply the SEOBNRv5 safeguard that forces the (2,2)
            amplitude to have a maximum at the attachment time.  As in
            ``pyseobnr``, it is applied to the (2,2) mode only.
    """

    amp_coeffs = ("c1f", "c2f")
    phase_coeffs = ("d1f", "d2f")

    def __init__(self, enforce_peak: bool = True):
        self.enforce_peak = enforce_peak

    def coefficients(self, lm: tuple[int, int], nu, chi) -> dict[str, float]:
        """The four free coefficients of mode ``lm`` from the SEOBNRv5 fits."""
        return {name: float(fits[lm](nu, chi)) for name, fits in FREE_COEFFS.items()}

    def amp_tilde(self, tau, att: Attachment, c1f: float, c2f: float) -> np.ndarray:
        """Rescaled amplitude ``tilde_A(tau)``."""
        if self.enforce_peak and att.lm == (2, 2):
            # 2 c1f |tanh(c2f)| <= 1/tau_qnm  <=>  A''(0) <= 0 when A'(0) = 0.
            c1f = min(c1f, -1.0 / (2 * att.tau_qnm * np.tanh(c2f)))
        c1c = att.damp_tilde0 * np.cosh(c2f) ** 2 / c1f
        c2c = att.amp_tilde0 - c1c * np.tanh(c2f)
        return c1c * np.tanh(c1f * tau + c2f) + c2c

    def phi_tilde(self, tau, att: Attachment, d1f: float, d2f: float) -> np.ndarray:
        """Rescaled phase ``tilde_phi(tau)``."""
        d1c = att.dphi_tilde0 * (1 + d2f) / (d1f * d2f)
        return -d1c * np.log((1 + d2f * np.exp(-d1f * tau)) / (1 + d2f))


def evaluate(
    tau: np.ndarray, att: Attachment, ansatz, coeffs: dict[str, float]
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate a ringdown model, returning ``(amplitude, phase)``.

    Args:
        tau: time since the attachment time [M].
        att: the NR attachment values.
        ansatz: object providing ``amp_tilde``/``phi_tilde`` and the names of
            their free coefficients, e.g. :class:`V5Ansatz`.
        coeffs: values of the free coefficients.
    """
    amp_t = ansatz.amp_tilde(tau, att, *(coeffs[k] for k in ansatz.amp_coeffs))
    phi_t = ansatz.phi_tilde(tau, att, *(coeffs[k] for k in ansatz.phase_coeffs))
    amp = att.nu * amp_t * np.exp(-tau / att.tau_qnm)
    phase = att.phi0 - att.omega_qnm * tau + phi_t
    return amp, phase
