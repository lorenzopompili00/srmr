"""Quasi-normal-mode frequencies and spherical-spheroidal mixing coefficients.

Sign convention used throughout this package (and different from the one in
``pyseobnr``, which carries the ``qnm`` package convention around):

* ``omega_qnm > 0``  -- real part of the least-damped QNM frequency, i.e. the
  ringdown angular frequency of ``|h_lm|``'s phase.
* ``tau_qnm > 0``    -- damping time, so that ``|h_lm| ~ exp(-t / tau_qnm)``.

Both are in units of the *total* binary mass, i.e. already divided by the
remnant mass ``Mf``.
"""

from __future__ import annotations

import numpy as np
import qnm

_cache: dict[tuple[int, int], object] = {}


def qnm_omega_tau(ell: int, m: int, af: float, Mf: float) -> tuple[float, float]:
    """Real frequency and damping time of the least-damped ``(ell, m)`` QNM.

    For a retrograde (negative) remnant spin the ``qnm`` package is called with
    ``(-m, |af|)``, as in ``pyseobnr``.

    Args:
        ell: :math:`\\ell` index of the mode.
        m: :math:`m` index of the mode (positive).
        af: dimensionless spin of the remnant.
        Mf: mass of the remnant, in units of the total binary mass.

    Returns:
        ``(omega_qnm, tau_qnm)``, both positive, in units of the total mass.
    """
    key = (ell, m if af > 0 else -m)
    if key not in _cache:
        _cache[key] = qnm.modes_cache(s=-2, l=ell, m=key[1], n=0)
    omega, _, _ = _cache[key](a=abs(af), interp_only=True)
    # The qnm package returns Re(omega) > 0, Im(omega) < 0.
    return omega.real / Mf, -Mf / omega.imag


def _mu_fit(j: float, p1: float, p2: float, p3: float, p4: float) -> float:
    """Eq. (11) of arXiv:1408.1860."""
    return p1 * abs(j) ** p2 + p3 * abs(j) ** p4


#: Fitting parameters of the spherical-spheroidal mixing coefficients
#: ``mu_{m, ell, ell', n=0}``, keyed by ``(m, ell, ell')``.  Real and imaginary
#: parts, plus whether a leading ``1`` is added (diagonal) and an overall sign
#: (see footnote 4 of arXiv:1902.02731).  Taken from
#: https://pages.jh.edu/eberti2/ringdown/ via ``pyseobnr``.
_MU_FITS: dict[tuple[int, int, int], tuple[tuple, tuple, int, int]] = {
    (2, 2, 2): (
        (-7.39789e-03, 2.88933e00, -6.61369e-03, 1.71287e01),
        (1.53046e-02, 1.21928e00, -9.34293e-03, 2.49915e01),
        1,
        1,
    ),
    (2, 3, 2): (
        (-1.03512e-01, 1.22285e00, -5.74989e-02, 8.70536e00),
        (-1.60040e-02, 9.53385e-01, 1.00344e-02, 1.47550e01),
        0,
        -1,
    ),
    (2, 3, 3): (
        (-2.51626e-02, 2.43323e00, -1.30029e-02, 1.09648e01),
        (-3.23812e-02, 9.21248e-01, 1.88165e-02, 1.06988e01),
        1,
        1,
    ),
    (3, 3, 3): (
        (-1.20858e-02, 2.94266e00, -1.12402e-02, 1.72866e01),
        (-1.72988e-02, 8.51080e-01, 1.43925e-02, 5.98677e00),
        1,
        1,
    ),
    (3, 4, 3): (
        (-1.32724e-01, 1.24145e00, -7.34932e-02, 8.51670e00),
        (-1.15656e-02, 9.01101e-01, 7.32291e-03, 1.17007e01),
        0,
        -1,
    ),
    (3, 4, 4): (
        (-3.42425e-02, 2.59986e00, -2.12509e-02, 1.25368e01),
        (-4.54823e-02, 9.25823e-01, 2.84355e-02, 1.14204e01),
        1,
        1,
    ),
}


def mu(m: int, ell: int, ellp: int, af: float) -> complex:
    """Spherical-spheroidal mixing coefficient :math:`\\mu_{m \\ell \\ell' 0}`.

    Args:
        m: azimuthal index.
        ell: :math:`\\ell` of the spherical harmonic.
        ellp: :math:`\\ell'` of the spheroidal harmonic.
        af: dimensionless spin of the remnant.
    """
    re, im, diag, sign = _MU_FITS[m, ell, ellp]
    return sign * (diag + _mu_fit(af, *re) + 1j * _mu_fit(af, *im))


def spheroidal_from_spherical(
    ell: int, m: int, af: float, h_lm: np.ndarray, h_mm: np.ndarray
) -> np.ndarray:
    """Remove the mode mixing from a spherical mode.

    Inverts ``h_lm = S_h_mm0 conj(mu_{m,l,m}) + S_h_lm0 conj(mu_{m,l,l})`` with
    ``S_h_mm0 = h_mm / conj(mu_{m,m,m})``, i.e. Eq. (60) of arXiv:2303.18039.

    Args:
        ell, m: indices of the mixed spherical mode, e.g. ``(3, 2)``.
        af: dimensionless spin of the remnant.
        h_lm: the mixed spherical mode, e.g. ``h_32``.
        h_mm: the ``(m, m)`` spherical mode it mixes with, e.g. ``h_22``.

    Returns:
        The spheroidal ``S_h_{ell m 0}`` mode.
    """
    s_h_mm0 = h_mm / np.conj(mu(m, m, m, af))
    return (h_lm - s_h_mm0 * np.conj(mu(m, ell, m, af))) / np.conj(mu(m, ell, ell, af))


def spherical_from_spheroidal(
    ell: int, m: int, af: float, s_h_lm0: np.ndarray, s_h_mm0: np.ndarray
) -> np.ndarray:
    """Re-introduce the mode mixing; inverse of :func:`spheroidal_from_spherical`."""
    return s_h_mm0 * np.conj(mu(m, ell, m, af)) + s_h_lm0 * np.conj(mu(m, ell, ell, af))
