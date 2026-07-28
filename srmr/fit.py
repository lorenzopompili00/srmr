"""Per-simulation least-squares extraction of the free coefficients of an ansatz.

This is step one of the SEOBNRv5 merger-ringdown pipeline: for each NR waveform,
fit the free coefficients of the amplitude and of the phase separately (the
attachment conditions are already built into the ansatz), and only afterwards
interpolate those coefficients across parameter space.

Comparing the mismatch obtained with the *refitted* coefficients against the one
obtained with the SEOBNRv5 parameter-space fits separates the error of the
functional form from the error of its interpolation across parameter space.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit

from .rd_model import Attachment

#: Bounds used by the SEOBNRv5 pipeline for the free coefficients.
V5_BOUNDS = {
    "c1f": (1e-4, np.inf),
    "c2f": (-np.inf, -1e-6),
    "d1f": (1e-4, np.inf),
    "d2f": (1.0, 1e3),
}


#: Used when a coefficient has no entry in ``bounds``; symbolically-regressed
#: shapes have coefficients whose sensible range is not known in advance.
_UNBOUNDED = (-np.inf, np.inf)


def _fit(model, x, y, p0, bounds, n_restarts, rng, sigma=None):
    """Least squares from ``p0``, plus ``n_restarts`` perturbed starts; best RMSE wins.

    Perturbing multiplicatively (sign preserved, then clipped into the bounds)
    keeps this independent of what the coefficients mean, so it works unchanged
    for a symbolic-regression ansatz.
    """
    lo = np.array([bounds.get(k, _UNBOUNDED)[0] for k in p0])
    hi = np.array([bounds.get(k, _UNBOUNDED)[1] for k in p0])
    start = np.clip(list(p0.values()), lo, hi)

    starts = [start] + [
        np.clip(start * np.exp(rng.normal(0.0, 0.6, size=start.size)), lo, hi)
        for _ in range(n_restarts)
    ]
    weight = 1.0 if sigma is None else 1.0 / sigma
    best, best_rmse = start, np.inf
    for s in starts:
        try:
            opt, _ = curve_fit(model, x, y, p0=s, bounds=(lo, hi), maxfev=20000, sigma=sigma)
        except (RuntimeError, ValueError):
            continue
        rmse = np.sqrt(np.mean((weight * (model(x, *opt) - y)) ** 2))
        if rmse < best_rmse:
            best, best_rmse = opt, rmse
    return dict(zip(p0, best))


def fit_coefficients(
    ansatz,
    tau: np.ndarray,
    amp_nr: np.ndarray,
    phase_nr: np.ndarray,
    att: Attachment,
    p0: dict[str, float],
    bounds: dict[str, tuple[float, float]] | None = None,
    n_restarts: int = 12,
    seed: int = 0,
    quantities: tuple[str, ...] = ("amplitude", "phase"),
    weight_phase_by_amplitude: bool = True,
) -> dict[str, float]:
    """Least-squares fit of the free coefficients of ``ansatz`` to one NR mode.

    Args:
        ansatz: e.g. :class:`~srmr.rd_model.V5Ansatz`.
        tau: time since the attachment time [M].
        amp_nr, phase_nr: the NR amplitude and unwrapped phase on ``tau``.
        att: NR attachment values of this mode.
        p0: starting values of all free coefficients.
        bounds: per-coefficient ``(low, high)``; coefficients absent from it are
            unbounded.  Defaults to :data:`V5_BOUNDS`, which leaves the
            coefficients of a symbolically-regressed shape unbounded.
        n_restarts: perturbed starting points tried in addition to ``p0``.
        seed: seed of the restart perturbations, for reproducibility.
        quantities: which of ``"amplitude"``/``"phase"`` to fit; the others keep
            their ``p0`` value.  Used to screen one shape at a time.
        weight_phase_by_amplitude: weight the phase residual by ``A_NR``, so the
            fit minimises the mismatch rather than the raw phase error.  Set
            ``False`` to reproduce the SEOBNRv5 procedure.

    Returns:
        The fitted coefficients, in the same layout as ``p0``.
    """
    bounds = bounds or V5_BOUNDS
    rng = np.random.default_rng(seed)
    good = np.isfinite(amp_nr) & np.isfinite(phase_nr)
    tau, amp_nr, phase_nr = tau[good], amp_nr[good], phase_nr[good]

    def amp_model(t, *free):
        return att.nu * ansatz.amp_tilde(t, att, *free) * np.exp(-t / att.tau_qnm)

    def phase_model(t, *free):
        return att.phi0 - att.omega_qnm * t + ansatz.phi_tilde(t, att, *free)

    args = (bounds, n_restarts, rng)
    out = dict(p0)
    if "amplitude" in quantities:
        # Unweighted L2 on the *absolute* amplitude is already the amplitude term
        # of the second-order mismatch expansion, so no sigma here.
        out |= _fit(amp_model, tau, amp_nr, {k: p0[k] for k in ansatz.amp_coeffs}, *args)
    if "phase" in quantities:
        # The phase term of that expansion is <A^2 dphi^2> / <A^2>, so the phase
        # residual must be weighted by the amplitude.  SEOBNRv5 fit the phase with
        # uniform weights, which chases late-time phase the mismatch cannot see.
        sigma = None
        if weight_phase_by_amplitude:
            floor = 1e-12 * np.max(amp_nr)
            sigma = 1.0 / np.maximum(amp_nr, floor)
        out |= _fit(
            phase_model, tau, phase_nr, {k: p0[k] for k in ansatz.phase_coeffs},
            *args, sigma=sigma,
        )
    return out
