"""Extraction of the merger-ringdown portion of SXS waveforms.

For every simulation and every requested ``(ell, m)`` mode we store, on a uniform
time grid ``tau`` starting at the attachment time,

* the amplitude ``|h_lm|`` and the unwrapped phase ``arg(h_lm)`` (kept separate,
  never a complex array),
* the "input values" at ``tau = 0``: amplitude and its first two derivatives,
  phase, frequency and its derivative,

plus the remnant mass/spin and the binary parameters read from the metadata.

Conventions
-----------
``phase`` is ``np.unwrap(np.angle(h_lm))`` as it comes out of the SXS data, so
for ``m > 0`` it *decreases* and ``omega = d(phase)/dt < 0``.  The absolute
value of the phase is arbitrary (it depends on the unwrapping branch); only
differences with respect to ``phase[0]`` are meaningful.

The peak of ``|h_22|`` is located by root-finding on the derivative of an
interpolating spline, so it is not limited by the NR sampling rate.
"""

from __future__ import annotations

import glob
import json
import os
import re
from dataclasses import dataclass, field

import h5py
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline as Spline

from . import config as cfg
from .qnm_utils import spheroidal_from_spherical

#: Padding added on both sides of the requested window before splining [M].
_PAD = 2.0


@dataclass
class ModeData:
    """Merger-ringdown data of a single ``(ell, m)`` mode of a single simulation."""

    amp: np.ndarray  #: ``|h_lm|`` on the ``tau`` grid
    phase: np.ndarray  #: unwrapped ``arg(h_lm)`` on the ``tau`` grid
    amp0: float  #: ``|h_lm|`` at ``tau = 0``
    damp0: float  #: ``d|h_lm|/dt`` at ``tau = 0``
    ddamp0: float  #: ``d^2|h_lm|/dt^2`` at ``tau = 0``
    phi0: float  #: ``arg(h_lm)`` at ``tau = 0``
    omega0: float  #: ``d arg(h_lm)/dt`` at ``tau = 0`` (negative for ``m > 0``)
    domega0: float  #: ``d^2 arg(h_lm)/dt^2`` at ``tau = 0``

    @property
    def iv(self) -> dict[str, float]:
        """The six input values as a dictionary."""
        return {
            k: getattr(self, k)
            for k in ("amp0", "damp0", "ddamp0", "phi0", "omega0", "domega0")
        }


#: The three flavours of series stored for a mode.  ``spheroidal`` and
#: ``partner`` only exist for the mode-mixed modes: ``spheroidal`` is the
#: ``S h_{ell m 0}`` mode of Eq. (60) of arXiv:2303.18039, ``partner`` is the
#: ``(m, m)`` spherical mode it mixes with, resampled on the same grid.
KINDS = ("spherical", "spheroidal", "partner")


@dataclass
class Simulation:
    """Everything extracted from one NR simulation."""

    name: str
    path: str
    m1: float
    m2: float
    chi1: float
    chi2: float
    Mf: float
    af: float
    t_peak22: float
    #: ``((ell, m), kind) -> ModeData``, with ``kind`` in :data:`KINDS`.
    data: dict[tuple[tuple[int, int], str], ModeData] = field(default_factory=dict)
    #: ``(ell, m) -> float``: last time of the NR data, relative to attachment.
    t_valid: dict[tuple[int, int], float] = field(default_factory=dict)

    def mode(self, lm: tuple[int, int], kind: str = "spherical") -> ModeData:
        return self.data[lm, kind]

    @property
    def modes(self) -> list[tuple[int, int]]:
        return [lm for (lm, kind) in self.data if kind == "spherical"]

    @property
    def q(self) -> float:
        return self.m1 / self.m2

    @property
    def nu(self) -> float:
        return self.m1 * self.m2 / (self.m1 + self.m2) ** 2

    @property
    def dm(self) -> float:
        return (self.m1 - self.m2) / (self.m1 + self.m2)

    @property
    def chi(self) -> float:
        """The spin variable used in the SEOBNRv5 merger-ringdown fits."""
        chi_s = 0.5 * (self.chi1 + self.chi2)
        chi_a = 0.5 * (self.chi1 - self.chi2)
        return chi_s + chi_a * self.dm / (1 - 2 * self.nu)


# --------------------------------------------------------------------------- #
# Reading the raw SXS data
# --------------------------------------------------------------------------- #


def read_metadata(path: str) -> dict:
    """Read the SXS ``metadata.json``, relabelling so that ``m1 >= m2``.

    Masses are normalised to ``m1 + m2 = 1``.
    """
    with open(os.path.join(path, "metadata.json")) as fp:
        meta = json.load(fp)
    m1, m2 = meta["reference_mass1"], meta["reference_mass2"]
    chi1 = meta["reference_dimensionless_spin1"][2]
    chi2 = meta["reference_dimensionless_spin2"][2]
    if m1 < m2:
        m1, m2, chi1, chi2 = m2, m1, chi2, chi1
    mtot = m1 + m2
    return dict(
        m1=m1 / mtot,
        m2=m2 / mtot,
        chi1=chi1,
        chi2=chi2,
        Mf=meta["remnant_mass"] / mtot,
        af=meta["remnant_dimensionless_spin"][2],
    )


def lower_resolution_path(path: str) -> str | None:
    """The same simulation at the highest resolution *below* the one in ``path``.

    SpEC catalogue directories end in ``LevN``.  Returns ``None`` if no lower
    level is available.
    """
    match = re.search(r"Lev(-?\d+)$", path)
    if match is None:
        return None
    lev, parent = int(match.group(1)), os.path.dirname(path)
    levels = []
    for candidate in glob.glob(os.path.join(parent, "Lev*")):
        m = re.search(r"Lev(-?\d+)$", candidate)
        if m and int(m.group(1)) < lev and os.path.exists(
            os.path.join(candidate, "rhOverM_Asymptotic_GeometricUnits_CoM.h5")
        ):
            levels.append((int(m.group(1)), candidate))
    return max(levels)[1] if levels else None


def read_modes(
    path: str, modes: list[tuple[int, int]], extrap: str = cfg.EXTRAP_DIR
) -> tuple[np.ndarray, dict[tuple[int, int], np.ndarray]]:
    """Read the requested strain modes from an SXS simulation directory."""
    fname = os.path.join(path, "rhOverM_Asymptotic_GeometricUnits_CoM.h5")
    out = {}
    with h5py.File(fname, "r") as f:
        grp = f[extrap]
        t = None
        for ell, m in modes:
            data = grp[f"Y_l{ell}_m{m}.dat"][:]
            if t is None:
                t = data[:, 0]
            out[ell, m] = data[:, 1] + 1j * data[:, 2]
    return t, out


def find_peak_time(t: np.ndarray, amp: np.ndarray, window: float = 10.0) -> float:
    """Locate the maximum of ``amp`` to sub-sample accuracy.

    A quartic spline is built in a ``window`` around the discrete maximum and the
    roots of its derivative are computed exactly; the highest one is returned.
    """
    i = int(np.argmax(amp))
    sel = np.abs(t - t[i]) < window
    spline = Spline(t[sel], amp[sel], k=4)
    roots = spline.derivative().roots()
    if roots.size == 0:
        return float(t[i])
    return float(roots[np.argmax(spline(roots))])


# --------------------------------------------------------------------------- #
# Resampling and input values
# --------------------------------------------------------------------------- #


def _sample_mode(
    t: np.ndarray, h: np.ndarray, t_match: float, tau: np.ndarray, k: int
) -> np.ndarray:
    """Resample one complex mode on the ``t_match + tau`` grid.

    Amplitude and phase are interpolated separately.  Samples beyond the end of
    the NR data are set to ``NaN``.
    """
    lo, hi = t_match + tau[0] - _PAD, t_match + tau[-1] + _PAD
    sel = (t >= lo) & (t <= hi)
    amp_spline = Spline(t[sel], np.abs(h[sel]), k=k)
    phase_spline = Spline(t[sel], np.unwrap(np.angle(h[sel])), k=k)

    times = t_match + tau
    out = np.full(tau.shape, np.nan, dtype=complex)
    good = times <= t[sel][-1]
    out[good] = amp_spline(times[good]) * np.exp(1j * phase_spline(times[good]))
    return out


def _mode_data_from_series(tau: np.ndarray, h: np.ndarray, k: int) -> ModeData:
    """Build a :class:`ModeData` from a complex mode already on the ``tau`` grid.

    The input values at ``tau = 0`` are obtained from splines of the resampled
    amplitude and phase; ``tau`` is expected to be padded on both sides so that
    ``tau = 0`` is an interior point.
    """
    good = np.isfinite(h)
    amp = np.where(good, np.abs(h), np.nan)
    phase = np.full(tau.shape, np.nan)
    phase[good] = np.unwrap(np.angle(h[good]))
    amp_spline = Spline(tau[good], amp[good], k=k)
    phase_spline = Spline(tau[good], phase[good], k=k)
    return ModeData(
        amp=amp,
        phase=phase,
        amp0=float(amp_spline(0.0)),
        damp0=float(amp_spline.derivative(1)(0.0)),
        ddamp0=float(amp_spline.derivative(2)(0.0)),
        phi0=float(phase_spline(0.0)),
        omega0=float(phase_spline.derivative(1)(0.0)),
        domega0=float(phase_spline.derivative(2)(0.0)),
    )


# --------------------------------------------------------------------------- #
# Per-simulation extraction
# --------------------------------------------------------------------------- #


def extract_simulation(
    path: str,
    name: str = "",
    modes: list[tuple[int, int]] | None = None,
    t_start: dict[tuple[int, int], float] | None = None,
    t_end: float = cfg.T_END,
    dt: float = cfg.DT,
    k: int = cfg.SPLINE_K,
    extrap: str = cfg.EXTRAP_DIR,
) -> Simulation:
    """Extract the merger-ringdown data of a single SXS simulation.

    Args:
        path: directory containing ``metadata.json`` and the strain HDF5 file.
        name: label for the simulation.
        modes: modes to extract; defaults to :data:`srmr.config.MODES`.
        t_start: per-mode start of the merger-ringdown, relative to the peak of
            ``|h_22|`` [M]; defaults to :data:`srmr.config.T_START`.
        t_end: length of the extracted portion [M].
        dt: sampling of the output grid [M].
        k: order of the interpolating splines.
        extrap: which extrapolation order to read, e.g. ``Extrapolated_N3.dir``.
    """
    modes = list(cfg.MODES if modes is None else modes)
    t_start = {**cfg.T_START, **(t_start or {})}

    # Modes we need to read: the requested ones, plus (2,2) for the peak and the
    # (m,m) partners of the mode-mixed ones.
    needed = set(modes) | {(2, 2)}
    for lm in modes:
        if lm in cfg.MIXED_MODES:
            needed.add(cfg.MIXED_MODES[lm])

    t, h = read_modes(path, sorted(needed), extrap)
    sim = Simulation(name=name, path=path, **read_metadata(path), t_peak22=0.0)
    sim.t_peak22 = find_peak_time(t, np.abs(h[2, 2]))

    n_pad = int(round(_PAD / dt))
    tau_full = (np.arange(-n_pad, round(t_end / dt) + n_pad + 1)) * dt
    for lm in modes:
        t_match = sim.t_peak22 + t_start[lm]
        h_lm = _sample_mode(t, h[lm], t_match, tau_full, k)

        sim.t_valid[lm] = float(t[-1] - t_match)
        sim.data[lm, "spherical"] = _mode_data_from_series(tau_full, h_lm, k)

        if lm in cfg.MIXED_MODES:
            mm = cfg.MIXED_MODES[lm]
            h_mm = _sample_mode(t, h[mm], t_match, tau_full, k)
            s_h = spheroidal_from_spherical(lm[0], lm[1], sim.af, h_lm, h_mm)
            sim.data[lm, "spheroidal"] = _mode_data_from_series(tau_full, s_h, k)
            sim.data[lm, "partner"] = _mode_data_from_series(tau_full, h_mm, k)

        # Trim the padding.
        for kind in KINDS:
            md = sim.data.get((lm, kind))
            if md is not None:
                md.amp = md.amp[n_pad:-n_pad]
                md.phase = md.phase[n_pad:-n_pad]

    return sim


def tau_grid(t_end: float = cfg.T_END, dt: float = cfg.DT) -> np.ndarray:
    """The uniform time grid, relative to the attachment time."""
    return np.arange(round(t_end / dt) + 1) * dt
