"""Storage and in-memory container for the merger-ringdown NR dataset.

The dataset is a single HDF5 file with the layout::

    attrs: dt, spline_k, extrap, lower_resolution, remove_memory, csv
    params/            m1, m2, chi1, chi2, Mf, af, q, nu, dm, chi, t_peak22, name
    modes/2_2/         tau (T,), amp (N, T), phase (N, T), t_valid (N,),
                       amp0, damp0, ddamp0, phi0, omega0, domega0 (N,)
    modes/3_2_spheroidal/   idem, for the mode-mixing-removed mode
    modes/3_2_partner/      idem, for the (2,2) mode on the (3,2) grid

``N`` runs over simulations (same ordering in every group) and ``T`` over the
uniform time grid ``tau``, which starts at the attachment time of that mode.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import h5py
import numpy as np
import pandas as pd

from . import config as cfg
from .nr_dataset import KINDS, Simulation, tau_grid

#: The (N, T) arrays are smooth doubles, so shuffle+gzip saves ~25% at negligible
#: read cost.  Worth it: these files are versioned with git-lfs.
_COMPRESS = dict(compression="gzip", compression_opts=4, shuffle=True)

_IV_NAMES = ("amp0", "damp0", "ddamp0", "phi0", "omega0", "domega0")
_PARAM_NAMES = ("m1", "m2", "chi1", "chi2", "Mf", "af", "q", "nu", "dm", "chi", "t_peak22")


def _group_name(lm: tuple[int, int], kind: str) -> str:
    return cfg.mode_key(*lm) + ("" if kind == "spherical" else f"_{kind}")


def save(
    sims: list[Simulation],
    filename: str,
    dt: float = cfg.DT,
    t_start: dict | None = None,
    t_end: float = cfg.T_END,
    **attrs,
) -> None:
    """Write a list of :class:`~srmr.nr_dataset.Simulation` to an HDF5 file."""
    t_start = {**cfg.T_START, **(t_start or {})}
    tau = tau_grid(t_end, dt)
    with h5py.File(filename, "w") as f:
        f.attrs["dt"] = dt
        for key, value in attrs.items():
            f.attrs[key] = value

        p = f.create_group("params")
        p.create_dataset("name", data=[s.name for s in sims])
        p.create_dataset("path", data=[s.path for s in sims])
        for key in _PARAM_NAMES:
            p.create_dataset(key, data=np.array([getattr(s, key) for s in sims]))

        g_modes = f.create_group("modes")
        for lm in sims[0].modes:
            for kind in KINDS:
                if (lm, kind) not in sims[0].data:
                    continue
                data = [s.data[lm, kind] for s in sims]
                g = g_modes.create_group(_group_name(lm, kind))
                g.create_dataset("tau", data=tau)
                g.create_dataset("amp", data=np.array([d.amp for d in data]), **_COMPRESS)
                g.create_dataset("phase", data=np.array([d.phase for d in data]), **_COMPRESS)
                g.create_dataset("t_valid", data=np.array([s.t_valid[lm] for s in sims]))
                for key in _IV_NAMES:
                    g.create_dataset(key, data=np.array([getattr(d, key) for d in data]))
                g.attrs["ell"], g.attrs["m"] = lm
                g.attrs["kind"] = kind
                g.attrs["t_start"] = t_start[lm]
                g.attrs["t_end"] = tau[-1]


@dataclass
class ModeArrays:
    """Stacked merger-ringdown data of one mode, for every simulation."""

    ell: int
    m: int
    kind: str  #: one of :data:`srmr.nr_dataset.KINDS`
    t_start: float  #: attachment time, relative to the peak of ``|h_22|`` [M]
    tau: np.ndarray  #: ``(T,)`` time grid, starting at the attachment time
    amp: np.ndarray  #: ``(N, T)``
    phase: np.ndarray  #: ``(N, T)``
    t_valid: np.ndarray  #: ``(N,)`` end of the NR data, relative to attachment
    iv: pd.DataFrame  #: ``(N, 6)`` input values at ``tau = 0``

    @property
    def lm(self) -> tuple[int, int]:
        return self.ell, self.m

    def complete(self) -> np.ndarray:
        """Boolean mask of the simulations that cover the whole ``tau`` range."""
        return np.isfinite(self.amp).all(axis=1)

    def truncate(self, t_end: float | None) -> "ModeArrays":
        """A copy restricted to ``tau <= t_end``; ``None`` keeps everything."""
        if t_end is None or t_end >= self.tau[-1]:
            return self
        n = int(np.searchsorted(self.tau, t_end, side="right"))
        return replace(self, tau=self.tau[:n], amp=self.amp[:, :n], phase=self.phase[:, :n])


class Dataset:
    """In-memory view of a merger-ringdown dataset file."""

    def __init__(self, filename: str):
        self.filename = filename
        self._modes: dict[str, ModeArrays] = {}
        with h5py.File(filename, "r") as f:
            self.attrs = dict(f.attrs)
            self.params = pd.DataFrame(
                {k: f["params"][k][:] for k in _PARAM_NAMES}
                | {
                    k: f["params"][k].asstr()[:]
                    for k in ("name", "path")
                    if k in f["params"]
                }
            )
            for name, g in f["modes"].items():
                self._modes[name] = ModeArrays(
                    ell=int(g.attrs["ell"]),
                    m=int(g.attrs["m"]),
                    kind=str(g.attrs["kind"]),
                    t_start=float(g.attrs["t_start"]),
                    tau=g["tau"][:],
                    amp=g["amp"][:],
                    phase=g["phase"][:],
                    t_valid=g["t_valid"][:],
                    iv=pd.DataFrame({k: g[k][:] for k in _IV_NAMES}),
                )

    def __len__(self) -> int:
        return len(self.params)

    @property
    def dt(self) -> float:
        return float(self.attrs["dt"])

    @property
    def modes(self) -> list[tuple[int, int]]:
        """The spherical modes present in the dataset."""
        return sorted(m.lm for m in self._modes.values() if m.kind == "spherical")

    def mode(
        self, lm: tuple[int, int], kind: str = "spherical", t_end: float | None = None
    ) -> ModeArrays:
        """Data of mode ``lm``; see :data:`srmr.nr_dataset.KINDS` for ``kind``.

        ``t_end`` truncates the stored window, which is the same for every mode.
        """
        return self._modes[_group_name(lm, kind)].truncate(t_end)

    def has(self, lm: tuple[int, int], kind: str) -> bool:
        return _group_name(lm, kind) in self._modes
