"""Configuration of the merger-ringdown dataset.

The defaults reproduce the SEOBNRv5HM choices (Eq. (23) of arXiv:2303.18039 for
the attachment time, and the fitting windows of the ``MR_pipeline`` config files
of ``pyseobnr-toolkit``).  Everything here is meant to be overridden.
"""

from __future__ import annotations

#: Modes to extract into the dataset.  SEOBNRv5HM set, minus (5,5).
MODES: list[tuple[int, int]] = [(2, 2), (3, 3), (2, 1), (4, 4), (3, 2), (4, 3)]

#: Modes the analysis scripts use unless ``--modes`` says otherwise.
#:
#: (3,2) and (4,3) are left out on purpose: they are the only modes with
#: spherical-spheroidal mode mixing, so modelling them means modelling the
#: spheroidal mode *and* the (m,m) mode it mixes with, and recombining.  That
#: extra layer is worth avoiding in a first experiment.  They are still in the
#: dataset -- pass ``--modes 3,2 4,3`` (or all six) to include them.
DEFAULT_MODES: list[tuple[int, int]] = [(2, 2), (3, 3), (2, 1), (4, 4)]

#: Start of the merger-ringdown portion, relative to the peak of |h_22| [M].
#: SEOBNRv5 attaches every mode at the (2,2) peak, except (5,5) 10M earlier.
T_START: dict[tuple[int, int], float] = {
    (2, 2): 0.0,
    (3, 3): 0.0,
    (2, 1): 0.0,
    (4, 4): 0.0,
    (3, 2): 0.0,
    (4, 3): 0.0,
    (5, 5): -10.0,
}

#: Length of the merger-ringdown portion stored after ``T_START`` [M].  The same
#: for every mode: the window actually used when fitting or evaluating is a
#: separate choice, applied by truncating the stored data.
T_END: float = 100.0

#: Windows used by SEOBNRv5 to fit each mode [M], reachable with ``--t-end v5``.
#: Not the default: the mismatch is amplitude-weighted, so it saturates by ~60M
#: for every mode and a uniform 100M window costs nothing while making the
#: numbers directly comparable across modes.  Kept so the published v5 choices
#: can be reproduced.
T_END_V5: dict[tuple[int, int], float] = {
    (2, 2): 90.0,
    (3, 3): 70.0,
    (2, 1): 70.0,
    (4, 4): 40.0,
    (3, 2): 40.0,
    (4, 3): 40.0,
    (5, 5): 40.0,
}

#: Sampling of the uniform time grid the data is stored on [M].
DT: float = 0.1

#: Modes affected by spherical-spheroidal mode mixing, and the (m,m) mode they
#: mix with.  See Sec. IV C of arXiv:2303.18039.
MIXED_MODES: dict[tuple[int, int], tuple[int, int]] = {(3, 2): (2, 2), (4, 3): (3, 3)}

#: Which extrapolation order of the SXS strain to read.
EXTRAP_ORDER: int = 2

#: Subtract the displacement memory from the strain.  The current SXS format adds
#: it to every waveform; the old ``rhOverM_*`` files, and SEOBNRv5, have none.
REMOVE_MEMORY: bool = True

#: Order of the interpolating splines used to resample and differentiate.
SPLINE_K: int = 5


def mode_key(ell: int, m: int) -> str:
    """Name used for a mode inside the HDF5 dataset."""
    return f"{ell}_{m}"
