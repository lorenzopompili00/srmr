# Symbolic regression for merger-ringdown modeling

Code for revisiting the merger-ringdown ansatz of
[SEOBNRv5HM](https://arxiv.org/abs/2303.18039) with symbolic regression: the NR dataset,
and the existing SEOBNRv5 model implemented as a **standalone** ringdown model so it can
be evaluated on the same footing as any replacement.

Standalone means everything the model needs at the attachment time — amplitude, its
derivative, phase and frequency — is read from the NR waveform, and the remnant mass and
spin from the NR metadata. No inspiral, no input-value fits, no NQCs. Only the shape of
the merger-ringdown after the attachment time is predicted, so the numbers isolate the
merger-ringdown model itself.

## Install

The NR datasets are stored with **git-lfs**, so install it before cloning:

```bash
git lfs install          # once per machine
git clone git@github.com:lorenzopompili00/srmr.git
```

If you cloned without it, `git lfs install && git lfs pull` fetches the real files.
Check: `data/nr_mrd_N2.h5` should be ~70 MB, not a ~130-byte pointer file.

Dependencies are pinned in `uv.lock`, so this reproduces the exact environment:

```bash
uv sync --extra test        # creates .venv and installs everything
source .venv/bin/activate   # or prefix commands with `uv run`
```

`uv run pytest` and `uv run python scripts/...` work without activating.
To refresh the pins after editing `pyproject.toml`: `uv lock && uv sync --extra test`.

## Use

```bash
# 1. the NR datasets (a few minutes each; already in git, see below)
python scripts/build_dataset.py --out data/nr_mrd_N2.h5 # Default
python scripts/build_dataset.py --extrap N3 --out data/nr_mrd_N3.h5 
python scripts/build_dataset.py --lower-resolution --out data/nr_mrd_N2_lowres.h5 

# 2. how good is SEOBNRv5 as a standalone ringdown model?
python scripts/eval_v5.py            # -> data/v5_mismatches.csv
python scripts/refit_v5.py           # -> data/v5_refit.csv      (the ansatz floor)
python scripts/nr_uncertainty.py     # -> data/nr_uncertainty.csv

# 3. figures
python scripts/plot_dataset.py --out-prefix data/dataset
python scripts/plot_mismatches.py --out data/mismatches.png
python scripts/plot_case.py --name SXS:BBH:0335 --out data/case_200.png

pytest
```

The three `.h5` datasets (~70 MB each, gzip-compressed) are committed via git-lfs, so
you do not need access to the source SXS simulations. `build_dataset.py` only matters
if you want to change the modes, the attachment time or the window — it reads the raw
SXS directories listed in the CSV above.

## The model

With `tau = t - t_match`, each mode is written with the `n = 0` QNM factored out:

```
|h_lm|(tau)   = nu * A~(tau) * exp(-tau / tau_qnm)
arg h_lm(tau) = phi_match - omega_qnm * tau + phi~(tau)
```

so a mode ringing down exactly at the QNM has `A~ = const`, `phi~ = 0`. A model supplies
only `A~` and `phi~`, subject to the attachment conditions

```
A~(0) = A_0/nu    A~'(0) = (A_0' + A_0/tau_qnm)/nu    phi~(0) = 0    phi~'(0) = omega_0 + omega_qnm
```

SEOBNRv5HM (`srmr.rd_model.V5Ansatz`) takes

```
A~(tau)   = c1c tanh(c1f tau + c2f) + c2c
phi~(tau) = -d1c log[(1 + d2f exp(-d1f tau)) / (1 + d2f)]
```

with `c1c, c2c, d1c` fixed by the attachment conditions and `c1f, c2f, d1f, d2f` from the
polynomial fits of Appendix D (`srmr.v5_fits`, generated mechanically from `pyseobnr`
and verified to reproduce it exactly). An alternative ansatz only has to expose the same
`amp_tilde` / `phi_tilde` interface to work with everything here.

`tests/test_rd_model.py` checks this implementation against a literal transcription of
`pyseobnr`'s `compute_MR_mode_free`.

## The dataset

591 SXS simulations, `q` from 1 to 20 and spins to |chi| = 0.998, listed in
`/work/cfoo/aligned_spin_exploration/csv_files/all_spinning.csv` (the highest resolution
`LevN` of each, given by its `num_res` column).

* The waveforms are read from the current SXS format (`LevN/Strain_N2.h5`, loaded with
  `sxs`).  Only N=2 is a file of its own; N=3 is read from the `Strain_N3.dir` group of
  `LevN/ExtraWaveforms.h5`.  The older `rhOverM_Asymptotic_GeometricUnits_CoM.h5` layout
  is still understood, so old CSVs keep working.
* The current format has the **displacement memory added**; it is subtracted again
  (`sxs.waveforms.memory.J_E` from the relaxation time, as in `pyseobnr-toolkit`'s
  `NRModel_SXS_AlignedSpin`), since neither SEOBNRv5 nor the old files include it.
  `--keep-memory` skips that.  With it subtracted, the new files reproduce the old ones
  to a (2,2) mismatch of ~4e-8, below the NR resolution error; with it kept they differ
  by ~6e-6 (`tests/test_rd_model.py` checks this on SXS:BBH:0335).

* The peak of `|h_22|` is found by root-finding on a spline derivative, so it is **not**
  quantised to the NR sampling rate (residual `d|h_22|/dt` at the recovered peak is
  `< 3e-7` across the catalogue).
* Modes `(2,2) (3,3) (2,1) (4,4) (3,2) (4,3)` — the SEOBNRv5 set minus (5,5). The
  analysis scripts default to the first four (`config.DEFAULT_MODES`); see below.
* 100 M is stored after the attachment point at `dt = 0.1 M`; analysis scripts truncate
  with `--t-end` (`--t-end v5` gives the per-mode SEOBNRv5 fit windows).
* Amplitude and unwrapped phase are stored **separately**, never as complex arrays.
* For the mode-mixed (3,2) and (4,3), the spheroidal `S h_{lm0}` mode is stored as well,
  together with the `(m,m)` mode it mixes with resampled onto the same grid — so the
  SEOBNRv5 mode-mixing treatment can be rebuilt for any attachment time.
* `--extrap` and `--lower-resolution` rebuild the same simulations at a different
  extrapolation order or resolution, if you want to check NR convergence.  23
  simulations have no lower resolution, so `nr_mrd_N2_lowres.h5` holds 568; pair the
  files by `name`, not by row.

```python
from srmr.dataset import Dataset
ds = Dataset("data/nr_mrd_N2.h5")
ds.params                                       # one row per simulation
nr = ds.mode((3, 3))                            # tau, amp, phase, attachment values
nr = ds.mode((3, 2), "spheroidal", t_end=40.0)  # mixing removed, truncated
```

Layout of the HDF5 file is documented at the top of `srmr/dataset.py`.

### Conventions worth knowing

* `phase` is `unwrap(angle(h_lm))` as it comes from SXS, so for `m > 0` it **decreases**
  and `omega = d phase/dt < 0`. Its absolute value depends on the unwrapping branch;
  only differences from `phase[:, 0]` mean anything.
* `m1 >= m2` by relabelling, masses normalised to `m1 + m2 = 1`.
* Every mode group is row-aligned with `ds.params`.

## The mismatch

Flat-weighted time-domain overlap, as used to build the v5 fits:

```
M = 1 - Re[ int h1 h2* dt ] / sqrt( int |h1|^2 dt  int |h2|^2 dt )
```

It is **not** maximised over a constant phase by default, since a standalone model is
anchored to the NR phase at the attachment time (`--optimize-phase` restores the
v5-pipeline definition, which takes the modulus of the numerator).

Modes whose amplitude at the attachment time is a negligible fraction of the (2,2) are
numerical noise, not signal — the odd-`m` modes vanish for equal masses and equal spins.
`eval_v5.py` reports both the full set and the set above `--min-amp-ratio` (default
`1e-3`); use the latter.

Median mismatch over the 591 simulations, 100 M window, modes above that cut:

| mode          | (2,2)   | (2,1)   | (3,3)   | (4,4)   |
|---------------|---------|---------|---------|---------|
| SEOBNRv5      | 1.3e-05 | 5.1e-04 | 1.4e-03 | 3.1e-03 |
| ansatz floor  | 5.2e-06 | 1.7e-04 | 1.8e-05 | 4.2e-04 |
| NR resolution | 1.3e-07 | 4.6e-06 | 6.5e-06 | 2.3e-06 |

`refit_v5.py` gives the middle row: the free coefficients refitted case by case, so it
separates the error of the *functional form* from the error of interpolating those
coefficients across parameter space. `nr_uncertainty.py` gives the bottom row by
comparing the N=2 dataset against the lower-resolution one, which is the level below
which improving the model is meaningless.

### (3,2) and (4,3)

Excluded by default. They are the only modes with spherical–spheroidal **mode mixing**,
so modelling them means modelling the spheroidal mode *and* the `(m,m)` mode it mixes
with, then recombining — an extra layer worth avoiding in a first experiment. The data
and the machinery are there; add them with `--modes`, which every analysis script takes:

```bash
python scripts/eval_v5.py --modes 2,2 3,3 2,1 4,4 3,2 4,3
```

| mode          | (3,2)   | (4,3)   |
|---------------|---------|---------|
| SEOBNRv5      | 2.3e-03 | 7.8e-03 |
| ansatz floor  | 3.2e-04 | 3.5e-04 |
| NR resolution | 1.5e-06 | 4.8e-06 |

One convention that matters there: the phase residual is fitted **weighted by the
amplitude**, which is what the mismatch asks for. SEOBNRv5 used uniform weights;
`refit_v5.py --v5-phase-weighting` reproduces that and gives a floor 1.1–2.8x worse.

## Layout

```
srmr/config.py      modes, attachment times, windows -- the free parameters
srmr/nr_dataset.py  peak finding, resampling, attachment values
srmr/dataset.py     HDF5 storage and reader
srmr/qnm_utils.py   QNM frequencies, spherical <-> spheroidal mixing
srmr/rd_model.py    the rescaled model and the SEOBNRv5 ansatz
srmr/v5_fits.py     Appendix D polynomial fits
srmr/mismatch.py    time-domain mode-by-mode mismatch
srmr/fit.py         per-case least-squares fit of an ansatz' free coefficients
srmr/evaluate.py    driver: model vs NR over the whole dataset
```

Figures in `data/`: `dataset_params.png` and `dataset_modes.png` show the data itself
(coverage, and every waveform); `mismatches.png` shows the mismatch distributions per
mode; `case_200.png` shows one simulation with the lower-resolution and N=3 extractions
overlaid, so the model error can be read against the NR uncertainty.
