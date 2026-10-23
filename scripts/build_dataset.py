#!/usr/bin/env python
"""Build the merger-ringdown NR dataset from a list of SXS simulations.

Example::

    python scripts/build_dataset.py --out data/nr_mrd_N2.h5
    python scripts/build_dataset.py --extrap N3 --out data/nr_mrd_N3.h5
    python scripts/build_dataset.py --lower-resolution --out data/nr_mrd_N2_lowres.h5
    python scripts/build_dataset.py --modes 2,2 3,3 --t-start 2,2=-5 --t-end 120
"""

from __future__ import annotations

import argparse
import ast
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from srmr import config as cfg
from srmr import dataset
from srmr.cli import parse_modes, parse_overrides
from srmr.nr_dataset import extract_simulation, lower_resolution_path

DEFAULT_CSV = "/work/cfoo/aligned_spin_exploration/csv_files/all_spinning.csv"


def simulation_name(alternative_names: str) -> str:
    """Pick the ``SXS:BBH:...`` label out of the CSV's list of names."""
    try:
        names = ast.literal_eval(alternative_names)
    except (ValueError, SyntaxError):
        return str(alternative_names)
    for name in names:
        if name.startswith("SXS:BBH"):
            return name
    return names[0] if names else ""


def simulation_path_and_name(row: pd.Series) -> tuple[str, str]:
    """The ``LevN`` directory and the label of one CSV row.

    Two CSV layouts are in use: ``metadata_path`` pointing at the ``LevN``
    directory itself, with the labels in ``alternative_names``; or pointing at the
    simulation directory (named after the simulation), with the level in ``num_res``.
    """
    path = row["metadata_path"]
    if "num_res" in row:
        return os.path.join(path, f"Lev{int(row['num_res'])}"), os.path.basename(path)
    return path, simulation_name(row["alternative_names"])


def _worker(args):
    path, name, kwargs = args
    try:
        return extract_simulation(path, name, **kwargs)
    except Exception as exc:  # noqa: BLE001 - report and skip bad simulations
        print(f"  [skip] {name} ({path}): {type(exc).__name__}: {exc}", flush=True)
        return None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", default=DEFAULT_CSV, help="list of NR simulations")
    p.add_argument("--out", default="data/nr_mrd_N2.h5", help="output HDF5 file")
    p.add_argument("--modes", nargs="+", default=None, help="e.g. 2,2 3,3 (default: v5 minus 5,5)")
    p.add_argument("--t-start", nargs="+", default=None, help="e.g. 2,2=-5 [M from the h22 peak]")
    p.add_argument("--t-end", type=float, default=cfg.T_END, help="stored window length [M]")
    p.add_argument("--dt", type=float, default=cfg.DT, help="output time step [M]")
    p.add_argument("--extrap", default="N2", choices=["N2", "N3", "N4"],
                   help="extrapolation order of the SXS strain")
    p.add_argument("--keep-memory", action="store_true",
                   help="do not subtract the displacement memory (new SXS format)")
    p.add_argument("--lower-resolution", action="store_true",
                   help="use the highest resolution below the one listed in the CSV")
    p.add_argument("--nproc", type=int, default=32, help="number of worker processes")
    p.add_argument("--limit", type=int, default=None, help="only the first N simulations")
    args = p.parse_args()

    modes = parse_modes(args.modes, cfg.MODES)
    t_start = parse_overrides(args.t_start)
    extrap = int(args.extrap[1:])
    kwargs = dict(modes=modes, t_start=t_start, t_end=args.t_end, dt=args.dt, extrap=extrap,
                  remove_memory=not args.keep_memory)

    table = pd.read_csv(args.csv)
    if args.limit:
        table = table.head(args.limit)

    jobs, no_lower = [], []
    for _, row in table.iterrows():
        path, name = simulation_path_and_name(row)
        if args.lower_resolution:
            path = lower_resolution_path(path)
            if path is None:
                no_lower.append(name)
                continue
        jobs.append((path, name, kwargs))

    print(f"Extracting {len(jobs)} simulations, modes {modes}, {args.extrap}"
          f"{', one resolution down' if args.lower_resolution else ''}")
    if no_lower:
        print(f"  {len(no_lower)} simulations have no lower resolution available: {no_lower[:3]}...")

    with ProcessPoolExecutor(args.nproc) as pool:
        sims = list(tqdm(pool.map(_worker, jobs), total=len(jobs)))
    sims = [s for s in sims if s is not None]

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    dataset.save(
        sims,
        args.out,
        dt=args.dt,
        t_start=t_start,
        t_end=args.t_end,
        spline_k=cfg.SPLINE_K,
        extrap=args.extrap,
        lower_resolution=args.lower_resolution,
        remove_memory=not args.keep_memory,
        csv=args.csv,
    )
    print(f"Wrote {len(sims)} simulations to {args.out}")

    for lm in modes:
        short = [s.name for s in sims if s.t_valid[lm] < args.t_end]
        if short:
            print(f"  {lm}: {len(short)} simulations shorter than {args.t_end:g}M: {short[:5]}")


if __name__ == "__main__":
    main()
