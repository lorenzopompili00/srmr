"""Small helpers shared by the command-line scripts."""

from __future__ import annotations

from . import config as cfg


def parse_mode(text: str) -> tuple[int, int]:
    """``"3,2"`` -> ``(3, 2)``."""
    ell, m = text.split(",")
    return int(ell), int(m)


def parse_modes(items: list[str] | None, default: list[tuple[int, int]]) -> list:
    return [parse_mode(s) for s in items] if items else list(default)


def parse_overrides(items: list[str] | None) -> dict[tuple[int, int], float]:
    """Parse ``--t-start 2,2=-5 3,3=0`` style options."""
    out = {}
    for item in items or []:
        mode, value = item.split("=")
        out[parse_mode(mode)] = float(value)
    return out


def parse_window(items: list[str] | None) -> dict[tuple[int, int], float]:
    """Parse the evaluation window option.

    Accepts ``full`` (the whole stored window, the default), ``v5`` (the
    SEOBNRv5 per-mode fit windows), a single number (the same for every mode),
    or per-mode assignments such as ``2,2=60 3,3=50``.
    """
    if not items or items == ["full"]:
        return {}
    if items == ["v5"]:
        return dict(cfg.T_END_V5)
    if len(items) == 1 and "=" not in items[0]:
        return {lm: float(items[0]) for lm in cfg.MODES}
    return parse_overrides(items)


WINDOW_HELP = "evaluation window: 'full' (default), 'v5', a number, or e.g. 2,2=60"
