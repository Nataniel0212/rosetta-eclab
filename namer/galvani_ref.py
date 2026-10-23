"""galvani as a reference parser, without depending on one particular galvani.

Two distributions install the module `galvani`: galvani (echemdata, 0.5.0 on PyPI)
and datalab-org-galvani (0.6.0 reads EC-Lab 11.50+ column IDs in full). Rosetta
does not install either, so it never overwrites the one a user already has.

reference() returns the installed galvani's own column map, and whether that
galvani reads only the low byte of each column ID in the 11.50+ layout. Without
galvani installed, the frozen 0.5.0 map (spike2/galvani_snapshot.py) stands in.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass

from spike2.galvani_snapshot import snapshot_map

LOW_BYTE_IDIOM = "column_types[1::2]"  # galvani 0.5.0 keeps every second byte of the ID array


@dataclass(frozen=True)
class Reference:
    colmap: dict[int, tuple[str, str]]
    label: str  # e.g. "galvani 0.5.0", "datalab-org-galvani 0.6.0"
    low_byte: bool | None  # True: reads only the low byte of 11.50+ column IDs; None: cannot tell
    installed: bool


def _label() -> str:
    from importlib.metadata import PackageNotFoundError, packages_distributions, version

    parts = []
    for d in sorted(set(packages_distributions().get("galvani", []))):
        try:
            parts.append(f"{d} {version(d)}")
        except PackageNotFoundError:
            parts.append(d)
    if not parts:
        return "galvani"
    if len(parts) == 1:
        return parts[0]
    return f"galvani (installed by both {' and '.join(parts)}; its files are from whichever was installed last)"


def _reads_low_byte(biologic) -> bool | None:
    try:
        src = inspect.getsource(biologic.MPRfile.__init__)
    except (OSError, TypeError, AttributeError):
        return None
    return LOW_BYTE_IDIOM in src


def installed() -> Reference | None:
    try:
        from galvani import BioLogic
    except ImportError:
        return None
    colmap = {int(k): (v[0], v[1]) for k, v in BioLogic.VMPdata_colID_dtype_map.items()}
    return Reference(colmap, _label(), _reads_low_byte(BioLogic), installed=True)


def reference() -> Reference:
    ref = installed()
    if ref is not None:
        return ref
    return Reference(snapshot_map(), "galvani 0.5.0", low_byte=True, installed=False)
