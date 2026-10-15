"""Reference decoding of every corpus .mpr into named, typed columns.

The v2 benchmark isolates the semantic step: which physical quantity is this
column? Byte structure is code's job (shown 6/6 by the v1 code baseline), so
here every file is decoded with yadg 7.0.1's own column map and offsets, and
the map's names and units are the answer key. Flags are dropped; they are bit
fields, not quantities.

Usage:  python -m spike2.corpus          # prints a summary per file
"""

from __future__ import annotations

import json
import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXT = ROOT / "data" / "external"

# Real format drifts (galvani 0.5.0 cannot read them): kept out of the dev split.
DRIFT_FILES = {"gcpl.issue_230.mpr", "peis.issue_225.with_ece.mpr"}


@dataclass
class Column:
    position: int  # index in the file's record layout (flags excluded)
    raw_id: int  # full column ID (little-endian u2; for 11.50+ files at module offset 6, see docs/history.md)
    ref_id: int  # ID yadg uses for its map lookup (its own reading % 256); names come from this
    name: str
    unit: str | None
    dtype: str
    values: np.ndarray = field(repr=False)


@dataclass
class FileData:
    file: str
    technique: str | None
    version: int
    n_points: int
    columns: list[Column]
    galvani_unknown_ids: list[int]


def _modules(contents: bytes):
    """Yield (name, version, module_data) like yadg's process_modules, without converting values."""
    from yadg import dgutils
    from yadg.extractors.eclab.mpr_columns import module_header_dtypes

    for module in contents.split(b"MODULE")[1:]:
        for mhd in module_header_dtypes:
            try:
                header = dgutils.read_value(module, 0x0000, mhd)
                if len(module) == mhd.itemsize + header["length"]:
                    version = header.get("newver", 0) + header["oldver"]
                    break
            except UnicodeDecodeError:
                continue
        else:
            raise RuntimeError("unknown module header")
        yield header["short_name"].strip(), version, module[mhd.itemsize:]


def decode(path: Path) -> FileData:
    from yadg.extractors.eclab.mpr import parse_columns, process_settings

    logging.getLogger("yadg").setLevel(logging.CRITICAL)
    raw = path.read_bytes()
    technique, data, data_version = None, None, None
    for name, version, mdata in _modules(raw):
        if name == "VMP Set":
            try:
                technique, _, _ = process_settings(mdata, "11.50" if version >= 10 else "10.40")
            except Exception:  # old galvani regression files: settings layout yadg does not read
                technique = None
        elif name == "VMP data":
            data, data_version = mdata, version
    if data is None:
        raise ValueError("no data module")
    n_points = int(np.frombuffer(data, offset=0, dtype="<u4", count=1)[0])
    n_columns = int(np.frombuffer(data, offset=4, dtype="|u1", count=1)[0])
    if data_version in (10, 11):
        ids = np.frombuffer(data, offset=5, dtype=">u2", count=n_columns).tolist()
        full_ids = np.frombuffer(data, offset=6, dtype="<u2", count=n_columns).tolist()
        offset = 0x3EF
    elif data_version in (2, 3):
        ids = np.frombuffer(data, offset=5, dtype="<u2", count=n_columns).tolist()
        full_ids = list(ids)
        offset = 0x195 if data_version == 2 else 0x196
    else:
        raise NotImplementedError(f"data module version {data_version}")
    names, dtypes, units, _flags = parse_columns(ids, technique)
    arr = np.frombuffer(data, offset=offset, dtype=np.dtype(list(zip(names, dtypes))), count=n_points)
    from galvani import BioLogic

    km = {int(k): v for k, v in BioLogic.VMPdata_colID_dtype_map.items()}  # galvani 0.5.0: full IDs it knows
    cols, pos = [], 0
    # yadg's own flag test (on its reading of the IDs) decides which entries are flags
    data_ids = [(i, f) for i, f in zip(ids, full_ids) if i not in (1, 2, 3, 21, 31, 65)]
    galvani_unknown = []
    k = 0
    for name, dt, unit in zip(names, dtypes, units):
        if name == "flags":
            continue
        yid, cid = data_ids[k]
        k += 1
        if name.startswith("unknown_"):
            continue
        ref = yid % 256
        if cid not in km:  # galvani 0.5.0 looks up the full ID
            galvani_unknown.append(cid)
        cols.append(Column(position=pos, raw_id=cid, ref_id=ref, name=name, unit=unit, dtype=dt,
                           values=np.asarray(arr[name])))
        pos += 1
    return FileData(file=path.name, technique=technique, version=data_version, n_points=n_points,
                    columns=cols, galvani_unknown_ids=galvani_unknown)


# Files that come from the same experiment or contributor (same GitHub issue/PR, the VSP set,
# or sharing an identical non-constant column; checked 2026-10-14). Used for leave-one-group-out.
_GROUP_TAGS = ("issue_149", "issue_185", "issue_225", "issue_226", "pr_182", "vsp_", "bio_logic")


def source_group(name: str) -> str:
    for tag in _GROUP_TAGS:
        if tag in name:
            return tag
    return name


def corpus_files() -> list[Path]:
    return sorted(EXT.glob("*.mpr"))


def load_all() -> tuple[list[FileData], dict[str, str]]:
    out, failed = [], {}
    warnings.filterwarnings("ignore")
    for p in corpus_files():
        try:
            out.append(decode(p))
        except Exception as exc:
            failed[p.name] = f"{type(exc).__name__}: {exc}"
    return out, failed


def main() -> int:
    files, failed = load_all()
    for f in files:
        print(f"{f.file:30s} {str(f.technique):8s} v{f.version:<3d} n={f.n_points:<6d} cols={len(f.columns)} "
              f"galvani_unknown={f.galvani_unknown_ids}")
        print("    " + "; ".join(f"{c.name} [{c.unit}] {c.dtype}" for c in f.columns))
    print("\nfailed:", json.dumps(failed, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
