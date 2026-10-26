"""Read a BioLogic EC-Lab .mpr file, including columns the reference parser does not know.

Known columns are typed and named with the reference parser's column map. For
unknown column IDs the byte width is solved from the record size, the storage
type is chosen by plausibility (finite, sensible magnitudes, integers for
integer types), and the values are decoded so the namer can identify them.
"""

from __future__ import annotations

import itertools
import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import galvani_ref

FLAG_IDS = (1, 2, 3, 21, 31, 65)
CANDIDATES = {1: ["|u1"], 2: ["<u2", "<i2"], 4: ["<f4", "<u4", "<i4"], 8: ["<f8", "<u8", "<i8"]}


@dataclass
class Col:
    position: int
    raw_id: int
    name: str
    unit: str | None
    dtype: str
    known: bool
    values: np.ndarray = field(repr=False)


@dataclass
class MprData:
    path: str
    technique: str | None
    n_points: int
    record_size: int
    columns: list[Col]
    notes: list[str]

    @property
    def unknown(self) -> list[Col]:
        return [c for c in self.columns if not c.known]


def _modules(raw: bytes):
    from yadg import dgutils
    from yadg.extractors.eclab.mpr_columns import module_header_dtypes

    for module in raw.split(b"MODULE")[1:]:
        for mhd in module_header_dtypes:
            try:
                header = dgutils.read_value(module, 0x0000, mhd)
                if len(module) == mhd.itemsize + header["length"]:
                    yield header["short_name"].strip(), header.get("newver", 0) + header["oldver"], module[mhd.itemsize:]
                    break
            except UnicodeDecodeError:
                continue


def _plausibility(v: np.ndarray, dtype: str) -> float:
    x = v.astype("f8")
    fin = np.isfinite(x)
    if fin.mean() < 0.999:
        return 0.0
    y = x[fin]
    if y.size == 0:
        return 0.0
    if dtype.endswith(("f4", "f8")):
        a = np.abs(y[y != 0])
        if a.size and (np.mean((a < 1e-30) | (a > 1e15)) > 0.01):
            return 0.05  # denormal-looking or astronomically large values: wrong type
        return 1.0 if np.mean(np.isclose(y, np.round(y))) < 0.95 else 0.7
    # integers: plausible when they are small relative to the type's range
    span = float(np.max(np.abs(y))) if y.size else 0.0
    width = np.dtype(dtype).itemsize
    return 0.9 if span < 2 ** (8 * width - 4) else 0.2


def _galvani_notes(full_ids: list[int], ref: galvani_ref.Reference, layout_11_50: bool) -> list[str]:
    """What the reference galvani makes of this file, beyond plain unknown IDs.

    galvani 0.5.0 reads only the low byte of each column ID in the 11.50+ layout.
    IDs above 255 then become other IDs (471 -> 215), which galvani either does not
    know (it stops) or knows as a different column. Its map repeats a few entries at
    the low-byte ID (211 = 467), so only IDs whose low-byte entry differs are listed.
    """
    notes = []
    if not ref.installed:
        notes.append("galvani is not installed, so 'known to galvani' means the column map of galvani 0.5.0 "
                     "(bundled with Rosetta)")
    if layout_11_50 and ref.low_byte:
        gmap = ref.colmap
        misread = [x for x in full_ids if x > 255 and gmap.get(x & 0xFF) != gmap.get(x)]
        if misread:
            pairs = ", ".join(f"{x} as {x & 0xFF}" for x in misread)
            notes.append(f"{ref.label} reads only the low byte of each column ID in this file layout, "
                         f"so it misreads {pairs}")
    return notes


def read(path: str | Path, reference: str = "galvani") -> MprData:
    from yadg.extractors.eclab.mpr import parse_columns, process_settings

    logging.getLogger("yadg").setLevel(logging.CRITICAL)
    warnings.filterwarnings("ignore")
    raw = Path(path).read_bytes()
    if not raw.startswith(b"BIO-LOGIC MODULAR FILE"):
        raise ValueError("not a BioLogic EC-Lab .mpr file")
    technique, data, version = None, None, None
    for name, ver, mdata in _modules(raw):
        if name == "VMP Set":
            try:
                technique, _, _ = process_settings(mdata, "11.50" if ver >= 10 else "10.40")
            except Exception:
                technique = None
        elif name == "VMP data":
            data, version = mdata, ver
    if data is None:
        raise ValueError("no 'VMP data' module")
    n_points = int(np.frombuffer(data, offset=0, dtype="<u4", count=1)[0])
    n_columns = int(np.frombuffer(data, offset=4, dtype="|u1", count=1)[0])
    if version in (10, 11):
        yids = np.frombuffer(data, offset=5, dtype=">u2", count=n_columns).tolist()
        full = np.frombuffer(data, offset=6, dtype="<u2", count=n_columns).tolist()
        offset = 0x3EF
    elif version in (2, 3):
        yids = full = np.frombuffer(data, offset=5, dtype="<u2", count=n_columns).tolist()
        offset = 0x195 if version == 2 else 0x196
    elif version == 0:  # older EC-Lab files; offsets as in galvani (yadg 7.0.1 does not read version 0)
        if data[5]:  # one byte per column ID; all public version-0 files use this layout
            yids = full = list(data[5:5 + n_columns])
            offset = 100
        else:  # galvani: EC-Lab 11.50+ writes [0 1 0 3 0 174 ...]; no public file to test against
            yids = full = list(data[6:5 + 2 * n_columns:2])
            offset = 1007
    else:
        raise NotImplementedError(f"data module version {version} is not supported yet")
    if n_points == 0:
        raise ValueError("the file has no data points")
    region = len(data) - offset
    if region % n_points:
        raise ValueError(f"record size is not a whole number of bytes ({region} / {n_points})")
    rec = region // n_points

    names, dtypes, units, _ = parse_columns(yids, technique)  # yadg's map: names for known columns
    gref = galvani_ref.reference() if reference == "galvani" else None
    gmap = gref.colmap if gref else {}
    notes = _galvani_notes(full, gref, version in (10, 11)) if gref else []
    if version == 0 and reference == "yadg":
        notes.append("yadg 7.0.1 cannot read this file at all (data module version 0); "
                     "'known' here means listed in yadg's column map")
    layout = []  # record layout in file order: (raw_id or None for the flags byte, ...)
    for yid, fid in zip(yids, full):
        if yid in FLAG_IDS:  # all flag IDs share one byte, at the position of the first one
            if not any(entry[0] is None for entry in layout):
                layout.append((None, "flags", None, "|u1", True))
            continue
        layout.append((fid, None, None, None, None))
    # names/dtypes for data columns, in order, from yadg's map (it skips flags the same way)
    data_names = [(n, d, u) for n, d, u in zip(names, dtypes, units) if n != "flags"]
    filled, j = [], 0
    for entry in layout:
        if entry[0] is None:
            filled.append(entry)
            continue
        n, d, u = data_names[j]
        j += 1
        fid = entry[0]
        known_yadg = not n.startswith("unknown_")
        known_ref = (fid in gmap) if reference == "galvani" else known_yadg
        if known_ref and known_yadg:
            filled.append((fid, n, u, d, True))
        elif known_ref:  # galvani knows it, yadg does not
            gname, gd = gmap[fid]
            filled.append((fid, gname.split("/")[0], None, gd, True))
        else:
            filled.append((fid, None, None, None, False))
    known_width = sum(np.dtype(d).itemsize for _, _, _, d, kn in filled if kn)
    unknown_idx = [i for i, e in enumerate(filled) if not e[4]]
    rest = rec - known_width
    best = None
    if unknown_idx:
        combos = [c for c in itertools.product((1, 2, 4, 8), repeat=len(unknown_idx)) if sum(c) == rest]
        for widths in combos[:200]:
            for types in itertools.product(*[CANDIDATES[w] for w in widths]):
                trial = list(filled)
                for i, t in zip(unknown_idx, types):
                    trial[i] = (trial[i][0], None, None, t, False)
                arr = np.frombuffer(data, offset=offset, count=n_points,
                                    dtype=np.dtype([(f"f{i}", e[3]) for i, e in enumerate(trial)]))
                score = sum(_plausibility(arr[f"f{i}"], t) for i, t in zip(unknown_idx, types))
                if best is None or score > best[0]:
                    best = (score, trial)
        if best is None:
            raise ValueError(f"cannot fit {len(unknown_idx)} unknown column(s) into {rest} unexplained bytes per record")
        filled = best[1]
        notes.append(f"{len(unknown_idx)} column ID(s) unknown to {reference}: "
                     + ", ".join(str(filled[i][0]) for i in unknown_idx) + f"; widths solved from the record size ({rec} bytes)")
    elif rest != 0:
        raise ValueError(f"record size {rec} does not match the known columns ({known_width} bytes)")
    arr = np.frombuffer(data, offset=offset, count=n_points,
                        dtype=np.dtype([(f"f{i}", e[3]) for i, e in enumerate(filled)]))
    cols, pos = [], 0
    for i, (fid, n, u, d, kn) in enumerate(filled):
        if fid is None:
            continue  # flags byte
        cols.append(Col(pos, fid, n or f"unknown {fid}", u, d, kn, np.asarray(arr[f"f{i}"])))
        pos += 1
    return MprData(str(path), technique, n_points, rec, cols, notes)
