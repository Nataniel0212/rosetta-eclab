"""Read a BioLogic EC-Lab .mpr file, including columns the reference parser does not know.

Known columns are typed and named with the reference parser's column map. For
unknown column IDs the byte width is solved from the record size, the storage
type is chosen by plausibility (finite, sensible magnitudes, integers for
integer types), and the values are decoded so the namer can identify them.
Every split of the unexplained bytes into widths is tried. Among the splits
nearly as plausible as the best, the one that leaves the most known columns
readable wins. The other near splits that leave as many known columns readable
are kept as alternatives, so a conclusion that depends on where one column ends
can be checked against them.
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
# Splits whose unknown columns' summed plausibility is within this of the best are nearly as likely. In public
# files a wrong split beat the right one by 0.5: an integer-valued f8 column read as a zero u4 and an f4.
NEAR = 0.5


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
    alternatives: list[list[Col]] = field(default_factory=list, repr=False)  # other splits within NEAR, no worse
    doubtful: bool = False  # another split reads more known columns as their own storage type

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


def _layouts(data: bytes, offset: int, n_points: int, rec: int, filled: list, unknown_idx: list[int], rest: int):
    """Every way to give the unknown columns byte widths that fill the record: [(score, garbled, widths, types)], in
    itertools.product order. Each unknown column gets its most plausible storage type, and the score adds up their
    plausibility. `garbled` counts the known columns that the split moves onto bytes that do not read as their storage
    type (plausibility below 0.5); the flags byte counts too, but it comes first in EC-Lab records, so no split moves
    it. A column's plausibility depends only on its own bytes, so each (byte offset, type) is scored once."""
    cache = {}

    def plaus(off: int, t: str) -> float:
        if (off, t) not in cache:
            field_dt = np.dtype({"names": ["x"], "formats": [t], "offsets": [off], "itemsize": rec})
            cache[(off, t)] = _plausibility(np.frombuffer(data, offset=offset, count=n_points, dtype=field_dt)["x"], t)
        return cache[(off, t)]

    kinds = [(e[3], np.dtype(e[3]).itemsize) if e[4] else None for e in filled]
    out = []
    for widths in itertools.product((1, 2, 4, 8), repeat=len(unknown_idx)):
        if sum(widths) != rest:
            continue
        score, garbled, types, pos, j = 0.0, 0, [], 0, 0
        for k in kinds:
            if k is None:  # unknown column: its most plausible type (the first, on a tie)
                w = widths[j]
                p, t = max(((plaus(pos, t), t) for t in CANDIDATES[w]), key=lambda x: x[0])
                score += p
                types.append(t)
                j += 1
            else:
                w = k[1]
                garbled += plaus(pos, k[0]) < 0.5
            pos += w
        out.append((round(score, 6), garbled, widths, types))
    return out


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
    elif version == 0:  # older EC-Lab files (yadg 7.0.1 does not read version 0)
        from spike2.corpus import v0_layout

        yids, offset = v0_layout(data, n_columns)
        full = yids
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
    near, doubtful = [], False
    if unknown_idx:
        layouts = _layouts(data, offset, n_points, rec, filled, unknown_idx, rest)
        if not layouts:
            raise ValueError(f"cannot fit {len(unknown_idx)} unknown column(s) into {rest} unexplained bytes per record")
        top = max(lay[0] for lay in layouts)
        close = [lay for lay in layouts if top - lay[0] <= max(NEAR, 0.0) + 1e-9]
        best = min(close, key=lambda lay: (lay[1], -lay[0]))  # fewest garbled known columns, then most plausible

        def with_types(lay) -> list:
            trial = list(filled)
            for i, t in zip(unknown_idx, lay[3]):
                trial[i] = (trial[i][0], None, None, t, False)
            return trial

        near = [with_types(lay) for lay in close if lay is not best and lay[1] <= best[1]]
        doubtful = best[1] > min(lay[1] for lay in layouts)
        filled = with_types(best)
        notes.append(f"{len(unknown_idx)} column ID(s) unknown to {reference}: "
                     + ", ".join(str(filled[i][0]) for i in unknown_idx) + f"; widths solved from the record size ({rec} bytes)")
        if doubtful:
            notes.append("another split of the record reads more of the known columns as their own storage type, "
                         "so the widths found for the unknown columns may be wrong")
    elif rest != 0:
        raise ValueError(f"record size {rec} does not match the known columns ({known_width} bytes)")

    def decode(layout: list) -> list[Col]:
        arr = np.frombuffer(data, offset=offset, count=n_points,
                            dtype=np.dtype([(f"f{i}", e[3]) for i, e in enumerate(layout)]))
        cols, pos = [], 0
        for i, (fid, n, u, d, kn) in enumerate(layout):
            if fid is None:
                continue  # flags byte
            cols.append(Col(pos, fid, n or f"unknown {fid}", u, d, kn, np.asarray(arr[f"f{i}"])))
            pos += 1
        return cols

    return MprData(str(path), technique, n_points, rec, decode(filled), notes, [decode(t) for t in near], doubtful)
