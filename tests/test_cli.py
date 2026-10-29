"""Fast tests for the command-line tool (the full regression check is `python -m namer.golden check`).

Needs the public test files: python scripts/fetch_testdata.py
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

from namer import cli, galvani_ref, mprfile
from spike2 import corpus
from spike2.galvani_snapshot import snapshot_map

EXT = corpus.EXT
pytestmark = pytest.mark.skipif(not (EXT / "gcpl.mpr").exists(), reason="run scripts/fetch_testdata.py first")


def recode(src: Path, dst: Path, old_id: int, new_id: int) -> None:
    """Simulate a vendor format change: give one column a column ID no parser knows."""
    raw = bytearray(src.read_bytes())
    i = raw.index(b"VMP data")
    m = raw.index(b"MODULE", i - 10)
    data_start = None
    for name, _ver, mdata in mprfile._modules(bytes(raw)):
        if name == "VMP data":
            data_start = bytes(raw).index(mdata)
    assert data_start is not None and m >= 0
    n_cols = raw[data_start + 4]
    ids = np.frombuffer(bytes(raw[data_start + 5: data_start + 5 + 2 * n_cols]), dtype="<u2").tolist()
    k = ids.index(old_id)
    raw[data_start + 5 + 2 * k: data_start + 7 + 2 * k] = int(new_id).to_bytes(2, "little")
    dst.write_bytes(bytes(raw))


def recode_low_byte(src: Path, dst: Path, mapping: dict[int, int]) -> None:
    """recode() for the EC-Lab 11.50 layout (data module version 10 or 11), where the full column IDs are stored
    little-endian from byte 6: only the low byte changes, so yadg, which reads the ID bytes from byte 5, sees the
    same change."""
    raw = bytearray(src.read_bytes())
    start = next(bytes(raw).index(md) for name, _v, md in mprfile._modules(bytes(raw)) if name == "VMP data")
    ids = np.frombuffer(bytes(raw[start + 6: start + 6 + 2 * raw[start + 4]]), dtype="<u2").tolist()
    for old, new in mapping.items():
        assert old >> 8 == new >> 8
        k = ids.index(old)
        raw[start + 6 + 2 * k: start + 8 + 2 * k] = int(new).to_bytes(2, "little")
    dst.write_bytes(bytes(raw))


def test_reader_split_keeps_the_known_columns_readable(tmp_path):
    """An integer-valued f8 column such as cycle number can also be read as a zero u4 followed by an f4, which moves
    every later column by four bytes. Here, with cycle number, NSR I and |I h6| under IDs no parser knows, that split
    scores 0.2 higher on the unknown columns but leaves three known columns unreadable as their own type. Among the
    splits within 0.5 of the best, the reader takes the one that leaves the most known columns readable, so every
    column decodes as in the original, and |I h6| is named from its numbered siblings."""
    f = tmp_path / "three.mpr"
    recode_low_byte(EXT / "peis.issue_149.mpr", f, {24: 40, 480: 297, 496: 298})
    m = mprfile.read(f, reference="yadg")
    ref = mprfile.read(EXT / "peis.issue_149.mpr", reference="yadg")
    assert sorted(c.raw_id for c in m.unknown) == [40, 297, 298]
    assert [c.values.tobytes() for c in m.columns] == [c.values.tobytes() for c in ref.columns]
    res = {r["column_id"]: r for r in cli.name_unknown(m)}
    assert res[298]["name"] == "|I h6|" and "elimination" in res[298]["evidence"], res[298]
    assert [i for i, r in res.items() if r["name"] == "|I h6|"] == [298], res


def test_reader_keeps_nearly_as_plausible_splits_beyond_the_first_200(tmp_path):
    """With eight column IDs unknown there are 1,498 ways to split the record. The true split here is number 1,198 in
    the search order, and two integer columns side by side (cycle number and I Range) can swap widths at almost no
    cost. The reader must have tried it: it is either the chosen split or one of the alternatives that a conclusion
    has to hold in as well."""
    f = tmp_path / "eight.mpr"
    recode_low_byte(EXT / "peis.issue_149.mpr", f, {4: 40, 77: 41, 24: 42, 39: 43, 131: 44, 13: 47, 70: 48, 473: 305})
    m = mprfile.read(f, reference="yadg")
    want = [c.values.tobytes() for c in mprfile.read(EXT / "peis.issue_149.mpr", reference="yadg").columns]
    assert len(m.unknown) == 8
    assert any([c.values.tobytes() for c in cols] == want for cols in [m.columns] + m.alternatives)


def _with_corrupt_column(src: Path, dst: Path, name: str) -> None:
    """Copy an EC-Lab 11.50-layout file with the f4 column `name` overwritten by 1e20, a value no measurement has, on
    every third row."""
    raw = bytearray(src.read_bytes())
    start = next(bytes(raw).index(md) for n, _v, md in mprfile._modules(bytes(raw)) if n == "VMP data")
    ref = mprfile.read(src, reference="yadg")
    pos, flags = 0, False
    for i in np.frombuffer(bytes(raw[start + 6: start + 6 + 2 * raw[start + 4]]), dtype="<u2").tolist():
        if i in mprfile.FLAG_IDS:  # one shared flags byte
            pos, flags = pos + (not flags), True
            continue
        c = next(c for c in ref.columns if c.raw_id == i)
        if c.name == name:
            break
        pos += c.values.dtype.itemsize
    for r in range(0, ref.n_points, 3):
        o = start + 0x3EF + r * ref.record_size + pos
        raw[o:o + 4] = np.array([1e20], "<f4").tobytes()
    dst.write_bytes(bytes(raw))


def test_reader_doubts_a_split_when_another_reads_more_known_columns(tmp_path):
    """A known column with values no storage type explains (here <Ewe>, overwritten with 1e20 on every third row) can
    be read out of the way by another split of the record. The reader keeps the most plausible split, says that it may
    be wrong, and the several-unknown rule names nothing from such a split."""
    bad, f = tmp_path / "bad.mpr", tmp_path / "doubt.mpr"
    _with_corrupt_column(EXT / "peis.issue_149.mpr", bad, "<Ewe>")
    recode_low_byte(bad, f, {4: 40, 76: 41})  # time and <I>
    m = mprfile.read(f, reference="yadg")
    assert sorted(c.raw_id for c in m.unknown) == [40, 41]
    assert m.doubtful and any("may be wrong" in n for n in m.notes), m.notes
    clean = mprfile.read(EXT / "peis.issue_149.mpr", reference="yadg")
    assert not clean.doubtful


def test_reader_matches_reference_decoder():
    for name in ("gcpl.mpr", "bcd.issue_241.mpr", "peis.mpr", "cv.mpr"):
        ref = corpus.decode(EXT / name)
        got = mprfile.read(EXT / name, reference="yadg")
        a = [(c.name, c.dtype, c.values.tobytes()) for c in got.columns]
        b = [(c.name, c.dtype, np.asarray(c.values).tobytes()) for c in ref.columns]
        assert a == b, name


def _mpt(path: Path) -> dict[str, np.ndarray]:
    lines = path.read_bytes().decode("latin-1").splitlines()
    n_header = int(lines[1].split(":")[1])
    head = lines[n_header - 1].rstrip("\t").split("\t")
    rows = [line.rstrip("\t").split("\t") for line in lines[n_header:] if line.strip()]
    return {h: np.array([float(r[i].replace(",", ".")) for r in rows]) for i, h in enumerate(head)}


def test_version_0_files_match_the_vendor_export():
    """Data module version 0 (older EC-Lab): yadg cannot read it; the values must match EC-Lab's own .mpt."""
    m = mprfile.read(EXT / "bio_logic1.mpr", reference="galvani")
    mpt = _mpt(EXT / "bio_logic1.mpt")
    cols = {c.name: c.values.astype("f8") for c in m.columns}
    assert m.n_points == len(mpt["time/s"]) == 3119 and not m.unknown
    for name, ref in (("time", "time/s"), ("Ewe", "Ewe/V"), ("dq", "dq/mA.h"), ("Pwe", "P/W")):
        np.testing.assert_allclose(cols[name], mpt[ref], rtol=1e-6, atol=1e-9, err_msg=name)
    assert any("yadg 7.0.1 cannot read" in n for n in mprfile.read(EXT / "bio_logic1.mpr", reference="yadg").notes)
    assert not any("yadg" in n for n in m.notes)


def test_empty_file_is_reported_plainly():
    with pytest.raises(ValueError, match="no data points"):
        mprfile.read(EXT / "wait.mpr")


def test_power_magnitude_in_version_0_file_is_named(tmp_path):
    """Version-0 files store |Ewe*I| as the power; hidden, it must come back as Pwe (it was named Ece)."""
    raw = bytearray((EXT / "bio_logic1.mpr").read_bytes())
    mdata = next(md for name, _v, md in mprfile._modules(bytes(raw)) if name == "VMP data")
    start = bytes(raw).index(mdata)
    ids = list(raw[start + 5: start + 5 + raw[start + 4]])
    raw[start + 5 + ids.index(70)] = 255  # one-byte column IDs in version 0
    f = tmp_path / "old_drift.mpr"
    f.write_bytes(bytes(raw))
    m = mprfile.read(f, reference="galvani")
    assert [c.raw_id for c in m.unknown] == [255]
    res = cli.name_unknown(m)
    assert res[0]["verdict"] == "NAME" and res[0]["name"] == "Pwe", res[0]


def test_real_drift_duplicates_are_reported():
    m = mprfile.read(EXT / "gcpl.issue_230.mpr", reference="galvani")
    assert [c.raw_id for c in m.unknown] == [883, 884]
    res = cli.name_unknown(m)
    assert [r["verdict"] for r in res] == ["DUPLICATE", "DUPLICATE"]


def test_galvani_low_byte_misreads_are_reported():
    """galvani 0.5.0 stops on this file only because it reads 471 (<Ece>/V) as 215; nothing is unknown."""
    m = mprfile.read(EXT / "peis.issue_225.with_ece.mpr", reference="galvani")
    assert m.unknown == []
    assert any("471 as 215" in n for n in m.notes), m.notes
    assert not any("low byte" in n for n in mprfile.read(EXT / "gcpl.mpr", reference="galvani").notes)
    assert not any("low byte" in n for n in mprfile.read(EXT / "peis.issue_225.with_ece.mpr", reference="yadg").notes)


def _galvani_version() -> str | None:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("galvani")
    except PackageNotFoundError:
        return None


def test_frozen_map_is_galvani_050():
    if _galvani_version() != "0.5.0":
        pytest.skip("needs galvani 0.5.0 (pip install '.[test]')")
    from galvani import BioLogic

    assert snapshot_map() == {int(k): (v[0], v[1]) for k, v in BioLogic.VMPdata_colID_dtype_map.items()}
    ref = galvani_ref.installed()
    assert ref.label == "galvani 0.5.0" and ref.low_byte is True


def test_full_id_galvani_gets_no_low_byte_note(monkeypatch):
    """A galvani that reads column IDs in full (datalab-org-galvani 0.6.0) misreads nothing."""
    ref = galvani_ref.Reference(snapshot_map(), "datalab-org-galvani 0.6.0", low_byte=False, installed=True)
    monkeypatch.setattr(galvani_ref, "reference", lambda: ref)
    m = mprfile.read(EXT / "peis.issue_225.with_ece.mpr", reference="galvani")
    assert m.unknown == [] and not any("low byte" in n for n in m.notes), m.notes


def test_without_galvani_the_frozen_map_stands_in(monkeypatch):
    monkeypatch.setattr(galvani_ref, "installed", lambda: None)
    m = mprfile.read(EXT / "gcpl.issue_230.mpr", reference="galvani")
    assert [c.raw_id for c in m.unknown] == [883, 884]
    assert any("not installed" in n for n in m.notes), m.notes
    assert any("galvani 0.5.0 reads only the low byte" in n and "438 as 182" in n for n in m.notes), m.notes


@pytest.mark.parametrize("old_id,expected", [(467, "Q charge or discharge"), (7, "dq"), (4, "time")])
def test_simulated_drift_is_named(tmp_path, old_id, expected):
    f = tmp_path / "drift.mpr"
    recode(EXT / "gcpl.mpr", f, old_id, 950)
    m = mprfile.read(f, reference="galvani")
    assert [c.raw_id for c in m.unknown] == [950]
    res = cli.name_unknown(m)
    assert res[0]["verdict"] == "NAME" and res[0]["name"] == expected, res[0]


def test_noise_column_is_not_named(tmp_path):
    """Replace a column's bytes with noise: the tool must abstain, not invent a name."""
    f = tmp_path / "noise.mpr"
    shutil.copy(EXT / "gcpl.mpr", f)
    recode(EXT / "gcpl.mpr", f, 467, 950)
    m = mprfile.read(f, reference="galvani")
    c = m.unknown[0]
    c.values = np.random.default_rng(0).normal(5.0, 2.0, c.values.size)
    res = cli.name_unknown(m)
    assert res[0]["verdict"] != "NAME", res[0]


def test_rules_from_what_the_file_lacks_need_one_unknown_column(tmp_path):
    """A lone unknown current in an impedance file is named I because such files always hold one. With a second
    unknown column that rule is off (the CLI passes alone=False), since the other column could be the current."""
    one, two = tmp_path / "one.mpr", tmp_path / "two.mpr"
    recode(EXT / "peis.mpr", one, 76, 950)  # <I>
    recode(one, two, 96, 951)  # and |Ece|
    m = mprfile.read(one, reference="galvani")
    assert [c.raw_id for c in m.unknown] == [950]
    res = cli.name_unknown(m)[0]
    assert res["verdict"] == "NAME" and res["name"] == "I" and "current is missing" in res["evidence"], res
    m = mprfile.read(two, reference="galvani")
    assert sorted(c.raw_id for c in m.unknown) == [950, 951]
    res = {r["column_id"]: r for r in cli.name_unknown(m)}
    assert "current is missing" not in res[950]["evidence"], res[950]
