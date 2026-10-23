"""Fast tests for the command-line tool (the full regression check is `python -m namer.golden check`).

Needs the public test files: python scripts/fetch_testdata.py
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

from namer import cli, mprfile
from spike2 import corpus

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


def test_reader_matches_reference_decoder():
    for name in ("gcpl.mpr", "bcd.issue_241.mpr", "peis.mpr", "cv.mpr"):
        ref = corpus.decode(EXT / name)
        got = mprfile.read(EXT / name, reference="yadg")
        a = [(c.name, c.dtype, c.values.tobytes()) for c in got.columns]
        b = [(c.name, c.dtype, np.asarray(c.values).tobytes()) for c in ref.columns]
        assert a == b, name


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
