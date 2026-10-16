"""Regression tests for two cases found on files attached to parser issues (galvani #74, yadg #270)."""

from __future__ import annotations

import numpy as np
import pytest

from namer import mprfile
from namer.evaluate import answer
from spike2 import corpus, vocab

EXT = corpus.EXT


def col(position: int, name: str, unit: str | None, dtype: str, values) -> mprfile.Col:
    return mprfile.Col(position, position, name, unit, dtype, True, np.asarray(values, dtype=dtype))


def test_potential_on_a_cccv_set_point_is_not_a_current():
    """In CC-CV cycling the control column holds mA in CC steps and V in CV steps. A hidden Ewe sits on the
    potential set point through the CV hold, here most of the rows; that must not count as a current held at
    its set point."""
    rng = np.random.default_rng(0)
    n_cc, n_cv, n_dis = 200, 900, 100
    dt = 10.0
    i = np.r_[np.full(n_cc, 1.0), np.exp(-np.arange(n_cv) / 150.0), np.full(n_dis, -1.0)]  # mA
    ewe = np.r_[np.linspace(3.0, 4.2, n_cc), 4.2 + rng.normal(0, 2e-4, n_cv), np.linspace(4.2, 3.0, n_dis)]
    control = np.r_[np.full(n_cc, 1.0), np.full(n_cv, 4.2), np.full(n_dis, -1.0)]
    known = [
        col(0, "Ns", None, "<u2", np.r_[np.zeros(n_cc), np.ones(n_cv), np.full(n_dis, 2)]),
        col(1, "time", "s", "<f8", np.arange(i.size) * dt),
        col(2, "control", "V/mA", "<f4", control),
        col(3, "dq", "mA·h", "<f8", i * dt / 3600.0),
    ]
    a = answer(ewe.astype("<f4"), "<f4", known, "GCPL", vocab.names())
    assert not (a["verdict"] == "NAME" and a["name"] == "I"), a


@pytest.mark.skipif(not (EXT / "gcpl.issue_211.mpr").exists(), reason="run scripts/fetch_testdata.py first")
def test_ewe_minus_ece_is_named():
    m = mprfile.read(EXT / "gcpl.issue_211.mpr", reference="galvani")
    known = [c for c in m.columns if c.known]
    by = {vocab.canonical(c.name): c.values.astype("f8") for c in known}
    target = (by["Ewe"] - by["Ece"]).astype("<f4")
    a = answer(target, "<f4", known, m.technique, vocab.names(), exclude_group=corpus.source_group(m.path))
    assert a["verdict"] == "NAME" and a["name"] == "Ewe-Ece", a
