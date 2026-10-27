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


def test_current_does_not_have_to_follow_a_potential_control():
    """In CV the control column is a potential (control_V). A current that does not follow it must not be ruled
    out as I: the relation 'follows current control' is then not testable (hold-out 6, Zenodo 3631173)."""
    from namer.engine import Context, Known
    from namer.library import LIBRARY

    t = np.arange(2000) * 0.5
    sweep = 0.9 - 0.8 * np.abs(((t / 200.0) % 2) - 1)  # V, triangular CV sweep
    current = 50 * np.gradient(sweep) / 0.5 + 20 * np.exp(-sweep / 0.05)  # mA, follows dE/dt, not E
    known = {"time": Known("time", "s", t, "<f8"), "control": Known("control", "V", sweep, "<f4"),
             "Ewe": Known("Ewe", "V", sweep + 1e-3, "<f4")}
    ctx = Context(None, known, current, "<f8")
    assert LIBRARY["I"].best_reliable_relation(ctx) is None
    known["control"] = Known("control", "mA", sweep, "<f4")  # a current control: the relation applies again
    assert LIBRARY["I"].best_reliable_relation(Context(None, known, current, "<f8")) is not None
    # and the mirror case: Ewe need not follow a current set point (control_I, galvanostatic)
    ewe_known = {"time": known["time"], "control": Known("control", "mA", np.sign(np.sin(t / 50.0)), "<f4")}
    assert LIBRARY["Ewe"].best_reliable_relation(Context(None, ewe_known, sweep + 3.0, "<f4")) is None
