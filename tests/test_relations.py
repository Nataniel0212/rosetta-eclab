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


def test_statistics_may_not_name_a_charge_without_a_current():
    """EC-Lab computes (Q-Qo) from the current; every public file with (Q-Qo) also holds I, dq or Pwe. Where the
    known columns hold none of them, the statistics vote may not name (Q-Qo) (hold-out 6: it named <I> so)."""
    from namer.engine import Context, Known
    from namer.evaluate import consistent

    t = np.arange(1500) * 2.0
    i = 5.0 + 2.0 * np.sin(t / 300.0)  # mA
    q = np.cumsum(i * np.r_[0.0, np.diff(t)]) / 3600.0  # mA·h
    no_current = {"time": Known("time", "s", t, "<f8"), "Ewe": Known("Ewe", "V", 3.0 + 0.1 * np.sin(t / 300.0), "<f4")}
    assert not consistent("(Q-Qo)", Context(None, no_current, q, "<f8"))
    with_current = {**no_current, "I": Known("I", "mA", i, "<f4")}
    assert consistent("(Q-Qo)", Context(None, with_current, q, "<f8"))


def test_missing_current_is_named_and_not_called_ece():
    """Every public impedance record (202/202) and all but one record with control in V (376/377; galvani's
    bio_logic4.mpr stores dq) hold the measured current. Where such a file names no current, a smooth column next
    to Ewe is that current, not a second potential (hold-out 7: <I> was named Ece 31 times). In an impedance record
    Ece also needs a counter-electrode impedance quantity beside it, as in all 25 public impedance records with Ece.
    Both rules need the column to be the only one missing, and a known charge column leaves I to its relations."""
    from namer.engine import Context, Known
    from namer.evaluate import consistent
    from namer.library import missing_current

    t = np.arange(600) * 0.5
    sweep = 0.2 + 0.6 * np.abs(((t / 75.0) % 2) - 1)  # V, triangular CV sweep
    current = (0.4 * np.sign(np.gradient(sweep)) + 2.0 * np.exp(-(sweep - 0.5) ** 2 / 0.01)).astype("<f4")  # mA
    cv = [col(0, "time", "s", "<f8", t), col(1, "control", "V", "<f4", sweep),
          col(2, "Ewe", "V", "<f4", sweep + 0.002 * np.sin(t))]
    a = answer(current, "<f4", cv, None, vocab.names())
    assert a["verdict"] == "NAME" and a["name"] == "I", a
    a = answer(current, "<f4", cv, None, [n for n in vocab.names() if n != "I"])  # I not allowed: no other name
    assert a["verdict"] == "ABSTAIN", a
    cv_known = {c.name: Known(c.name, c.unit, c.values, c.dtype) for c in cv}
    assert missing_current(Context(None, cv_known, current, "<f4")) == 1.0
    assert missing_current(Context(None, cv_known, current, "<f4", alone=False)) == 0.0  # another column is unknown
    noise = np.random.default_rng(1).normal(0, 1, t.size).astype("<f4")
    dq = col(3, "dq", "mA·h", "<f8", current * 0.5 / 3600)
    a = answer(noise, "<f4", cv + [dq], None, vocab.names())  # a charge column is known: no current to infer
    assert not (a["verdict"] == "NAME" and a["name"] == "I"), a

    f = np.logspace(5, -1, 61)
    w = 2 * np.pi * f
    re, im = 20 + 100 / (1 + (w * 1e-3) ** 2), 100 * w * 1e-3 / (1 + (w * 1e-3) ** 2)
    peis = [col(0, "freq", "Hz", "<f4", f), col(1, "Re(Z)", "Ohm", "<f4", re), col(2, "-Im(Z)", "Ohm", "<f4", im),
            col(3, "|Z|", "Ohm", "<f4", np.hypot(re, im)), col(4, "time", "s", "<f8", np.cumsum(1 / f + 1)),
            col(5, "Ewe", "V", "<f4", 0.31 + 0.001 * np.linspace(0, 1, 61))]
    dc = (0.02 + 0.01 * np.linspace(0, 1, 61) ** 2).astype("<f4")  # mA, slow drift of the DC current
    a = answer(dc, "<f4", peis, "PEIS", vocab.names())
    assert a["verdict"] == "NAME" and a["name"] == "I", a
    known = {c.name: Known(c.name, c.unit, c.values, c.dtype) for c in peis}
    known["I"] = Known("I", "mA", dc, "<f4")
    assert not consistent("Ece", Context("PEIS", known, dc - 0.5, "<f4"))  # no counter-electrode quantity
    known["|Ece|"] = Known("|Ece|", "V", np.full(61, 0.004), "<f4")
    assert consistent("Ece", Context("PEIS", known, dc - 0.5, "<f4"))
