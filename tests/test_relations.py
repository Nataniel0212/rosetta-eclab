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
    """Every public impedance record (202/202) and all but one record with control in V (319/320; galvani's
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


def test_a_column_that_follows_the_time_step_is_not_the_missing_current():
    """galvani's bio_logic4.mpr (control in V, Ewe, dq, no I) stores the current only as the per-point charge
    dq = I*dt, whose size follows the time step. Hidden there, dq must not be named the missing current, while the
    current itself still is."""
    from namer.engine import Context, Known
    from namer.library import follows_time_step, missing_current

    dt = np.where(np.arange(600) % 2, 2.0, 0.2)  # s, two alternating time steps
    t = np.cumsum(dt)
    sweep = 0.2 + 0.6 * np.abs(((t / 300.0) % 2) - 1)  # V
    current = (1.5 + 0.8 * np.exp(-(sweep - 0.5) ** 2 / 0.01)).astype("<f4")  # mA
    dq = (current * dt / 3600.0).astype("<f8")  # mA·h per point
    cv = [col(0, "time", "s", "<f8", t), col(1, "control", "V", "<f4", sweep),
          col(2, "Ewe", "V", "<f4", sweep + 0.002 * np.sin(t))]
    known = {c.name: Known(c.name, c.unit, c.values, c.dtype) for c in cv}
    assert follows_time_step(Context(None, known, dq, "<f8"))
    assert missing_current(Context(None, known, dq, "<f8")) == 0.0
    a = answer(dq, "<f8", cv, None, vocab.names())
    assert not (a["verdict"] == "NAME" and a["name"] == "I"), a
    assert not follows_time_step(Context(None, known, current, "<f4"))
    a = answer(current, "<f4", cv, None, vocab.names())
    assert a["verdict"] == "NAME" and a["name"] == "I", a


def test_name_elimination_needs_the_column_to_be_the_only_one_missing():
    """With |Ewe h2| ... |Ewe h7| known except h4, a hidden column is |Ewe h4| by elimination, but only when it is
    the only column missing. With several unknown columns each of them got the gap's name (public impedance files
    with THD Ece, NSD Ece, NSR Ece and |Ece h7| unknown: all named |Ece h7|)."""
    f = np.logspace(5, -1, 61)
    known = [col(0, "freq", "Hz", "<f4", f), col(1, "time", "s", "<f8", np.cumsum(1 / f + 1)),
             col(2, "Ewe", "V", "<f4", 0.31 + 0.001 * np.linspace(0, 1, 61))]
    for k, h in enumerate((2, 3, 5, 6, 7)):
        known.append(col(3 + k, f"|Ewe h{h}|", "V", "<f4", 1e-4 / h * (1 + 0.1 * np.sin(f / (h * 1e3)))))
    target = (1e-4 / 4 * (1 + 0.1 * np.cos(np.arange(61)))).astype("<f4")
    a = answer(target, "<f4", known, "PEIS", vocab.names())
    assert a["verdict"] == "NAME" and a["name"] == "|Ewe h4|" and "elimination" in a["evidence"], a
    a = answer(target, "<f4", known, "PEIS", vocab.names(), alone=False)  # another column is unknown too
    assert "elimination" not in a["evidence"], a


def test_self_check_never_gives_the_files_own_gap():
    """--self-check hides one known column at a time. A sibling gap that the whole file already has (here |Ewe h7|,
    stored under a column ID the parser does not know) belongs to none of the known columns, so it is never their
    answer. A gap that hiding a column opens is that column's own slot and is still named by elimination, also when
    the file has an unrelated unknown column."""
    from namer import cli

    f = np.logspace(5, -1, 61)
    base = [col(0, "freq", "Hz", "<f4", f), col(1, "time", "s", "<f8", np.cumsum(1 / f + 1)),
            col(2, "Ewe", "V", "<f4", 0.31 + 0.001 * np.linspace(0, 1, 61))]

    def harmonic(h, position):
        return col(position, f"|Ewe h{h}|", "V", "<f4", 1e-4 / h * (1 + 0.1 * np.sin(f / (h * 1e3))))

    def unknown(position, raw_id, dtype, values):
        return mprfile.Col(position, raw_id, f"unknown {raw_id}", None, dtype, False, np.asarray(values, dtype=dtype))

    own_gap = base + [harmonic(h, 3 + k) for k, h in enumerate((2, 3, 4, 5, 6))]
    own_gap.append(unknown(8, 503, "<f4", 1e-4 / 7 * (1 + 0.1 * np.sin(f / 7e3))))
    rows = cli.self_check(mprfile.MprData("synthetic.mpr", "PEIS", 61, 0, own_gap, []))["rows"]
    assert not any(r["answer"] == "|Ewe h7|" for r in rows), rows
    no_gap = base + [harmonic(h, 3 + k) for k, h in enumerate((2, 3, 4, 5, 6, 7))]
    no_gap.append(unknown(9, 506, "<u2", np.arange(61) % 3))
    rows = cli.self_check(mprfile.MprData("synthetic.mpr", "PEIS", 61, 0, no_gap, []))["rows"]
    assert all(r["correct"] for r in rows if r["column"].startswith("|Ewe h")), rows
    # Two gaps tie (|Ewe h4| and |I h7|), so elimination names neither; hiding an |I h| column breaks the tie, and
    # the other gap must still not become its answer.
    tied = base + [harmonic(h, 3 + k) for k, h in enumerate((2, 3, 5, 6, 7))]
    tied.append(col(8, "I", "mA", "<f4", 0.02 + 0.001 * np.linspace(0, 1, 61)))
    tied += [col(9 + k, f"|I h{h}|", "mA", "<f4", 1e-5 / h * (1 + 0.1 * np.cos(f / (h * 1e3))))
             for k, h in enumerate((2, 3, 5, 6))]
    tied.append(unknown(13, 506, "<u2", np.arange(61) % 3))
    rows = cli.self_check(mprfile.MprData("synthetic.mpr", "PEIS", 61, 0, tied, []))["rows"]
    assert not any(r["answer"] in ("|Ewe h4|", "|I h7|") for r in rows), rows
