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


def _unknown(position, raw_id, dtype, values):
    return mprfile.Col(position, raw_id, f"unknown {raw_id}", None, dtype, False, np.asarray(values, dtype=dtype))


def _ece_file():
    """freq, time, Ewe and |Ece h2| ... |Ece h6| of an impedance file. Like the harmonics of real files, they are
    nonzero only on some rows (here one in four)."""
    f = np.logspace(5, -1, 61)
    on = np.arange(61) % 4 == 0
    known = [col(0, "freq", "Hz", "<f4", f), col(1, "time", "s", "<f8", np.cumsum(1 / f + 1)),
             col(2, "Ewe", "V", "<f4", 0.31 + 0.001 * np.linspace(0, 1, 61))]
    known += [col(3 + k, f"|Ece h{h}|", "V", "<f4", on * 1e-5 * (1 + 0.1 * np.sin(f / (h * 1e3))))
              for k, h in enumerate((2, 3, 4, 5, 6))]
    return known, on


def test_with_several_unknown_columns_the_gap_goes_to_the_one_on_its_siblings_scale():
    """Twelve public impedance files store |Ece h7| under an ID galvani does not know, next to THD, NSD and NSR Ece
    (percent) and energy columns (f8). With |Ece h2| ... |Ece h6| known, |Ece h7| is the sibling gap: it goes to the
    one unknown column with their storage type, sign and scale, and to none when a second column fits too, even a
    constant one. The other rivals here are on the right scale but f8, f4 but signed, or 20 times too large."""
    from namer import cli

    known, on = _ece_file()
    n = np.arange(61)
    h7 = _unknown(8, 503, "<f4", on * 1e-5 * (1 + 0.1 * np.cos(n)))
    rivals = [_unknown(9, 475, "<f4", 5 + np.sin(n)),  # percent
              _unknown(10, 878, "<f8", 1e-5 * (1 + 0.5 * np.sin(n / 5))),
              _unknown(11, 478, "<f4", 1e-5 * np.sin(n / 5)),
              _unknown(12, 481, "<f4", 2.2e-4 * (1 + 0.1 * np.sin(n)))]
    res = cli.name_unknown(mprfile.MprData("synthetic.mpr", "PEIS", 61, 0, known + [h7] + rivals, []))
    by_id = {r["column_id"]: r for r in res}
    assert by_id[503]["name"] == "|Ece h7|" and "elimination" in by_id[503]["evidence"], by_id[503]
    assert [r["column_id"] for r in res if r["name"] == "|Ece h7|"] == [503], res
    for second in (_unknown(13, 504, "<f4", 1.5e-5 * (1 + 0.1 * np.sin(n))), _unknown(13, 504, "<f4", np.full(61, 1.2e-5))):
        res = cli.name_unknown(mprfile.MprData("synthetic.mpr", "PEIS", 61, 0, known + [h7, second, rivals[0]], []))
        assert not any(r["name"] == "|Ece h7|" for r in res), res


def test_a_gap_needs_two_numbered_siblings_with_data_when_several_columns_are_unknown():
    """THD Ece is the sibling gap of THD Ewe, THD I, NSD Ece and NSR Ece, but those are other quantities (THD, NSD
    and NSR of one channel differ up to 94-fold in public files), so their scale cannot pick THD Ece among several
    unknown columns. Neither can one numbered sibling (|Ece h6| alone for |Ece h7|), nor one with data among siblings
    that are all zeros. A lone unknown column still gets the gap."""
    from namer.evaluate import gap_column

    f = np.logspace(5, -1, 61)
    n = np.arange(61)
    base = [col(0, "freq", "Hz", "<f4", f), col(1, "time", "s", "<f8", np.cumsum(1 / f + 1))]
    thd = base + [col(2 + k, name, "%", "<f4", (1 + k) * (1 + 0.1 * np.sin(f / ((k + 1) * 1e3)))) for k, name in
                  enumerate(("THD Ewe", "THD I", "NSD Ewe", "NSD I", "NSD Ece", "NSR Ewe", "NSR I", "NSR Ece"))]
    one = base + [col(2 + k, name, unit, "<f4", 1e-5 * (1 + 0.1 * np.sin(f / ((k + 1) * 1e3)))) for k, (name, unit) in
                  enumerate((("|Ece h6|", "V"), ("|Ewe h7|", "V"), ("|I h7|", "mA")))]
    zeros, _ = _ece_file()
    for c in zeros[3:7]:  # |Ece h2| ... |Ece h5| hold only zeros; |Ece h6| has data
        c.values = np.zeros(61, dtype="<f4")
    h7 = 1e-5 * (1 + 0.1 * np.cos(n))
    cases = ((thd, "THD Ece", 2 + 0.2 * np.cos(n), 1e-5),  # the rival is off the siblings' scale in all cases
             (one, "|Ece h7|", h7, 7.0), (zeros, "|Ece h7|", h7, 7.0))
    for known, gap, target, rival in cases:
        target = target.astype("<f4")
        a = answer(target, "<f4", known, "PEIS", vocab.names())
        assert a["name"] == gap and "elimination" in a["evidence"], a
        unknown = [_unknown(20, 900, "<f4", target), _unknown(21, 901, "<f4", np.full(61, rival))]
        assert gap_column(known, unknown, vocab.names()) is None, gap


def test_with_several_unknown_columns_the_gap_must_be_the_only_hole_in_a_numbered_family():
    """When |I h4| and |I h5| are missing as well, some unknown columns may be harmonics. A harmonic whose own column
    is off its siblings' scale would then pass its name to a sibling's column (public files with three or more
    harmonics hidden: 137 wrong names), so |Ece h7| is named only once it is the file's one numbered gap."""
    from namer.evaluate import gap_column

    known, on = _ece_file()
    n = np.arange(61)
    i_harm = {h: col(8 + k, f"|I h{h}|", "mA", "<f4", on * 1e-3 * (1 + 0.1 * np.cos(n / h)))
              for k, h in enumerate((2, 3, 4, 5, 6, 7))}
    h7 = _unknown(20, 503, "<f4", on * 1e-5 * (1 + 0.1 * np.cos(n)))
    holes = known + [i_harm[h] for h in (2, 3, 6, 7)]
    unknown = [h7] + [_unknown(21 + k, 495 + k, "<f4", i_harm[h].values) for k, h in enumerate((4, 5))]
    assert gap_column(holes, unknown, vocab.names()) is None
    whole = known + list(i_harm.values())
    g = gap_column(whole, [h7, _unknown(21, 475, "<f4", 5 + np.sin(n))], vocab.names())
    assert g is not None and g[:2] == (0, "|Ece h7|"), g
    # Five of the six |I hN| unknown: with |I h2| the only one known, the other five are holes too, so |Ece h7| is not
    # the only one even though no |I hN| has two known siblings.
    few = known + [i_harm[2]]
    unknown = [h7] + [_unknown(21 + k, 493 + k, "<f4", i_harm[h].values) for k, h in enumerate((3, 4, 5, 6, 7))]
    assert gap_column(few, unknown, vocab.names()) is None


def test_the_gap_is_not_named_when_a_nearly_as_plausible_split_lets_another_column_fit():
    """The reader keeps the splits of the record into columns that are nearly as plausible as the best one
    (mprfile.NEAR). If another unknown column fits the gap in one of them, which column is the gap's depends on where
    a column ends, so none is named. A split in which no other column fits does not stand in the way."""
    from namer.evaluate import gap_column

    known, on = _ece_file()
    n = np.arange(61)
    h7 = _unknown(8, 503, "<f4", on * 1e-5 * (1 + 0.1 * np.cos(n)))
    other = _unknown(9, 475, "<f4", 5 + np.sin(n))
    g = gap_column(known, [h7, other], vocab.names())
    assert g is not None and g[:2] == (0, "|Ece h7|"), g
    shifted = known + [_unknown(8, 503, "<u4", n % 7), _unknown(9, 475, "<f4", 1.1e-5 * (1 + 0.1 * np.sin(n)))]
    assert gap_column(known, [h7, other], vocab.names(), [shifted]) is None
    harmless = known + [_unknown(8, 503, "<u4", n % 7), _unknown(9, 475, "<f4", 5 + np.cos(n))]
    g = gap_column(known, [h7, other], vocab.names(), [harmless])
    assert g is not None and g[:2] == (0, "|Ece h7|"), g
    assert gap_column(known, [h7, other], vocab.names(), doubtful=True) is None  # the reader doubts its split


def test_siblings_that_disagree_by_more_than_a_hundredfold_set_no_scale():
    """The members of one harmonic family agree within a factor 48.5 in public files. When a misread split moves
    another column in among the siblings, their scales spread far wider, and no column is matched to them."""
    from namer.evaluate import gap_column

    known, on = _ece_file()
    n = np.arange(61)
    h7 = _unknown(8, 503, "<f4", on * 1e-5 * (1 + 0.1 * np.cos(n)))
    other = _unknown(9, 475, "<f4", 500 + np.sin(n))
    g = gap_column(known, [h7, other], vocab.names())
    assert g is not None and g[:2] == (0, "|Ece h7|"), g
    known[7].values = np.zeros(61, dtype="<f4")  # an all-zero |Ece h6| is left out; four siblings with data remain
    g = gap_column(known, [h7, other], vocab.names())
    assert g is not None and g[:2] == (0, "|Ece h7|"), g
    known[3].values = (on * 2.6 * (1 + 0.1 * np.sin(n))).astype("<f4")  # |Ece h2| now holds another column's values
    assert gap_column(known, [h7, other], vocab.names()) is None


def _peis(n=61):
    """freq, Re(Z), -Im(Z), |Z|, time and Ewe of a PEIS record (an RC element) without a counter-electrode channel."""
    f = np.logspace(5, -1, n)
    w = 2 * np.pi * f
    re, im = 20 + 100 / (1 + (w * 1e-3) ** 2), 100 * w * 1e-3 / (1 + (w * 1e-3) ** 2)
    return re, im, [col(0, "freq", "Hz", "<f4", f), col(1, "Re(Z)", "Ohm", "<f4", re), col(2, "-Im(Z)", "Ohm", "<f4", im),
                    col(3, "|Z|", "Ohm", "<f4", np.hypot(re, im)), col(4, "time", "s", "<f8", np.cumsum(1 / f + 1)),
                    col(5, "Ewe", "V", "<f4", 0.31 + 0.001 * np.linspace(0, 1, n))]


def test_ece_needs_a_counter_electrode_channel_in_impedance_records_also_with_several_unknown_columns():
    """THD, NSD and NSR (in %), harmonics and currents look like a smooth potential next to a known Ewe. In impedance
    records Ece comes with the counter-electrode impedance quantities (every public impedance record with Ece holds
    |Ece|, |Zce|, Re(Zce), -Im(Zce) and Phase(Zce)), so with none of them known Ece is ruled out unless the other
    unknown columns could hold them: five or more, three of which fit |Zce| = sqrt(Re(Zce)^2 + Im(Zce)^2) where the
    parts are not both zero. A whole family of six THD, NSD and NSR columns does not, nor do harmonics that are zero
    in nearly every row, nor a column with one huge misread value. An unknown frequency does not hide the sweep: |Z|,
    Phase(Z), Re(Z) and -Im(Z) mark an impedance record too."""
    from namer import cli
    from namer.engine import Context, Known
    from namer.library import ece_ruled_out

    re_, im, peis = _peis()
    n = np.arange(61)
    pct = [(((k + 1) * 0.4 + 0.3 * np.sin(n / (6 + k))).astype("<f4"), "<f4") for k in range(6)]  # %, smooth
    nsd = pct[0][0]
    res = cli.name_unknown(mprfile.MprData("synthetic.mpr", "PEIS", 61, 0,
                                           peis + [_unknown(6, 300, "<f4", nsd), _unknown(7, 303, "<u2", n % 5)], []))
    assert not any(r["name"] == "Ece" for r in res), res
    six = peis + [_unknown(6 + k, 300 + k, "<f4", v) for k, (v, _) in enumerate(pct)]
    res = cli.name_unknown(mprfile.MprData("synthetic.mpr", "PEIS", 61, 0, six, []))
    assert not any(r["name"] == "Ece" for r in res), res
    known = {c.name: Known(c.name, c.unit, c.values, c.dtype) for c in peis}

    def ctx(kn, others):
        return Context("PEIS", kn, nsd, "<f4", alone=False, others=others)

    zce = (0.1 * np.hypot(re_, im)).astype("<f4")
    channel = ((zce, "<f4"), ((0.1 * re_).astype("<f4"), "<f4"), ((0.1 * im).astype("<f4"), "<f4"))
    assert ece_ruled_out(ctx(known, tuple(pct[1:])))  # five others, no counter-electrode impedance among them
    assert not ece_ruled_out(ctx(known, channel + tuple(pct[1:3])))  # five others, three of them fit |Zce|
    assert ece_ruled_out(ctx(known, channel))  # too few for the whole channel
    assert ece_ruled_out(ctx(known, channel + tuple(pct[1:2])))  # four others: still too few for the five
    assert not ece_ruled_out(ctx(known, None))  # the other unknown columns not given: only a lone column
    no_freq = {k: v for k, v in known.items() if k != "freq"}
    assert ece_ruled_out(ctx(no_freq, tuple(pct[1:3])))
    re_im = {k: v for k, v in known.items() if k in ("Re(Z)", "-Im(Z)", "time", "Ewe")}
    assert ece_ruled_out(ctx(re_im, tuple(pct[1:3])))  # Re(Z) and -Im(Z) also mark an impedance record
    z_phase = {"|Z|": known["|Z|"], "Phase(Z)": Known("Phase(Z)", "deg", np.degrees(np.arctan2(-im, re_)), "<f4"),
               "time": known["time"], "Ewe": known["Ewe"]}
    assert ece_ruled_out(ctx(z_phase, tuple(pct[1:3])))  # and so do |Z| and Phase(Z) alone
    ce = [col(6, "|Zce|", "Ohm", "<f4", zce), col(7, "Re(Zce)", "Ohm", "<f4", 0.1 * re_)]
    ece = (-0.21 + 0.004 * np.linspace(0, 1, 61) ** 2).astype("<f4")  # V, slow drift
    res = cli.name_unknown(mprfile.MprData("synthetic.mpr", "PEIS", 61, 0,
                                           peis + ce + [_unknown(8, 299, "<f4", ece), _unknown(9, 303, "<u2", n % 5)], []))
    assert res[0]["verdict"] == "NAME" and res[0]["name"] == "Ece", res[0]  # the channel is known: Ece is named


def test_the_channel_check_needs_a_real_magnitude_triple():
    """Among unknown columns the counter-electrode channel shows itself as three f4 columns with |Zce| =
    sqrt(Re(Zce)^2 + Im(Zce)^2). Rows where all three are zero do not count (in one public file the harmonics are zero
    in 99 % of the rows), and one misread huge value must not widen the match."""
    from namer.library import could_hold_ce_channel

    n = 300
    k = np.arange(n)
    pct = [(((j + 1) * 0.4 + 0.3 * np.sin(k / (6 + j))).astype("<f4"), "<f4") for j in range(5)]
    re_, im = 20 + 5 * np.cos(k / 30), 3 + np.sin(k / 30)
    channel = [(np.hypot(re_, im).astype("<f4"), "<f4"), (re_.astype("<f4"), "<f4"), (im.astype("<f4"), "<f4")]
    assert could_hold_ce_channel(channel + pct[:2])
    assert not could_hold_ce_channel(pct)
    mostly_zero = []
    for h in range(3):
        x = np.zeros(n)
        x[[10, 200]] = [1e-4 * (h + 1), 3e-4 / (h + 1)]
        mostly_zero.append((x.astype("<f4"), "<f4"))
    assert not could_hold_ce_channel(mostly_zero + pct[:2])
    huge = pct[0][0].copy()
    huge[7] = 3e30
    assert not could_hold_ce_channel([(huge, "<f4")] + pct[1:])


def test_the_phase_shape_gives_way_to_the_exact_relation():
    """A phase angle swings within ±90 degrees over a sweep, and so do NSD and NSR (in %). With Re(Z) and -Im(Z)
    known the relation atan2(Im, Re) decides, so the shape names nothing: an unknown NSR column is not Phase(Z),
    while the real phase still is."""
    from namer import cli

    re_, im, peis = _peis()
    peis = [c for c in peis if c.name != "Ewe"]  # no potential, so that only the phase shape could name the NSR
    phase = np.degrees(np.arctan2(-im, re_)).astype("<f4")
    nsr = (20 + 15 * np.sin(np.arange(61) / 6)).astype("<f4")  # %
    res = cli.name_unknown(mprfile.MprData("synthetic.mpr", "PEIS", 61, 0,
                                           peis + [_unknown(5, 300, "<f4", nsr), _unknown(6, 299, "<f4", phase)], []))
    by_id = {r["column_id"]: r for r in res}
    assert by_id[300]["name"] != "Phase(Z)", by_id[300]
    assert by_id[299]["name"] == "Phase(Z)" and by_id[299]["evidence"].startswith("relation"), by_id[299]
    # With only one part known, atan2 cannot be tested, so the shape still names the real phase.
    one_part = [c for c in peis if c.name != "-Im(Z)"]
    res = cli.name_unknown(mprfile.MprData("synthetic.mpr", "PEIS", 61, 0,
                                           one_part + [_unknown(5, 299, "<f4", phase), _unknown(6, 298, "<f4", im)], []))
    by_id = {r["column_id"]: r for r in res}
    assert by_id[299]["name"] == "Phase(Z)" and by_id[299]["evidence"].startswith("shape"), by_id[299]


def test_self_check_passes_the_files_unknown_columns_to_the_ece_check():
    """--self-check hides one known column at a time while the file's unknown columns stay missing. In an impedance
    record without a counter-electrode channel and with one unknown column, a hidden smooth NSD column next to a known
    Ewe must not come back as Ece (0.1.7 named it so)."""
    from namer import cli

    _, _, peis = _peis()
    n = np.arange(61)
    nsd = col(6, "NSD Ewe", "%", "<f4", 0.8 + 0.3 * np.sin(n / 8))
    record = mprfile.MprData("synthetic.mpr", "PEIS", 61, 0, peis + [nsd, _unknown(7, 303, "<u2", n % 5)], [])
    row = next(r for r in cli.self_check(record)["rows"] if r["column"] == "NSD Ewe")
    assert row["answer"] != "Ece", row


def test_a_column_held_between_sweeps_is_not_a_potential():
    """In records with rows outside the impedance sweep, EC-Lab keeps THD, NSD and NSR at their last value there,
    while a measured potential moves. Such a column is not named Ece or Ewe; the potentials still are."""
    from namer import cli

    n = 120
    dc = np.arange(n) % 40 < 20  # 20 DC rows, then 20 sweep rows, three times
    f = np.where(dc, 0.0, np.tile(np.logspace(5, 0, 20), 6)[:n])
    t = np.cumsum(np.where(dc, 10.0, 1 / np.maximum(f, 1e-9) + 1))
    sweep_values = 0.3 + 0.1 * np.sin(np.arange(n) / 3)
    nsd = np.empty(n)
    for i in range(n):  # held at the last sweep value through each DC block
        nsd[i] = sweep_values[i] if not dc[i] else (nsd[i - 1] if i else 0.25)
    ewe = 3.4 + 0.02 * np.sin(t / 50) + 0.001 * np.cos(np.arange(n))
    ece = -0.1 + 0.01 * np.sin(t / 70)
    zre = np.where(dc, 0.0, 20 + 5 * np.cos(np.arange(n) / 4))
    zim = np.where(dc, 0.0, 3 + np.sin(np.arange(n) / 4))
    base = [col(0, "freq", "Hz", "<f4", f), col(1, "time", "s", "<f8", t), col(2, "Re(Z)", "Ohm", "<f4", zre),
            col(3, "-Im(Z)", "Ohm", "<f4", zim), col(4, "|Z|", "Ohm", "<f4", np.hypot(zre, zim)),
            col(5, "|Zce|", "Ohm", "<f4", 0.1 * np.hypot(zre, zim))]  # a counter-electrode channel: Ece is possible
    held = _unknown(10, 300, "<f4", nsd)
    res = cli.name_unknown(mprfile.MprData("synthetic.mpr", "MB", n, 0,
                                           base + [col(6, "Ewe", "V", "<f4", ewe), held, _unknown(11, 299, "<f4", ece)], []))
    by_id = {r["column_id"]: r for r in res}
    assert by_id[300]["name"] not in ("Ece", "Ewe"), by_id[300]
    assert by_id[299]["name"] == "Ece", by_id[299]
    res = cli.name_unknown(mprfile.MprData("synthetic.mpr", "MB", n, 0,
                                           base + [held, _unknown(11, 299, "<f4", ewe)], []))
    by_id = {r["column_id"]: r for r in res}
    assert by_id[300]["name"] not in ("Ece", "Ewe"), by_id[300]
    assert by_id[299]["name"] == "Ewe", by_id[299]
