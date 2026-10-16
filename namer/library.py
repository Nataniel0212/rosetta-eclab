"""Formula library: what code knows about each EC-Lab quantity.

Each quantity has relation checks (its defining relation to other columns,
with the expected scale factors for the units the vendor uses) and/or a shape
check. Relations are written from the definitions of the quantities (see
docs/definitions.md for sources); scale factors cover the unit variants seen in
parsers and vendor exports (mA·h vs C, W·h, µF, degrees).

Conventions (EC-Lab): time in s, current in mA, potential in V, charge in mA·h
(or C), energy in W·h, power in W, impedance in Ω, frequency in Hz, phase in
degrees, capacitance in µF, impedance amplitudes |I| in A.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .engine import Context, Evidence, fit, scale_match

MAH_PER_MA_S = 1 / 3600.0  # mA * s -> mA·h
C_PER_MA_S = 1 / 1000.0  # mA * s -> C
WH_PER_V_MA_S = 1 / 3.6e6  # V * mA * s -> W·h
W_PER_V_MA = 1 / 1000.0  # V * mA -> W


def d0(x):
    return np.concatenate([[0.0], np.diff(x)])


def cumsum_by(x, key):
    x = np.nan_to_num(np.asarray(x, dtype="f8"))
    key = np.asarray(key)
    out = np.empty_like(x)
    acc = 0.0
    for i in range(len(x)):
        if i and key[i] != key[i - 1]:
            acc = 0.0
        acc += x[i]
        out[i] = acc
    return out


def since_change(t, key):
    t = np.asarray(t, dtype="f8")
    key = np.asarray(key)
    out = np.empty_like(t)
    start = t[0] if len(t) else 0.0
    for i in range(len(t)):
        if i and key[i] != key[i - 1]:
            start = t[i]
        out[i] = t[i] - start
    return out


def sign_changes_count(i):
    """Cumulative count of sign changes of a current (zeros keep the previous sign)."""
    s = np.sign(np.asarray(i, dtype="f8"))
    for k in range(1, len(s)):
        if s[k] == 0:
            s[k] = s[k - 1]
    return np.concatenate([[0], np.cumsum(s[1:] * s[:-1] < 0)])


@dataclass
class Relation:
    detail: str
    needs: tuple  # required known columns; an inner tuple means "any of these"
    pred: Callable[[Context], np.ndarray | None]
    scales: list[float] = field(default_factory=list)  # expected |a| values (empty: any scale)
    signed: bool = False  # True: sign of a matters (scales given with sign)
    gate: Callable[[Context], float] | None = None  # multiplies this relation's score
    rows: str = "auto"  # "dc": rows without an impedance sweep, "eis": sweep rows, "all"; auto = by quantity family
    abs_target: bool = False  # fit |target| (e.g. when the only current available is |P/E|)
    reliable: bool = True  # an exact definition: if testable and it fails (with the right units), the name is ruled out


@dataclass
class Shape:
    detail: str
    test: Callable[[Context], float]  # 0..1
    intrinsic: bool = True  # a property of the column alone (False: depends on other columns)


@dataclass
class Quantity:
    name: str
    relations: list[Relation] = field(default_factory=list)
    shapes: list[Shape] = field(default_factory=list)
    prior: float = 0.0  # weak score when nothing else applies
    gate: Callable[[Context], float] | None = None  # multiplies the final score (a property the quantity must have)

    def best_reliable_relation(self, ctx: Context) -> Evidence | None:
        """Best evidence from exact (reliable) relations only; None if none is testable here."""
        saved = (self.shapes, self.relations)
        try:
            self.shapes, self.relations = [], [r for r in self.relations if r.reliable]
            ev = self.evaluate(ctx)
        finally:
            self.shapes, self.relations = saved
        return ev if ev is not None and ev.kind == "relation" else None

    def evaluate(self, ctx: Context) -> Evidence | None:
        best: Evidence | None = None
        for rel in self.relations:
            if not all(any(n in ctx.known for n in (need if isinstance(need, tuple) else (need,))) for need in rel.needs):
                continue
            with np.errstate(all="ignore"):
                try:
                    p = rel.pred(ctx)
                except Exception:
                    p = None
            if p is None:
                continue
            mask = row_mask(ctx, rel.rows if rel.rows != "auto" else ("eis" if self.name in EIS_FAMILY else "dc"))
            tgt = np.asarray(np.abs(np.asarray(ctx.target, dtype="f8")) if rel.abs_target else ctx.target)
            fits = [fit(tgt, p, rel.reliable)] + ([fit(tgt[mask], np.asarray(p)[mask], rel.reliable)] if mask is not None else [])
            fits = [f for f in fits if f is not None]
            if not fits:
                continue
            f = max(fits, key=lambda f: f[0])
            r2, a, _ = f
            sm = scale_match(a if rel.signed else abs(a), rel.scales)
            score = max(0.0, r2) ** 4 * (1.0 if sm is None else 0.25 + 0.75 * sm)
            if sm is None:
                score *= 0.9  # an unscaled relation is weaker evidence
            if rel.gate is not None:
                score *= float(rel.gate(ctx))
            if not rel.reliable:
                score = min(score, 0.7)  # an approximate definition is never decisive on its own
            if self.name in CONTINUOUS and _integer_valued(ctx):
                score *= 0.3  # charges, energies and potentials are not exact integers
            ev = Evidence(self.name, score, "relation", f"{rel.detail}: r2 {r2:.4f}, scale {a:.4g}",
                          {"r2": r2, "reliable": rel.reliable, "scale_match": 1.0 if sm is None else sm})
            if best is None or ev.score > best.score:
                best = ev
        for sh in self.shapes:
            s = float(sh.test(ctx))
            ev = Evidence(self.name, s, "shape", sh.detail)
            if best is None or ev.score > best.score:
                best = ev
        if best is None and self.prior > 0:
            best = Evidence(self.name, self.prior, "prior", "no relation testable; prior only")
        if best is not None and self.gate is not None:
            best.score *= float(self.gate(ctx))
        if best is not None and self.name in COUNTERS and not _all_integers(ctx):
            best.score *= 0.05  # counters and codes are whole numbers by definition
        if best is not None and self.name in EIS_FAMILY and ctx.col("freq") is None and self.name != "freq":
            best.score *= 0.2  # impedance quantities need a frequency sweep in the file
        exp = DTYPES.get(self.name)
        if best is not None and exp and ctx.target_dtype not in exp:
            best.score *= 0.3
            best.detail += f" (stored as {ctx.target_dtype}, expected {'/'.join(sorted(exp))})"
        return best


EIS_FAMILY = {"freq", "|Z|", "Re(Z)", "-Im(Z)", "Phase(Z)", "Cs", "Cp", "Cs⁻²", "Cp⁻²", "|Ewe|", "|I|", "|Ece|",
              "|Zce|", "Re(Zce)", "-Im(Zce)", "Phase(Zce)", "|Zwe-ce|", "Re(Zwe-ce)", "-Im(Zwe-ce)", "Phase(Zwe-ce)",
              "THD Ewe", "THD I", "THD Ece", "NSD Ewe", "NSD I", "NSD Ece", "NSR Ewe", "NSR I", "NSR Ece", "z cycle",
              "Re(Y)", "Im(Y)", "|Y|", "Phase(Y)", "|Z1|", "|Z2|", "Phase(Z1)", "-Im(Z1)", "-Im(Z2)", "|Estack|",
              "|Zstack|", "Phase(Zstack)"}


CONTINUOUS = {"dq", "dQ", "(Q-Qo)", "Q charge or discharge", "Energy we charge", "Energy we discharge", "|Energy|",
              "Energy we", "Pwe", "Ewe", "I", "Capacitance charge", "Capacitance discharge", "x"}


COUNTERS = {"half cycle", "cycle number", "z cycle", "Ns", "I Range", "counter inc."}


def _all_integers(ctx) -> bool:
    y = np.asarray(ctx.target, dtype="f8")
    y = y[np.isfinite(y)]
    return y.size > 0 and bool(np.allclose(y, np.round(y)))


def _integer_valued(ctx) -> bool:
    y = np.asarray(ctx.target, dtype="f8")
    y = y[np.isfinite(y)]
    return y.size > 3 and np.ptp(y) > 0 and bool(np.allclose(y, np.round(y)))


def row_mask(ctx, which: str):
    """Boolean rows to fit on, or None for all rows."""
    f = ctx.col("freq")
    if f is None or which == "all":
        return None
    m = (f > 0) if which == "eis" else ~(f > 0)
    return m if m.sum() >= 5 else None


# --- shape helpers ----------------------------------------------------------------

def _fin(ctx):
    y = np.asarray(ctx.target, dtype="f8")
    return y[np.isfinite(y)]


def is_counter(ctx, step=1.0, monotonic=True) -> float:
    y = _fin(ctx)
    if y.size < 3 or not np.allclose(y, np.round(y)):
        return 0.0
    d = np.diff(y)
    nz = d[d != 0]
    if nz.size == 0:
        return 0.0
    ok_steps = np.mean(np.isclose(nz, step) | np.isclose(nz, 2 * step))  # a skipped value (rest step) is allowed
    mono = np.mean(d >= 0)
    return float(ok_steps * (mono if monotonic else 1.0))


def is_clock(ctx) -> float:
    y = _fin(ctx)
    if y.size < 3 or ctx.target_dtype != "<f8":
        return 0.0
    d = np.diff(y)
    if np.mean(d >= 0) < 0.999 or np.mean(d > 0) < 0.9:
        return 0.0
    return 1.0


def is_log_spaced(ctx) -> float:
    y = _fin(ctx)
    if (y < 0).any():
        return 0.0
    y = y[y > 0]  # sweeps in modulo-bat (MB) files are separated by zero rows
    if y.size < 4 or y.max() / y.min() < 10:
        return 0.0
    r = y[1:] / y[:-1]
    r = r[r != 1.0]
    if r.size < 3:
        return 0.0
    lr = np.log(r)
    return float(np.mean(np.abs(lr - np.median(lr)) < 0.15 * abs(np.median(lr)) + 1e-9))


def held_at(a, b, tol=0.005) -> float:
    """Share of rows where two potentials agree within tol (V): measured potential held at its set point."""
    if a is None or b is None:
        return 0.0
    a, b = np.asarray(a, dtype="f8"), np.asarray(b, dtype="f8")
    m = np.isfinite(a) & np.isfinite(b)
    return float(np.mean(np.abs(a[m] - b[m]) <= tol)) if m.any() else 0.0


def setpoint_shape(ctx) -> float:
    """A float column that holds a few values exactly between changes."""
    y = _fin(ctx)
    if ctx.target_dtype != "<f4" or y.size < 3 or np.allclose(y, np.round(y)):
        return 0.5 * piecewise_constant(ctx)
    few = np.unique(y).size <= 20
    return (0.68 if few else 0.5) * piecewise_constant(ctx)


def sweep_shape(ctx) -> float:
    """Programmed ramp: steps of (nearly) one constant size, changing sign at the vertices."""
    y = _fin(ctx)
    if ctx.target_dtype != "<f4" or y.size < 10:
        return 0.0
    d = np.diff(y)
    d = d[d != 0]
    if d.size < 8:
        return 0.0
    mag = np.median(np.abs(d))
    if mag == 0:
        return 0.0
    same = np.mean(np.abs(np.abs(d) - mag) <= 0.05 * mag)
    return 0.8 if same > 0.9 else 0.0


def current_held(ctx) -> float:
    """Share of rows where the column sits within 0.5% (or 1 µA) of a current set point."""
    u = ctx.unit("control") or ""
    ctl = ctx.col("control")
    if ctl is None or "mA" not in u or "V" in u.replace("V/mA", ""):
        return 0.0
    if "V" in u and not unit_explained(ctx):
        return 0.0  # control/V/mA also holds potentials (CC-CV steps): a potential on its set point is no current
    a = np.asarray(ctx.target, dtype="f8")
    m = np.isfinite(a) & np.isfinite(ctl) & (ctl != 0)
    if m.sum() < 3:
        return 0.0
    return float(np.mean(np.abs(a[m] - ctl[m]) <= np.maximum(1e-3, 0.02 * np.abs(ctl[m]))))


def ece_explains(c) -> float:
    """Ewe - Ece is evidence only where subtracting Ece leaves at most a tenth of what Ewe alone leaves unexplained.
    Ece is often nearly flat, and then any column that follows Ewe (a potential set point) also fits Ewe - Ece."""
    e, ce = _e(c), c.col("Ece")
    a, b = fit(c.target, e - ce), fit(c.target, e)
    if not a or not b or 1.0 - b[0] < 1e-4:
        return 0.0
    return 1.0 if 1.0 - a[0] <= 0.1 * (1.0 - b[0]) else 0.0


def potential_control(c):
    u = (c.unit("control") or "")
    return c.col("control") if "V" in u else None


def _near(a: float, expected: list[float], tol: float = 0.15) -> bool:
    return any(abs(abs(a) / e - 1) <= tol for e in expected if e)


def unit_explained(ctx) -> bool:
    """True if the column IS a current, charge or power in the file's own units: it fits the derived current
    with scale 1, the accumulated charge with a mA·h or C scale, or E·I with a W scale (r2 >= 0.8).
    Measured on public files: no potential (Ewe: 0/120, Ece: 2/31) passes this, most currents and charges do."""
    i, t = _i(ctx), ctx.col("time")
    y = ctx.target
    with np.errstate(all="ignore"):
        if i is not None:
            f = fit(y, i)
            if f and f[0] >= 0.8 and _near(f[1], [1.0]):
                return True
            if t is not None:
                f = fit(y, np.cumsum(np.nan_to_num(i * d0(t))))
                if f and f[0] >= 0.8 and _near(f[1], [MAH_PER_MA_S, C_PER_MA_S]):
                    return True
            e = ctx.col("Ewe")
            if e is not None:
                f = fit(y, e * i)
                if f and f[0] >= 0.8 and _near(f[1], [W_PER_V_MA]):
                    return True
    return False


def potential_like(ctx) -> float:
    """f4 values in the volts range that are either smooth or held nearly flat (not integer codes, not noise)."""
    if ctx.target_dtype != "<f4" or follows_sweep(ctx) or current_held(ctx) > 0.5 or unit_explained(ctx):
        return 0.0
    y = _fin(ctx)
    if y.size < 4 or np.abs(y).max() > 10 or np.ptp(y) == 0 or np.allclose(y, np.round(y)):
        return 0.0
    a, b = y[:-1] - y[:-1].mean(), y[1:] - y[1:].mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    ac = float((a * b).sum() / den) if den else 0.0
    flat = np.std(y) <= 0.01 * max(abs(np.mean(y)), 1e-3) and abs(np.mean(y)) >= 0.01
    d = np.diff(y)
    accumulator = y[0] == 0 and (np.mean(d >= 0) > 0.98 or np.mean(d <= 0) > 0.98)
    if accumulator:
        return 0.0  # starts at zero and only grows (or only falls): a running total, not a potential
    return 1.0 if (ac > 0.5 or flat) else 0.0


def follows_sweep(ctx) -> bool:
    """True if the column varies systematically with the frequency sweep (rank correlation with log f)."""
    f = ctx.col("freq")
    if f is None:
        return False
    y = np.asarray(ctx.target, dtype="f8")
    m = np.isfinite(y) & (f > 0)
    if m.sum() < 5 or np.ptp(y[m]) == 0:
        return False
    ry = np.argsort(np.argsort(y[m]))
    rf = np.argsort(np.argsort(np.log(f[m])))
    r = np.corrcoef(ry, rf)[0, 1]
    return bool(np.isfinite(r) and abs(r) > 0.6)


def second_potential(ctx) -> float:
    """A potential-like column in a file whose working-electrode potential is already known."""
    if ctx.col("Ewe") is None:
        return 0.0
    return 0.75 * potential_like(ctx) * (1.0 if piecewise_constant(ctx) < 0.9 else 0.2)


def _charge_like(ctx) -> bool:
    """True if the column follows a charge (cumulative current) closely enough to be one."""
    i, t = _i(ctx), ctx.col("time")
    if i is None or t is None:
        return False
    with np.errstate(all="ignore"):
        f = fit(ctx.target, np.cumsum(np.nan_to_num(i * d0(t))))
    return bool(f and f[0] > 0.95)


def piecewise_constant(ctx) -> float:
    """Share of steps where the value does not change at all (set points, counters)."""
    y = _fin(ctx)
    if y.size < 3:
        return 0.0
    return float(np.mean(np.diff(y) == 0))


def phase_shape(ctx) -> float:
    if ctx.target_dtype != "<f4" or ctx.col("|Z|") is None or ctx.col("freq") is None:
        return 0.0
    y = np.asarray(ctx.target, dtype="f8")
    f = ctx.col("freq")
    sweep = np.isfinite(f) & (f > 0) & np.isfinite(y)
    if sweep.sum() < 5 or np.nanmax(np.abs(y)) > 90:
        return 0.0
    ys = y[sweep]
    if np.ptp(ys) < 1.0:
        return 0.0  # a phase angle swings by degrees over a sweep; a held potential does not
    off = ~sweep & np.isfinite(y)
    if off.any() and np.mean(y[off] == 0) < 0.9:
        return 0.0  # outside the sweep rows a phase is not recorded (zero)
    return 0.7


def rcmp_shape(ctx) -> float:
    y = _fin(ctx)
    if ctx.target_dtype != "<f4" or y.size < 3 or (y <= 0).any():
        return 0.0
    rel = np.ptp(y) / np.mean(y)
    return 0.75 if 0 < rel < 1e-5 or (np.ptp(y) == 0 and np.mean(y) > 0.5) else 0.0


def ns_shape(ctx) -> float:
    y = _fin(ctx)
    if y.size < 3 or not np.allclose(y, np.round(y)) or y.min() not in (0, 1) or y.max() > 200 or (y < 0).any():
        return 0.0
    d = np.diff(y)
    moves = d[d != 0]
    if moves.size and (np.mean(d == 0) < 0.5 or np.mean((moves == 1) | (moves < 0)) < 0.8):
        return 0.0  # a sequence index holds for runs of rows and steps forward by one (or loops back)
    s = 0.6
    ctl = ctx.col("control")
    if ctl is not None:
        ch_y = np.diff(y) != 0
        ch_c = np.diff(ctl) != 0
        if ch_y.any():
            s += 0.2 * float(np.mean(ch_c[ch_y]))  # Ns changes where the control changes
    return s


def irange_shape(ctx) -> float:
    y = _fin(ctx)
    if y.size < 3 or not np.allclose(y, np.round(y)) or y.min() < 1 or y.max() > 200:
        return 0.0
    u = np.unique(y)
    if u.size > 12:
        return 0.0
    if ctx.col("Ns") is not None:
        return 0.75  # the file's sequence index is known, and this is the other small code column
    return 0.7 if (u.size <= 6 and np.mean(np.diff(y) == 0) > 0.9) else 0.5


def is_code(ctx, lo=0, hi=200, max_distinct=8) -> float:
    y = _fin(ctx)
    if y.size == 0 or not np.allclose(y, np.round(y)):
        return 0.0
    u = np.unique(y)
    return 1.0 if (u.size <= max_distinct and u.min() >= lo and u.max() <= hi) else 0.0


# --- expected storage types (public parser maps: yadg 7.0.1 and galvani 0.5.0) ---------

def expected_dtypes() -> dict[str, set[str]]:
    from galvani import BioLogic
    from yadg.extractors.eclab.mpr_columns import data_columns

    from spike2 import vocab

    out: dict[str, set[str]] = {}
    for dt, name, _unit in data_columns.values():
        out.setdefault(vocab.canonical(name), set()).add(dt)
    for full, dt in BioLogic.VMPdata_colID_dtype_map.values():
        name = full.split("/")[0] if not full.startswith("Q charge/discharge") else "Q charge/discharge"
        out.setdefault(vocab.canonical(name), set()).add(dt)
    return out


DTYPES = expected_dtypes()


# --- the library ------------------------------------------------------------------

def _t(c):
    return c.col("time")


def _i(c):
    """Current in mA: measured if present, otherwise derived from charge or power columns."""
    i = c.col("I")
    if i is not None:
        return i
    t = c.col("time")
    if t is not None:
        dt = d0(t)
        dt[dt == 0] = np.nan
        dq = c.col("dq")
        if dq is not None:
            return dq / dt * (1000.0 if (c.unit("dq") or "").strip() == "C" else 3600.0)
        qq = c.col("(Q-Qo)")
        if qq is not None:
            return d0(qq) / dt * (1000.0 if (c.unit("(Q-Qo)") or "").strip() == "C" else 3600.0)
    p, e = c.col("Pwe"), c.col("Ewe")
    if p is not None and e is not None:
        return 1000.0 * p / e
    return None


CURRENT = ("I", "dq", "(Q-Qo)", "Pwe")  # any of these lets _i() produce a current


def _e(c):
    return c.col("Ewe")


def build() -> dict[str, Quantity]:
    L: dict[str, Quantity] = {}

    def q(name, **kw):
        L[name] = Quantity(name, **kw)

    def i_dt(c):  # mA*s per point; rows of an impedance sweep carry no DC charge
        x = _i(c) * d0(_t(c))
        return x
    q("time", shapes=[Shape("monotonic f8 clock", is_clock)])
    q("step time", relations=[
        Relation("time since Ns changed", ("time", "Ns"), lambda c: since_change(_t(c), c.col("Ns")), [1.0]),
        Relation("time since half cycle changed", ("time", "half cycle"), lambda c: since_change(_t(c), c.col("half cycle")), [1.0]),
        Relation("time since control changed", ("time", "control"), lambda c: since_change(_t(c), c.col("control")), [1.0]),
        Relation("time since cycle number changed", ("time", "cycle number"), lambda c: since_change(_t(c), c.col("cycle number")), [1.0]),
        Relation("time since the record started (one step)", ("time",), lambda c: _t(c) - _t(c)[0], [1.0]),
        Relation("a clock that restarts: time since its own last reset", ("time",),
                 lambda c: since_change(_t(c), np.cumsum(np.r_[0, np.diff(np.asarray(c.target, dtype="f8")) < 0])), [1.0]),
        Relation("time since the sweep direction changed", ("time", "control"),
                 lambda c: since_change(_t(c), np.sign(d0(c.col("control")))), [1.0]),
    ])
    def i_dt_prev(c):  # some sequences (modulo bat) integrate the previous row's current over the interval
        i = _i(c)
        return np.r_[0.0, i[:-1]] * d0(_t(c))

    q("dq", relations=[Relation("I*dt per point", (CURRENT, "time"), i_dt, [MAH_PER_MA_S, C_PER_MA_S]),
                       Relation("previous-row I*dt per point", (CURRENT, "time"), i_dt_prev, [MAH_PER_MA_S, C_PER_MA_S]),
                       Relation("|I|*dt per point (current known only as |P/E|)", ("Pwe", "Ewe", "time"),
                                lambda c: np.abs(1000.0 * c.col("Pwe") / _e(c)) * d0(_t(c)), [MAH_PER_MA_S, C_PER_MA_S],
                                abs_target=True)])
    q("(Q-Qo)", relations=[
        Relation("cumulative I*dt", (CURRENT, "time"), lambda c: np.cumsum(np.nan_to_num(i_dt(c))), [MAH_PER_MA_S, C_PER_MA_S]),
        Relation("cumulative previous-row I*dt", (CURRENT, "time"), lambda c: np.cumsum(np.nan_to_num(i_dt_prev(c))), [MAH_PER_MA_S, C_PER_MA_S]),
        Relation("cumulative dq", ("dq",), lambda c: np.cumsum(np.nan_to_num(c.col("dq"))), [1.0, 3.6]),
        Relation("I*dt summed since Ns changed (Qo reset by a linked technique)", (CURRENT, "time", "Ns"),
                 lambda c: cumsum_by(i_dt(c), c.col("Ns")), [MAH_PER_MA_S, C_PER_MA_S]),
    ])
    q("dQ", relations=[
        Relation("I*dt summed within half cycle", (CURRENT, "time", "half cycle"), lambda c: cumsum_by(i_dt(c), c.col("half cycle")), [MAH_PER_MA_S, C_PER_MA_S]),
        Relation("I*dt summed since Ns changed", (CURRENT, "time", "Ns"), lambda c: cumsum_by(i_dt(c), c.col("Ns")), [MAH_PER_MA_S, C_PER_MA_S]),
        Relation("I*dt summed since control changed", (CURRENT, "time", "control"), lambda c: cumsum_by(i_dt(c), c.col("control")), [MAH_PER_MA_S, C_PER_MA_S]),
    ])
    q("Q charge or discharge", relations=[
        Relation("I*dt summed within cycle", (CURRENT, "time", "half cycle"), lambda c: cumsum_by(i_dt(c), np.floor(c.col("half cycle") / 2)), [MAH_PER_MA_S, C_PER_MA_S]),
        Relation("I*dt summed within half cycle", (CURRENT, "time", "half cycle"), lambda c: cumsum_by(i_dt(c), c.col("half cycle")), [MAH_PER_MA_S, C_PER_MA_S]),
        Relation("dq summed within half cycle", ("dq", "half cycle"), lambda c: cumsum_by(c.col("dq"), c.col("half cycle")), [1.0, 3.6]),
        Relation("I*dt summed within current-sign segment", (CURRENT, "time"), lambda c: cumsum_by(i_dt(c), sign_changes_count(_i(c))), [MAH_PER_MA_S, C_PER_MA_S]),
        Relation("I*dt summed within sequence", (CURRENT, "time", "Ns"), lambda c: cumsum_by(i_dt(c), c.col("Ns")), [MAH_PER_MA_S, C_PER_MA_S]),
        Relation("I*dt summed since control changed", (CURRENT, "time", "control"), lambda c: cumsum_by(i_dt(c), c.col("control")), [MAH_PER_MA_S, C_PER_MA_S]),
    ])
    q("half cycle", relations=[
        Relation("count of current sign changes", (CURRENT,), lambda c: sign_changes_count(_i(c)), [1.0, 2.0]),
    ], shapes=[Shape("integer counter stepping by 1 (no impedance sweep)", lambda c: 0.75 * is_counter(c) if c.col("freq") is None else 0.3 * is_counter(c))])
    q("cycle number", relations=[
        Relation("half cycle / 2", ("half cycle",), lambda c: np.floor(c.col("half cycle") / 2), [1.0]),
    ], shapes=[Shape("integer-valued counter stored as float", lambda c: 0.75 * is_counter(c))])
    q("z cycle", shapes=[Shape("integer counter in a sequence with impedance", lambda c: 0.75 * is_counter(c) if c.col("freq") is not None else 0.0)])
    q("Ns", shapes=[Shape("sequence index from 0 that changes with the control", lambda c: ns_shape(c))])
    q("I Range", shapes=[Shape("instrument range code (few values, not starting at 0)", lambda c: irange_shape(c))])
    q("Pwe", relations=[Relation("Ewe*I", ("Ewe", CURRENT), lambda c: _e(c) * _i(c), [W_PER_V_MA])])
    q("Ewe", relations=[
        Relation("Pwe/I", ("Pwe", ("I", "dq", "(Q-Qo)")), lambda c: c.col("Pwe") / _i(c), [1000.0]),
        Relation("follows potential control", ("control",), lambda c: c.col("control"), [1.0]),
    ], shapes=[Shape("held at the potential set point", lambda c: 0.9 * held_at(c.target, potential_control(c)), intrinsic=False)])
    q("I", relations=[
        Relation("Pwe/Ewe", ("Pwe", "Ewe"), lambda c: c.col("Pwe") / _e(c), [1000.0]),
        Relation("d(dq)/dt", ("dq", "time"), lambda c: c.col("dq") / d0(_t(c)), [3600.0, 1000.0]),
        Relation("d(Q-Qo)/dt", ("(Q-Qo)", "time"), lambda c: d0(c.col("(Q-Qo)")) / d0(_t(c)), [3600.0, 1000.0]),
        Relation("follows current control", ("control",), lambda c: c.col("control"), [1.0]),
    ])
    L["I"].shapes.append(Shape("measured current held at the current set point",
                               lambda c: (0.95 if current_held(c) > 0.5 else 0.85 * current_held(c))
                               * (1.0 if piecewise_constant(c) < 0.9 else 0.2), intrinsic=False))
    for rel in L["I"].relations:  # a measured current moves; a value held exactly between changes is a set point
        rel.gate = lambda c: 1.0 - 0.8 * min(1.0, max(0.0, piecewise_constant(c) - 0.8) / 0.18)
    q("control", relations=[
        Relation("set point followed by I", (CURRENT,), lambda c: _i(c), [1.0],
                 gate=lambda c: 0.4 + 0.6 * piecewise_constant(c)),  # a current set point holds still between changes
        Relation("set point followed by Ewe", ("Ewe",), lambda c: _e(c), [1.0]),
    ], shapes=[Shape("piecewise-constant set point", lambda c: setpoint_shape(c), intrinsic=False),
               Shape("programmed potential sweep (piecewise linear at a constant scan rate)", lambda c: sweep_shape(c)),
               Shape("potential set point that Ewe sits on", lambda c: 0.85 * held_at(c.target, _e(c)) * (0.5 + 0.5 * piecewise_constant(c)), intrinsic=False)])
    e_i_dt = lambda c: _e(c) * i_dt(c)  # noqa: E731
    cycle_key = lambda c: np.floor(c.col("half cycle") / 2)  # noqa: E731  EC-Lab resets per-cycle totals every 2 half cycles
    q("Energy we charge", relations=[
        Relation("E*I*dt summed while charging, per cycle", ("Ewe", CURRENT, "time", "half cycle"),
                 lambda c: cumsum_by(np.where(_i(c) > 0, e_i_dt(c), 0.0), cycle_key(c)), [WH_PER_V_MA_S]),
        Relation("E*I*dt summed while charging, per half cycle", ("Ewe", CURRENT, "time", "half cycle"),
                 lambda c: cumsum_by(np.where(_i(c) > 0, e_i_dt(c), 0.0), c.col("half cycle")), [WH_PER_V_MA_S]),
        Relation("E*I*dt summed while charging", ("Ewe", CURRENT, "time"),
                 lambda c: np.cumsum(np.nan_to_num(np.where(_i(c) > 0, e_i_dt(c), 0.0))), [WH_PER_V_MA_S])])
    q("Energy we discharge", relations=[
        Relation("E*I*dt summed while discharging, per cycle", ("Ewe", CURRENT, "time", "half cycle"),
                 lambda c: cumsum_by(np.where(_i(c) < 0, -e_i_dt(c), 0.0), cycle_key(c)), [WH_PER_V_MA_S]),
        Relation("E*I*dt summed while discharging, per half cycle", ("Ewe", CURRENT, "time", "half cycle"),
                 lambda c: cumsum_by(np.where(_i(c) < 0, -e_i_dt(c), 0.0), c.col("half cycle")), [WH_PER_V_MA_S]),
        Relation("E*I*dt summed while discharging", ("Ewe", CURRENT, "time"),
                 lambda c: np.cumsum(np.nan_to_num(np.where(_i(c) < 0, -e_i_dt(c), 0.0))), [WH_PER_V_MA_S])])
    q("|Energy|", relations=[
        Relation("|E*I|*dt summed", ("Ewe", CURRENT, "time"), lambda c: np.cumsum(np.nan_to_num(np.abs(e_i_dt(c)))), [WH_PER_V_MA_S]),
        Relation("|E*I|*dt summed per half cycle", ("Ewe", CURRENT, "time", "half cycle"),
                 lambda c: cumsum_by(np.abs(e_i_dt(c)), c.col("half cycle")), [WH_PER_V_MA_S])])
    q("Energy we", relations=[
        Relation("E*I*dt summed", ("Ewe", CURRENT, "time"), lambda c: np.cumsum(np.nan_to_num(e_i_dt(c))), [WH_PER_V_MA_S]),
        Relation("E*I*dt summed per half cycle", ("Ewe", CURRENT, "time", "half cycle"),
                 lambda c: cumsum_by(e_i_dt(c), c.col("half cycle")), [WH_PER_V_MA_S])])

    # --- impedance (PEIS/GEIS) ---
    w = lambda c: 2 * np.pi * c.col("freq")  # noqa: E731
    re, mim, mod = (lambda c: c.col("Re(Z)")), (lambda c: c.col("-Im(Z)")), (lambda c: c.col("|Z|"))
    q("freq", shapes=[Shape("log-spaced positive frequencies", is_log_spaced)])
    q("|Z|", relations=[Relation("sqrt(Re^2+Im^2)", ("Re(Z)", "-Im(Z)"), lambda c: np.hypot(re(c), mim(c)), [1.0]),
                        Relation("|Ewe|/|I|", ("|Ewe|", "|I|"), lambda c: c.col("|Ewe|") / c.col("|I|"), [1.0])])
    q("Re(Z)", relations=[Relation("|Z|*cos(phase)", ("|Z|", "Phase(Z)"), lambda c: mod(c) * np.cos(np.radians(c.col("Phase(Z)"))), [1.0]),
                          Relation("sqrt(|Z|^2-Im^2)", ("|Z|", "-Im(Z)"), lambda c: np.sqrt(np.maximum(mod(c) ** 2 - mim(c) ** 2, 0)), [1.0])])
    q("-Im(Z)", relations=[Relation("-|Z|*sin(phase)", ("|Z|", "Phase(Z)"), lambda c: -mod(c) * np.sin(np.radians(c.col("Phase(Z)"))), [1.0]),
                           Relation("sqrt(|Z|^2-Re^2)", ("|Z|", "Re(Z)"), lambda c: np.sqrt(np.maximum(mod(c) ** 2 - re(c) ** 2, 0)), [1.0])])
    q("Phase(Z)", relations=[Relation("atan2(Im, Re) in degrees", ("Re(Z)", "-Im(Z)"), lambda c: np.degrees(np.arctan2(-mim(c), re(c))), [1.0])])
    L["Phase(Z)"].shapes.append(Shape("phase angle in degrees (within ±90) that follows the frequency sweep",
                                      lambda c: phase_shape(c), intrinsic=False))
    q("Cs", relations=[Relation("1/(w*(-Im)) in µF", ("freq", "-Im(Z)"), lambda c: 1.0 / (w(c) * mim(c)), [1e6])])
    q("Cp", relations=[Relation("(-Im)/(w*|Z|^2) in µF", ("freq", "-Im(Z)", "Re(Z)"),
                                lambda c: mim(c) / (w(c) * (re(c) ** 2 + mim(c) ** 2)), [1e6])])
    q("|Ewe|", relations=[Relation("|Z|*|I|", ("|Z|", "|I|"), lambda c: mod(c) * c.col("|I|"), [1.0])])
    q("|I|", relations=[Relation("|Ewe|/|Z|", ("|Ewe|", "|Z|"), lambda c: c.col("|Ewe|") / mod(c), [1.0])])
    for el, amp in (("Zce", "|Ece|"), ):
        q(f"|{el}|", relations=[Relation(f"{amp}/|I|", (amp, "|I|"), lambda c, amp=amp: c.col(amp) / c.col("|I|"), [1.0]),
                                Relation(f"sqrt(Re^2+Im^2) of {el}", (f"Re({el})", f"-Im({el})"),
                                         lambda c, el=el: np.hypot(c.col(f"Re({el})"), c.col(f"-Im({el})")), [1.0])])
        q(amp, relations=[Relation(f"|{el}|*|I|", (f"|{el}|", "|I|"), lambda c, el=el: c.col(f"|{el}|") * c.col("|I|"), [1.0])])
    for el in ("Zce", "Zwe-ce"):
        q(f"Re({el})", relations=[Relation(f"|{el}|*cos(phase)", (f"|{el}|", f"Phase({el})"),
                                           lambda c, el=el: c.col(f"|{el}|") * np.cos(np.radians(c.col(f"Phase({el})"))), [1.0])])
        q(f"-Im({el})", relations=[Relation(f"-|{el}|*sin(phase)", (f"|{el}|", f"Phase({el})"),
                                            lambda c, el=el: -c.col(f"|{el}|") * np.sin(np.radians(c.col(f"Phase({el})"))), [1.0])])
        q(f"Phase({el})", relations=[Relation(f"atan2 of {el}", (f"Re({el})", f"-Im({el})"),
                                              lambda c, el=el: np.degrees(np.arctan2(-c.col(f"-Im({el})"), c.col(f"Re({el})"))), [1.0])])
    q("|Zwe-ce|", relations=[Relation("sqrt(Re^2+Im^2) of Zwe-ce", ("Re(Zwe-ce)", "-Im(Zwe-ce)"),
                                      lambda c: np.hypot(c.col("Re(Zwe-ce)"), c.col("-Im(Zwe-ce)")), [1.0]),
                             Relation("|Z(we) - Z(ce)|", ("Re(Z)", "-Im(Z)", "Re(Zce)", "-Im(Zce)"),
                                      lambda c: np.hypot(re(c) - c.col("Re(Zce)"), mim(c) - c.col("-Im(Zce)")), [1.0])])
    for name, amp in (("THD Ewe", "|Ewe|"), ("THD I", "|I|")):
        base = "Ewe" if "Ewe" in name else "I"
        q(name, relations=[Relation(f"sqrt(sum of {base} harmonics^2)/{amp} in %",
                                    (amp, f"|{base} h2|"),
                                    lambda c, base=base, amp=amp: np.sqrt(sum(np.nan_to_num(c.col(f"|{base} h{k}|")) ** 2
                                                                      for k in range(2, 8) if c.col(f"|{base} h{k}|") is not None)) / c.col(amp),
                                    [100.0])])
    # Capacitance charge/discharge (EC-Lab manual 2014, §5.1.5): Q / ΔE within the sequence, ΔE from its first point.
    def start_value(x, key):
        x, key = np.asarray(x, dtype="f8"), np.asarray(key)
        out = np.empty_like(x)
        cur = x[0] if len(x) else 0.0
        for i in range(len(x)):
            if i and key[i] != key[i - 1]:
                cur = x[i]
            out[i] = cur
        return out

    def _cap(c, key, sign):
        i = _i(c)
        q_seg = np.abs(cumsum_by(np.where(sign * i > 0, i * d0(_t(c)), 0.0), key))
        de = np.abs(_e(c) - start_value(_e(c), key))
        with np.errstate(all="ignore"):
            out = q_seg / de
        y = np.asarray(c.target, dtype="f8")
        bad = ~np.isfinite(out) | (de < 0.05 * np.nanmax(de)) | (y == 0)  # ΔE near zero is unstable; 0 = not reported
        out[bad] = np.nan
        return out

    def capacitance(c, key_name, sign):
        return _cap(c, c.col(key_name), sign)

    def capacitance_cycle(c, sign):
        return _cap(c, np.floor(c.col("half cycle") / 2), sign)

    capacitance_relations = []
    for cname, sign in (("Capacitance charge", 1.0), ("Capacitance discharge", -1.0)):
        q(cname, relations=[
            Relation("Q/ΔE within half cycle (either current sign)", ("Ewe", CURRENT, "time", "half cycle"),
                     lambda c, sign=sign: capacitance(c, "half cycle", -sign)),
            Relation(f"Q/ΔE within half cycle ({'charging' if sign > 0 else 'discharging'})", ("Ewe", CURRENT, "time", "half cycle"),
                     lambda c, sign=sign: capacitance(c, "half cycle", sign)),
            Relation(f"Q/ΔE within cycle ({'charging' if sign > 0 else 'discharging'})", ("Ewe", CURRENT, "time", "half cycle"),
                     lambda c, sign=sign: capacitance_cycle(c, sign)),
            Relation(f"Q/ΔE within sequence ({'charging' if sign > 0 else 'discharging'})", ("Ewe", CURRENT, "time", "Ns"),
                     lambda c, sign=sign: capacitance(c, "Ns", sign)),
        ])
    for cname in ("Capacitance charge", "Capacitance discharge"):
        for rel in L[cname].relations:
            rel.reliable = False  # ΔE near zero makes Q/ΔE unstable; never use it to rule the name out
        L[cname].gate = lambda c: 1.0 if np.nanmax(np.abs(np.asarray(c.target, dtype="f8"))) >= 1.0 else 0.1  # µF of a cell
    q("Ece", shapes=[Shape("a second smooth potential next to a known Ewe", second_potential, intrinsic=False)])
    q("Ewe-Ece", relations=[Relation("Ewe - Ece", ("Ewe", "Ece"), lambda c: _e(c) - c.col("Ece"), [1.0],
                                     gate=ece_explains)])
    L["Ewe"].shapes.append(Shape("the file's working-electrode potential is missing and this column is one",
                                 lambda c: 0.7 * potential_like(c) * (1.0 if piecewise_constant(c) < 0.9 else 0.2), intrinsic=False))
    q("Rcmp", shapes=[Shape("compensated resistance: positive, constant up to float rounding", rcmp_shape)])
    for name in ("Analog IN 1", "Analog IN 2", "NSD Ewe", "NSR Ewe", "NSD I", "NSR I"):
        q(name, prior=0.0)
    return L


LIBRARY = build()
