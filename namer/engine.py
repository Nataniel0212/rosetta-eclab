"""Standalone column namer: no language model at run time.

For an unnamed column TARGET and the file's named columns, every candidate
quantity in the library (namer/library.py) is scored by code:

  1. relation checks: the candidate's defining relation to other columns (e.g.
     dq = I*dt/3600) is computed from the known columns and fitted to TARGET;
     the score uses r2 and whether the fitted scale matches the expected one
     (unit conversions allowed), so dq, (Q-Qo) and Q charge/discharge separate.
  2. shape checks: properties that need no other column (monotonic clock,
     integer counter stepping by 1, log-spaced frequency, small set of codes).
  3. abstention: if no candidate is supported well enough, the answer is
     ABSTAIN with the best candidates listed as hints.

Every answer carries its evidence (the relation, r2, fitted scale), so a
maintainer can check it before adding the column to a parser.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Known:
    name: str  # canonical name (namer.library uses canonical names)
    unit: str | None
    values: np.ndarray
    dtype: str = ""


@dataclass
class Context:
    technique: str | None
    known: dict[str, Known]  # canonical name -> column
    target: np.ndarray
    target_dtype: str
    alone: bool = True  # the target is the only column of the record missing from `known`; rules that infer a
    #                     column from what the record lacks need this (False when other columns are unknown too)
    others: tuple | None = None  # with alone False: the record's other missing columns as (values, dtype) pairs
    #                              (None: not given)

    def col(self, *names: str) -> np.ndarray | None:
        for n in names:
            if n in self.known:
                return np.asarray(self.known[n].values, dtype="f8")
        return None

    def unit(self, *names: str) -> str | None:
        for n in names:
            if n in self.known:
                return self.known[n].unit
        return None


@dataclass
class Evidence:
    candidate: str
    score: float  # 0..1
    kind: str  # "relation" | "shape" | "prior"
    detail: str
    extras: dict = field(default_factory=dict)


def fit(target: np.ndarray, pred: np.ndarray, robust: bool = True) -> tuple[float, float, float] | None:
    """Least squares target ~ a*pred + b; returns (r2, a, b) or None if not testable."""
    y = np.asarray(target, dtype="f8")
    x = np.asarray(pred, dtype="f8")
    m = np.isfinite(y) & np.isfinite(x)
    if m.sum() < 3:
        return None
    y, x = y[m], x[m]
    if np.ptp(y) == 0 or np.ptp(x) == 0:
        return None
    a, b = np.polyfit(x, y, 1)
    res = y - (a * x + b)
    ss = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float((res ** 2).sum()) / ss
    # Robust variant: real records have single glitch rows at step transitions. Refit without the worst
    # 5% of rows (at least one, only for n >= 30 and a relation that already roughly holds).
    n = y.size
    if robust and n >= 30 and r2 < 0.999:
        k = max(1, int(np.ceil(0.05 * n)))
        a2, b2, r2t = a, b, r2
        lev = np.argsort(np.abs(x - np.median(x)))[: n - k]  # start without the k highest-leverage rows
        if np.ptp(x[lev]) > 0:
            a2, b2 = np.polyfit(x[lev], y[lev], 1)
        for _ in range(3):  # then trim by residual against the refitted line
            keep = np.argsort(np.abs(y - (a2 * x + b2)))[: n - k]
            yk, xk = y[keep], x[keep]
            if np.ptp(yk) == 0 or np.ptp(xk) == 0:
                break
            a2, b2 = np.polyfit(xk, yk, 1)
            ss2 = float(((yk - yk.mean()) ** 2).sum())
            r2t = 1.0 - float(((yk - (a2 * xk + b2)) ** 2).sum()) / ss2
        if r2t > r2:
            return (r2t, float(a2), float(b2))
    return (r2, float(a), float(b))


def scale_match(a: float, expected: list[float], tol: float = 0.03) -> float | None:
    """Relative closeness of the fitted scale to any expected scale (None if no expectation)."""
    if not expected:
        return None
    best = min(abs(a / e - 1.0) for e in expected if e)
    return 1.0 if best <= tol else max(0.0, 1.0 - (best - tol) / 0.5)


CE_EIS = ("|Ece|", "|Zce|", "Re(Zce)", "-Im(Zce)", "Phase(Zce)", "|Zwe-ce|", "Re(Zwe-ce)", "-Im(Zwe-ce)",
          "Phase(Zwe-ce)", "THD Ece", "NSD Ece", "NSR Ece")


def needs_ce_channel(name: str) -> bool:
    return name in CE_EIS or name.startswith("|Ece h")


def has_ce_channel(present: set[str]) -> bool:
    return any(n in present for n in ("Ece", "|Ece|", "|Zce|", "Re(Zce)", "-Im(Zce)", "Phase(Zce)"))


def single_sibling(ctx: Context, vocabulary: list[str], present: set[str], cooccur=None, library=None) -> Evidence | None:
    """Exactly one candidate is one token away from exactly one known column of the same storage and magnitude
    (e.g. Analog IN 2 next to a known Analog IN 1)."""
    from spike2.elim import one_token_apart

    y = np.asarray(ctx.target, dtype="f8")
    y = y[np.isfinite(y)]
    if y.size < 3:
        return None
    hits = []
    for cand in vocabulary:
        if cand in present:
            continue
        sibs = [k for k in present if one_token_apart(cand, k)]
        if len(sibs) != 1:
            continue
        if ctx.known[sibs[0]].dtype != ctx.target_dtype:
            continue
        together = cooccur(cand, sibs[0]) if cooccur is not None else 0.0
        if cooccur is not None and together < 0.5:
            continue  # not a pair that is normally recorded together
        spec = library.get(cand) if library is not None else None
        if spec is not None:
            ev = spec.evaluate(ctx)
            if ev is not None and ev.kind == "relation" and ev.extras.get("r2", 1.0) < 0.3:
                continue  # the candidate's own defining relation contradicts it
        s = np.asarray(ctx.known[sibs[0]].values, dtype="f8")
        s = s[np.isfinite(s)]
        if s.size == 0:
            continue
        mag_t, mag_s = np.abs(y).max() + 1e-12, np.abs(s).max() + 1e-12
        if 0.1 <= mag_t / mag_s <= 10 or together >= 0.9:
            hits.append((cand, sibs[0]))
    if len(hits) != 1:
        return None
    cand, sib = hits[0]
    return Evidence(cand, 0.62, "sibling", f"the only missing sibling of the known column {sib}, with similar magnitude")


def name_column(ctx: Context, library, vocabulary: list[str], abstain_below: float = 0.6,
                prior: dict[str, float] | None = None, cooccur=None) -> dict:
    """Rank the vocabulary names that are not already present; return the answer with evidence."""
    present = set(ctx.known)
    evidences: list[Evidence] = []
    for cand in vocabulary:
        if cand in present:
            continue
        spec = library.get(cand)
        if spec is None:
            continue
        ev = spec.evaluate(ctx)
        if ev is not None:
            if prior:  # how common the name is for this technique breaks near-ties between equal relations
                ev.score = ev.score + 0.03 * prior.get(cand, 0.0)  # may exceed 1: only the order matters
            evidences.append(ev)
    sib = single_sibling(ctx, vocabulary, present, cooccur, library) if ctx.alone else None  # "the only missing"
    if sib is not None:
        evidences.append(sib)
    for e in evidences:
        if e.kind != "relation" and needs_ce_channel(e.candidate) and not has_ce_channel(present):
            e.score *= 0.2  # no other counter-electrode quantity was recorded in this file
    evidences.sort(key=lambda e: -e.score)
    def rival(e):  # a relation with the wrong units is no rival
        return e.kind != "relation" or e.extras.get("scale_match", 1.0) >= 0.5

    rivals = [e for e in evidences[1:] if rival(e)] if evidences else []
    if (evidences and evidences[0].score < abstain_below and evidences[0].kind == "relation"
            and evidences[0].score >= 0.3 and evidences[0].extras.get("r2", 0) >= 0.8
            and evidences[0].extras.get("scale_match", 0) >= 0.5
            and (not rivals or rivals[0].score <= 0.25 * evidences[0].score)):
        top = evidences[0]
        return {"verdict": "NAME", "name": top.candidate, "confidence": round(0.5 * top.score, 3),
                "alternatives": [e.candidate for e in evidences[1:3]],
                "evidence": f"approximate relation, far ahead of every alternative: {top.detail}",
                "ranking": [(e.candidate, round(e.score, 3), e.detail[:90]) for e in evidences[:5]]}
    if not evidences or evidences[0].score < abstain_below:
        return {"verdict": "ABSTAIN", "name": "NONE", "confidence": evidences[0].score if evidences else 0.0,
                "alternatives": [e.candidate for e in evidences[:2]],
                "ranking": [(e.candidate, round(e.score, 3), e.detail[:90]) for e in evidences[:5]],
                "evidence": "; ".join(f"{e.candidate}: {e.detail}" for e in evidences[:3]) or "no candidate fits"}
    top = evidences[0]
    margin = top.score - (evidences[1].score if len(evidences) > 1 else 0.0)
    return {"verdict": "NAME", "name": top.candidate, "confidence": round(min(1.0, min(1.0, top.score) * (0.5 + margin)), 3),
            "alternatives": [e.candidate for e in evidences[1:3]],
            "evidence": f"{top.kind}: {top.detail}", "ranking": [(e.candidate, round(e.score, 3), e.detail[:90]) for e in evidences[:5]]}
