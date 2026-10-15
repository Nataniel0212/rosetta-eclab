"""Evaluate the standalone namer.

  python -m namer.evaluate --pool            # development pool: every corpus column NOT used by the v2 test split
  python -m namer.evaluate --split dev       # v2 dev tasks  -> runs3/rules_dev.jsonl  (score with bench.claude_cli --score)
  python -m namer.evaluate --split test      # v2 test tasks -> runs3/rules_test.jsonl (run rarely; record in results/open_runs.md)

Pipeline per column: constant -> ABSTAIN; identical to a known column -> DUPLICATE;
name elimination (spike2/elim.py) -> NAME; otherwise the formula library (namer/library.py).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from spike2 import corpus, elim, features, tasks, vocab

from .engine import Context, Known, name_column
from .fallback import Stats, lag1_autocorr
from .library import LIBRARY

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs3"


_STATS: Stats | None = None


def consistent(name: str, ctx: Context) -> bool:
    """A statistics guess must not contradict what the library knows about the quantity's form."""
    from .library import DTYPES

    exp = DTYPES.get(name)
    if exp and ctx.target_dtype not in exp:
        return False
    spec = LIBRARY.get(name)
    from .library import is_clock, piecewise_constant

    if name in ("Analog IN 1", "Analog IN 2", "Analog OUT"):
        return False  # external inputs have no physics to check; only the sibling rule may name them
    if is_clock(ctx) and name not in ("time", "step time", "(Q-Qo)", "Energy we charge", "Energy we discharge", "|Energy|"):
        return False  # a monotonic clock is a time (or a running total), nothing else
    from .engine import has_ce_channel, needs_ce_channel

    if needs_ce_channel(name) and not has_ce_channel(set(ctx.known)):
        return False
    if name == "control" and piecewise_constant(ctx) < 0.5:
        return False  # a set point holds still between changes
    form = [sh for sh in (spec.shapes if spec is not None else []) if sh.intrinsic]
    if form and max(float(sh.test(ctx)) for sh in form) <= 0:
        return False
    if spec is not None:
        f = ctx.col("freq")
        impedance_record = f is not None and np.mean(f > 0) > 0.5
        from .library import EIS_FAMILY as _EIS
        if not (impedance_record and name not in _EIS):
            # In impedance records the instrument integrates charge and current over measurement windows that the
            # time stamps do not show, so DC relations cannot rule a name out there.
            ev = spec.best_reliable_relation(ctx)
            if ev is not None and ev.score < 0.3:
                return False  # its defining relation was testable here and does not hold with the right units
        from .library import EIS_FAMILY
        if name in EIS_FAMILY and ctx.col("freq") is None:
            return False
    return True


def stats() -> Stats:
    global _STATS
    if _STATS is None:
        _STATS = Stats()
    return _STATS


def answer(target, dtype, known_cols: list, technique, vocabulary: list[str], abstain_below: float = 0.6,
           exclude_group: str | None = None, noise_autocorr: float = 0.3, fallback_min_share: float = 0.5) -> dict:
    if features.is_constant(target):
        return {"verdict": "ABSTAIN", "name": "NONE", "confidence": 1.0, "evidence": "constant column"}
    same = [c.name for c in known_cols if features.identical(target, c.values)]
    if same:
        return {"verdict": "DUPLICATE", "name": "NONE", "duplicate_of": same[0], "confidence": 1.0,
                "evidence": f"identical to {same[0]}"}
    e = elim.eliminate([c.name for c in known_cols], vocabulary)
    if e:
        return {"verdict": "NAME", "name": e, "confidence": 0.9, "evidence": "name elimination (sibling gap)"}
    known = {}
    for c in known_cols:
        known.setdefault(vocab.canonical(c.name), Known(vocab.canonical(c.name), c.unit, np.asarray(c.values), c.dtype))
    ctx = Context(technique, known, np.asarray(target), dtype)
    pr = stats().prior(technique, exclude_group)
    gl = stats().prior(None, exclude_group, all_techniques=True)
    top, gtop = (max(pr.values()) if pr else 0), (max(gl.values()) if gl else 0)
    prior = {k: (pr.get(k, 0) / top if top else 0) + 0.3 * (gl.get(k, 0) / gtop if gtop else 0) for k in set(pr) | set(gl)}
    a = name_column(ctx, LIBRARY, vocabulary, abstain_below, prior,
                    cooccur=lambda cand, sib: stats().cooccur(cand, sib, exclude_group))
    if a["verdict"] == "NAME":
        return a
    # (Tried 2026-10-15: naming "the column that similar files always record and this file lacks". It names
    # decoys whenever a file naturally lacks a common column: 10/19 decoys on hold-out 4, still 6/19 when limited
    # to structured columns. Trustworthiness wins: not used.)
    # No formula or shape identifies the column: statistics fallback, unless it looks like noise.
    ac = lag1_autocorr(target)
    if ac < noise_autocorr:
        a["evidence"] = f"no formula fits and the column looks like noise (lag-1 autocorrelation {ac:.2f})"
        return a
    name, share, alts = stats().predict(target, dtype, known_cols, technique, set(vocabulary), set(known),
                                        exclude_group)
    if name and not consistent(name, ctx):
        a["evidence"] = f"statistics suggest {name}, but the column does not have its form (shape or storage type)"
        return a
    if name and share >= fallback_min_share:
        return {"verdict": "NAME", "name": name, "confidence": round(0.5 * share, 3), "alternatives": alts,
                "evidence": f"no formula fits; statistics of similar columns in other files (vote share {share:.2f})"}
    return a


def test_keys() -> set[tuple[str, int]]:
    m = json.loads((ROOT / "prereg2" / "tasks_manifest.json").read_text(encoding="utf-8"))
    return {(t["file"], t["position"]) for t in m["tasks"] if t["split"] == "test" and t["position"] is not None}


def pool(verbose: int) -> None:
    excluded = test_keys()
    V = vocab.names()
    good, by_name, confusions = Counter(), defaultdict(list), Counter()
    for p in corpus.corpus_files():
        try:
            fd = tasks.file_data(p.name)
        except Exception:
            continue
        if fd.n_points < tasks.MIN_ROWS:
            continue
        cols = tasks.usable_columns(fd)
        for c in cols:
            if (p.name, c.position) in excluded or features.is_constant(c.values):
                continue
            others = [o for o in cols if o is not c]
            if any(features.identical(c.values, o.values) for o in others):
                continue
            truth = vocab.canonical(c.name)
            a = answer(c.values, c.dtype, others, fd.technique, V, exclude_group=corpus.source_group(p.name))
            ok = a["verdict"] == "NAME" and a["name"] == truth
            by_name[truth].append(ok)
            if not ok:
                confusions[(truth, a["name"] if a["verdict"] == "NAME" else a["verdict"])] += 1
    macro = np.mean([np.mean(v) for v in by_name.values()])
    micro = np.mean([x for v in by_name.values() for x in v])
    print(f"pool: {sum(len(v) for v in by_name.values())} columns, {len(by_name)} names, macro {macro:.3f}, micro {micro:.3f}")
    for n, v in sorted(by_name.items(), key=lambda kv: np.mean(kv[1])):
        if np.mean(v) < 1.0 or verbose > 1:
            print(f"  {np.mean(v):.2f} ({sum(v)}/{len(v)})  {n}")
    if verbose:
        print("top confusions:")
        for (t, a), k in confusions.most_common(25):
            print(f"  {k:3d}  {t} -> {a}")


def bench(split: str, abstain_below: float) -> None:
    RUNS.mkdir(exist_ok=True)
    out = []
    for t in tasks.load(split):
        x, dtype, kv, kc = tasks.hidden_and_known(t)
        a = answer(x, dtype, list(kc.values()), t["technique"], tasks.task_vocabulary(t), abstain_below,
                   exclude_group=corpus.source_group(t["file"]))
        out.append({"task_id": t["task_id"], "split": split, "status": "ok", "arm": "rules", **a})
    path = RUNS / f"rules_{split}.jsonl"
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False, default=str) for r in out) + "\n",
                    encoding="utf-8", newline="\n")
    print(f"{len(out)} answers -> {path}")


def main() -> int:
    import warnings

    warnings.filterwarnings("ignore")
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", action="store_true")
    ap.add_argument("--split", choices=["dev", "test", "holdout", "holdout2", "holdout3", "holdout4", "holdout5"])
    ap.add_argument("--abstain-below", type=float, default=0.6)
    ap.add_argument("-v", "--verbose", action="count", default=0)
    a = ap.parse_args()
    if a.pool:
        pool(a.verbose)
    if a.split:
        bench(a.split, a.abstain_below)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
