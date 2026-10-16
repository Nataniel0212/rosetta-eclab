"""Show how the library scores pool columns of one name (development aid).

  python -m namer.debug "Re(Z)" [--n 4]
"""

from __future__ import annotations

import argparse
import sys
import warnings

import numpy as np

from spike2 import corpus, features, tasks, vocab

from .engine import Context, Known, fit
from .evaluate import test_keys
from .library import LIBRARY


def main() -> int:
    warnings.filterwarnings("ignore")
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--fails", action="store_true", help="only show columns the pipeline gets wrong")
    ap.add_argument("--probe", help="python expression over c (Context) to fit against the target, e.g. 'c.col(\"I\")'")
    a = ap.parse_args()
    excluded = test_keys()
    shown = 0
    for p in corpus.corpus_files():
        try:
            fd = tasks.file_data(p.name)
        except Exception:
            continue
        if fd.n_points < tasks.MIN_ROWS:
            continue
        cols = tasks.usable_columns(fd)
        for c in cols:
            if vocab.canonical(c.name) != a.name or (p.name, c.position) in excluded or features.is_constant(c.values):
                continue
            others = [o for o in cols if o is not c]
            if any(features.identical(c.values, o.values) for o in others):
                continue  # duplicates are not part of the pool
            if a.fails:
                from .evaluate import answer
                ans = answer(c.values, c.dtype, others, fd.technique, vocab.names(), exclude_group=corpus.source_group(p.name))
                if ans["verdict"] == "NAME" and ans["name"] == a.name:
                    continue
                print(f"\n>> pipeline answer: {ans['verdict']} {ans.get('name')} | {ans.get('evidence', '')[:160]}")
            known = {}
            for o in others:
                known.setdefault(vocab.canonical(o.name), Known(vocab.canonical(o.name), o.unit, np.asarray(o.values)))
            ctx = Context(fd.technique, known, np.asarray(c.values), c.dtype)
            y = np.asarray(c.values, dtype="f8")
            print(f"\n== {p.name} ({fd.technique}) {c.name} [{c.unit}] {c.dtype}: min {y.min():.4g} max {y.max():.4g} "
                  f"distinct {np.unique(y).size} first {y[:6].round(5).tolist()}")
            print("   known:", ", ".join(f"{k} [{v.unit}]" for k, v in known.items()))
            ranking = []
            for cand, spec in LIBRARY.items():
                if cand in known:
                    continue
                ev = spec.evaluate(ctx)
                if ev:
                    ranking.append((ev.score, cand, ev.detail))
            ranking.sort(reverse=True)
            for s, cand, d in ranking[:5]:
                print(f"   {s:.3f} {cand}: {d}")
            own = LIBRARY.get(a.name)
            if own:
                ev = own.evaluate(ctx)
                print(f"   OWN: {ev.score if ev else None} {ev.detail if ev else 'no relation testable'}")
            if a.probe:
                with np.errstate(all="ignore"):
                    f = fit(y, eval(a.probe, {"np": np, "c": ctx, **vars(sys.modules[f'{__package__}.library'])}))
                print(f"   PROBE {a.probe}: {f}")
            shown += 1
            if shown >= a.n:
                return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
