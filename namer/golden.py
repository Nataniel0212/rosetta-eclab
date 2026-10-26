"""Regression safety net: snapshot every answer the namer gives, and compare later.

  python -m namer.golden save      # writes tests/golden/answers.jsonl (pool + every split)
  python -m namer.golden check     # recomputes and lists every difference (exit 1 if any)

A refactoring (e.g. replacing where the vocabulary or storage types come from) must
leave every answer identical: verdict, name, confidence and alternatives.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

from spike2 import corpus, features, tasks, vocab

from .evaluate import answer, test_keys

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "tests" / "golden" / "answers.jsonl"
SPLITS = ["dev", "test", "holdout", "holdout2", "holdout3", "holdout4", "holdout5", "holdout6"]
KEYS = ("verdict", "name", "confidence", "alternatives", "duplicate_of")


def compute() -> list[dict]:
    warnings.filterwarnings("ignore")
    out = []
    excluded = test_keys()
    V = vocab.names()
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
            a = answer(c.values, c.dtype, others, fd.technique, V, exclude_group=corpus.source_group(p.name))
            out.append({"id": f"pool:{p.name}:{c.position}", **{k: a.get(k) for k in KEYS}})
    for split in SPLITS:
        for t in tasks.load(split):
            x, dtype, kv, kc = tasks.hidden_and_known(t)
            a = answer(x, dtype, list(kc.values()), t["technique"], tasks.task_vocabulary(t),
                       exclude_group=corpus.source_group(t["file"]))
            out.append({"id": f"{split}:{t['task_id']}", **{k: a.get(k) for k in KEYS}})
    return out


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    cmd = sys.argv[1] if len(sys.argv) > 1 else "check"
    rows = compute()
    if cmd == "save":
        GOLDEN.parent.mkdir(parents=True, exist_ok=True)
        GOLDEN.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8", newline="\n")
        print(f"saved {len(rows)} answers -> {GOLDEN}")
        return 0
    old = {r["id"]: r for r in (json.loads(x) for x in GOLDEN.read_text(encoding="utf-8").splitlines() if x.strip())}
    new = {r["id"]: r for r in rows}
    diffs = [(k, old.get(k), new.get(k)) for k in sorted(set(old) | set(new)) if old.get(k) != new.get(k)]
    for k, a, b in diffs[:50]:
        print(f"DIFF {k}\n   before: {a}\n   after:  {b}")
    print(f"{len(new)} answers compared, {len(diffs)} differences")
    return 1 if diffs else 0


if __name__ == "__main__":
    raise SystemExit(main())
