"""List the namer's misses on a v2 split with the engine's ranking (development aid; never on the hold-out)."""

from __future__ import annotations

import sys
import warnings

from spike2 import corpus, score, tasks

from .evaluate import answer


def main() -> int:
    warnings.filterwarnings("ignore")
    sys.stdout.reconfigure(encoding="utf-8")
    split = sys.argv[1] if len(sys.argv) > 1 else "test"
    assert split in ("dev", "test", "holdout", "holdout2", "holdout3", "holdout4", "holdout5", "holdout6"), "development (spent) splits only"
    for t in tasks.load(split):
        if t["bucket"] not in ("main", "decoy"):
            continue
        x, dtype, kv, kc = tasks.hidden_and_known(t)
        a = answer(x, dtype, list(kc.values()), t["technique"], tasks.task_vocabulary(t),
                   exclude_group=corpus.source_group(t["file"]))
        r = {"status": "ok", **a}
        if t["bucket"] == "main" and score.correct(t, r):
            continue
        if t["bucket"] == "decoy" and not score.hallucinated(t, r):
            continue
        truth = t["truth"]["canonical"] if t["truth"] else f"DECOY {t['decoy']['kind']} {t['decoy'].get('source_name', '')}"
        print(f"\n{t['task_id']} {t['file']} ({t['technique']}) truth={truth} -> {a['verdict']} {a['name']}")
        print(f"   known: {', '.join(c.name for c in kc.values())}")
        print(f"   evidence: {a.get('evidence', '')[:200]}")
        for row in a.get("ranking", [])[:4]:
            print(f"   {row}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
