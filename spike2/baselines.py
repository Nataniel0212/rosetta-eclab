"""Code-only baselines for the naming benchmark (no model).

All use the same code rules the model is shown: a constant column -> ABSTAIN,
a column identical to a known one -> DUPLICATE. All respect the task's
vocabulary (out-of-vocabulary decoys remove one name). Otherwise:

  H_prior  technique prior: the canonical name that occurs in the largest share of
           files with the same technique code outside this file's source group,
           among names not already present in this file (global frequency as fallback).
  H_knn    k-nearest neighbours (k=5, distance-weighted) on scale-free column
           features, trained on every usable column of the files outside this
           file's source group (leave-one-group-out), excluding names already
           present in this file. A supervised lookup: it can only name columns
           it has seen labelled in other files.
  H_elim   name elimination (elim.py); no answer -> ABSTAIN. Settles the 'elim' bucket.
  H_combo  H_elim if it answers, else H_knn.

Source groups (corpus.source_group): files from the same GitHub issue or PR,
the VSP files and the galvani regression files are one group each, so
near-copies of the same experiment cannot leak labels.

The decision rule compares each model arm with the best of these on the main
bucket (where H_elim never answers, so H_combo equals H_knn there).

Usage:  python -m spike2.baselines --split test     # writes runs2/baselines_<split>.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from . import corpus, elim, features, tasks, vocab

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs2"
PREREG = ROOT / "prereg2"


def _training_rows():
    """(source group, technique, canonical name, feature vector) for every usable non-constant column."""
    rows = []
    for p in corpus.corpus_files():
        try:
            fd = tasks.file_data(p.name)
        except Exception:
            continue
        if fd.n_points < tasks.MIN_ROWS:
            continue
        cols = tasks.usable_columns(fd)
        tcol = next((c for c in cols if c.name == "time"), None)
        for c in cols:
            if features.is_constant(c.values):
                continue
            others = [o.values for o in cols if o is not c]
            f = features.knn_features(c.values, c.dtype, None if (tcol is None or tcol is c) else tcol.values, others)
            rows.append((corpus.source_group(p.name), fd.technique, vocab.canonical(c.name), f))
    return rows


def _file_names():
    out = {}
    for p in corpus.corpus_files():
        try:
            fd = tasks.file_data(p.name)
        except Exception:
            continue
        if fd.n_points >= tasks.MIN_ROWS:
            out[p.name] = (corpus.source_group(p.name), fd.technique,
                           {vocab.canonical(c.name) for c in tasks.usable_columns(fd)})
    return out


def code_rule(task: dict, x) -> dict | None:
    if features.is_constant(x):
        return {"status": "ok", "verdict": vocab.ABSTAIN, "name": "NONE", "rule": "constant"}
    if task["identical_to"]:
        return {"status": "ok", "verdict": vocab.DUPLICATE, "name": "NONE", "duplicate_of": task["identical_to"][0],
                "rule": "identical"}
    return None


def run(split: str) -> list[dict]:
    rows = _training_rows()
    X = np.stack([r[3] for r in rows])
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = (X - mu) / sd
    files = _file_names()
    out = []
    for t in tasks.load(split):
        x, dtype, kv, kc = tasks.hidden_and_known(t)
        allowed = set(tasks.task_vocabulary(t))
        present = {vocab.canonical(c.name) for c in kc.values()}
        tech = t["technique"]
        grp = corpus.source_group(t["file"])
        res = {"task_id": t["task_id"]}
        rule = code_rule(t, x)
        if rule:
            for arm in ("H_prior", "H_knn", "H_elim", "H_combo"):
                res[arm] = rule
            out.append(res)
            continue
        # --- H_prior
        others = [(tq, names) for f, (g, tq, names) in files.items() if g != grp]
        ok_name = lambda n: n not in present and n in allowed  # noqa: E731
        cnt = Counter(n for tq, names in others if tq == tech for n in names if ok_name(n))
        if not cnt:
            cnt = Counter(n for _, names in others for n in names if ok_name(n))
        best = sorted(cnt.items(), key=lambda kv_: (-kv_[1], kv_[0]))[:3]  # deterministic tie-break
        res["H_prior"] = {"status": "ok", "verdict": "NAME", "name": best[0][0], "alternatives": [b[0] for b in best[1:]]}
        # --- H_knn
        tcol = next((v for a, v in kv.items() if kc[a].name == "time"), None)
        f = (features.knn_features(x, dtype, tcol, list(kv.values())) - mu) / sd
        idx = [i for i, r in enumerate(rows) if r[0] != grp and ok_name(r[2])]
        d = np.linalg.norm(Z[idx] - f, axis=1)
        votes = defaultdict(float)
        for j in np.argsort(d)[:5]:
            votes[rows[idx[j]][2]] += 1.0 / (d[j] + 1e-6)
        ranked = sorted(votes, key=lambda k: (-votes[k], k))
        res["H_knn"] = {"status": "ok", "verdict": "NAME", "name": ranked[0], "alternatives": ranked[1:3]}
        # --- H_elim and H_combo
        e = elim.eliminate([c.name for c in kc.values()], [n for n in vocab.names() if n in allowed])
        res["H_elim"] = ({"status": "ok", "verdict": "NAME", "name": e, "alternatives": []} if e else
                         {"status": "ok", "verdict": vocab.ABSTAIN, "name": "NONE", "rule": "no unique gap"})
        res["H_combo"] = res["H_elim"] if e else res["H_knn"]
        out.append(res)
    RUNS.mkdir(exist_ok=True)
    path = RUNS / f"baselines_{split}.jsonl"
    text = "\n".join(json.dumps(r, ensure_ascii=False) for r in out) + "\n"
    path.write_text(text, encoding="utf-8", newline="\n")
    if split == "test":  # the baseline answers are part of the pre-registration
        (PREREG / "baselines_test.jsonl").write_text(text, encoding="utf-8", newline="\n")
    return out


def main() -> int:
    import warnings

    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["dev", "test", "holdout", "holdout2", "holdout3", "holdout4", "holdout5", "holdout6"])
    a = ap.parse_args()
    res = run(a.split)
    print(f"{len(res)} tasks -> runs2/baselines_{a.split}.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
