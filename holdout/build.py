"""Build and freeze the hold-out benchmark from files nobody developed on.

Same task design as spike v2 (spike2/tasks.py): one real column hidden per task,
code-written dossier, known columns under shuffled aliases, one sham vocabulary
hole per task, out-of-vocabulary and synthetic decoys. Answer key: yadg 7.0.1's
names, cross-checked against the vendor's own text exports where a pair exists.

  python -m holdout.build [holdout|holdout2]   # writes data/<set>/tasks.jsonl and holdout/manifest*.json (commit it)

Rule: nothing in namer/ may be changed after looking at hold-out results; the
hold-out is run once per arm, and every run is listed in results/open_runs.md.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import warnings
from collections import Counter, defaultdict
from pathlib import Path

from spike2 import corpus, elim, features, tasks, vocab

import sys

ROOT = Path(__file__).resolve().parents[1]
SET = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else "holdout"
DATA = ROOT / "data" / SET
OUT = DATA / "tasks.jsonl"
MANIFEST = ROOT / "holdout" / ("manifest.json" if SET == "holdout" else f"manifest_{SET}.json")
SEED = {"holdout": 20261016, "holdout2": 20261017, "holdout3": 20261018, "holdout4": 20261019, "holdout5": 20261020, "holdout6": 20261027, "holdout7": 20261028}[SET]
PREFIX = {"holdout": "H", "holdout2": "K", "holdout3": "M", "holdout4": "N", "holdout5": "P", "holdout6": "Q", "holdout7": "R"}[SET]
PER_NAME = 4
DECOYS = ["noise_f4"] * 6 + ["noise_f8"] * 2 + ["randint_u2"] * 3 + ["oov"] * 8 + ["const", "copy"]


def export_header(mpr: Path) -> list[str] | None:
    """Column names from the vendor text export next to the .mpr (same stem, .txt or .mpt), if any."""
    for ext in (".mpt", ".txt"):
        cand = [q for q in mpr.parent.glob("*" + ext) if q.stem.replace(" ", "").lower()[-25:] == mpr.stem.replace(" ", "").lower()[-25:]]
        for q in cand:
            text = q.read_text(encoding="latin-1", errors="replace")
            m = re.search(r"Nb header lines\s*:\s*(\d+)", text)
            lines = text.splitlines()
            head = lines[int(m.group(1)) - 1] if m else lines[0]
            return [h.strip() for h in head.split("\t") if h.strip()]
    return None


def build() -> list[dict]:
    warnings.filterwarnings("ignore")
    rng = random.Random(SEED)
    files = sorted(str(p.relative_to(ROOT / "data")).replace("\\", "/") for p in DATA.rglob("*.mpr"))
    def ok(f):
        try:
            return tasks.file_data(f).n_points >= tasks.MIN_ROWS
        except Exception:
            return False  # e.g. no data points

    readable = [f for f in files if ok(f)]
    V = vocab.names()
    tech_names: dict = defaultdict(set)
    for f in readable:
        tech_names[tasks.file_data(f).technique] |= {vocab.canonical(c.name) for c in tasks.usable_columns(tasks.file_data(f))}
    all_names = set().union(*tech_names.values())
    cands, sham = [], {}
    for f in readable:
        cols = tasks.usable_columns(tasks.file_data(f))
        for c in cols:
            present = {vocab.canonical(o.name) for o in cols if o is not c}
            hole = tasks.sham_hole(f, f"col{c.position}", present, vocab.canonical(c.name), tech_names, all_names)
            sham[(f, c.position)] = hole
            if features.is_constant(c.values):
                b = "constant"
            elif any(features.identical(c.values, o.values) for o in cols if o is not c):
                b = "duplicate"
            elif elim.eliminate([o.name for o in cols if o is not c], [n for n in V if n != hole]):
                b = "elim"
            else:
                b = "main"
            cands.append((f, c, b))
    by_name = defaultdict(list)
    for f, c, b in cands:
        if b in ("main", "elim"):
            by_name[vocab.canonical(c.name)].append((f, c, b))
    chosen, leftover = [], []
    for canon in sorted(by_name):
        g = by_name[canon]
        rng.shuffle(g)
        pick, used = [], set()
        for fcb in g:
            if len(pick) < PER_NAME and fcb[0] not in used:
                pick.append(fcb)
                used.add(fcb[0])
        for fcb in g:
            if len(pick) < PER_NAME and fcb not in pick:
                pick.append(fcb)
        chosen += pick
        leftover += [fcb for fcb in g if fcb not in pick and fcb[2] == "main"]
    dups = [x for x in cands if x[2] == "duplicate"]
    consts = [x for x in cands if x[2] == "constant"]
    chosen += rng.sample(dups, min(6, len(dups))) + rng.sample(consts, min(6, len(consts)))

    out = []

    def add(f, c, bucket, decoy=None, exclude=None):
        fd = tasks.file_data(f)
        known = [k.position for k in tasks.usable_columns(fd) if c is None or k.position != c.position]
        t = {"task_id": None, "split": SET, "bucket": bucket, "file": f, "technique": fd.technique,
             "position": None if c is None else c.position,
             "truth": None if (c is None or bucket == "decoy") else {
                 "name": c.name, "canonical": vocab.canonical(c.name), "unit": c.unit, "raw_id": c.raw_id,
                 "galvani_unknown": tasks.is_drift(f, c)},
             "decoy": decoy, "vocab_exclude": exclude or [], "aliases": tasks._aliases(known, rng)}
        x, _, kv, _ = tasks.hidden_and_known(t)
        t["identical_to"] = [a for a, v in kv.items() if features.identical(x, v)]
        out.append(t)

    for f, c, b in chosen:
        add(f, c, b, exclude=[sham[(f, c.position)]])
    rng.shuffle(leftover)
    used_oov = set()
    for kind in DECOYS:
        if kind == "oov":
            pick = next((x for x in leftover if vocab.canonical(x[1].name) not in used_oov), None)
            if pick is None:  # fewer distinct names than out-of-vocabulary decoys (hold-out 6 has five)
                continue
            f, c, _ = pick
            used_oov.add(vocab.canonical(c.name))
            add(f, c, "decoy", decoy={"kind": "oov", "source_position": c.position, "source_name": c.name},
                exclude=[vocab.canonical(c.name)])
            continue
        host = rng.choice(readable)
        seed = rng.randrange(1 << 30)
        _, _, meta = tasks.make_decoy(kind, tasks.file_data(host), seed)
        present = {vocab.canonical(o.name) for o in tasks.usable_columns(tasks.file_data(host))}
        add(host, None, "decoy", decoy={"seed": seed, **meta},
            exclude=[tasks.sham_hole(host, f"decoy{seed}", present, None, tech_names, all_names)])
    rng.shuffle(out)
    for i, t in enumerate(out, 1):
        t["task_id"] = f"{PREFIX}{i:03d}"
        t["dossier"] = tasks.dossier(t)
        t["dossier_sha256"] = hashlib.sha256(t["dossier"].encode("utf-8")).hexdigest()
    return out


def label_check(task_list: list[dict]) -> dict:
    """How often yadg's name for a hidden column appears in the vendor export header of the same file."""
    agree, total, missing = 0, 0, Counter()
    for t in task_list:
        if not t["truth"]:
            continue
        head = export_header(ROOT / "data" / t["file"])
        if head is None:
            continue
        total += 1
        nm = t["truth"]["name"]
        forms = {nm, nm.replace(" or ", "/"), vocab.canonical(nm)} | {a for a, c in vocab.ALIAS.items() if c == vocab.canonical(nm)}
        heads = {h.split("/")[0].strip() for h in head} | {h.rsplit("/", 1)[0].strip() for h in head} | set(head)
        if forms & heads:
            agree += 1
        else:
            missing[nm] += 1
    return {"tasks_with_export": total, "name_found_in_export": agree, "not_found": dict(missing)}


def main() -> int:
    t = build()
    OUT.write_text("\n".join(json.dumps(x, ensure_ascii=False, default=str) for x in t) + "\n", encoding="utf-8", newline="\n")
    counts = Counter(x["bucket"] + (f":{x['decoy']['kind']}" if x["decoy"] else "") for x in t)
    m = {"seed": SEED, "n_tasks": len(t), "counts": dict(sorted(counts.items())),
         "n_main_names": len({x["truth"]["canonical"] for x in t if x["bucket"] == "main"}),
         "tasks_jsonl_sha256": hashlib.sha256(OUT.read_bytes()).hexdigest(),
         "files_manifest_sha256": hashlib.sha256((DATA / "manifest.json").read_bytes()).hexdigest(),
         "label_check": label_check(t),
         "tasks": [{k: x[k] for k in ("task_id", "bucket", "file", "technique", "position", "truth", "decoy",
                                      "vocab_exclude", "identical_to", "dossier_sha256")} for x in t]}
    MANIFEST.parent.mkdir(exist_ok=True)
    MANIFEST.write_text(json.dumps(m, indent=1, ensure_ascii=False, default=str), encoding="utf-8", newline="\n")
    print(json.dumps({k: v for k, v in m.items() if k != "tasks"}, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
