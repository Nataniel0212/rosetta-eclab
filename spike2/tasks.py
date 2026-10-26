"""Build the v2 benchmark: leave-one-column-out tasks, decoys, dev/test split, dossiers.

Each task hides one column of a real EC-Lab file. The model gets a dossier
that code wrote (technique, the hidden column's statistics, generic relations
to the known columns, a sample table) and must name the hidden column TARGET
from the closed vocabulary, call it a duplicate, or abstain. Known columns
appear in shuffled order under neutral aliases (c0, c1, ...), so neither
column IDs nor file order give the answer away ("hard tier").

Buckets:
  main      non-constant, not identical to another column, not settled by name
            elimination (elim.py): the primary metric
  elim      settled by name elimination alone (e.g. the missing harmonic order): reported only
  duplicate identical to another column in the file: reported (DUPLICATE is right)
  constant  constant column: reported (ABSTAIN is right)
  decoy     ABSTAIN is right; naming it counts as a hallucination. Kinds:
              noise_f4/noise_f8/randint_u2  synthetic columns inserted by code
              oov    a real column whose name is removed from that task's vocabulary
                     (the product's real risk: a quantity the vocabulary does not have)
Every other task also has exactly one name removed from its vocabulary (a "sham hole":
a name typical for the technique but absent from the file), so a missing name is
never by itself a sign of an oov decoy.
              const/copy  synthetic constant or exact copy (code flags them; reported)

Everything is deterministic from SEED. Outputs:
  data/v2/tasks.jsonl            full tasks with dossier text (derived from GPL data, git-ignored)
  prereg2/tasks_manifest.json    task list, answer key and dossier hashes (committed before the run)

Usage:  python -m spike2.tasks
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np

from . import corpus, elim, features, vocab

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "v2"
PREREG = ROOT / "prereg2"

SEED = 20261015
MIN_ROWS = 10
TEST_PER_NAME = 3
DEV_MAIN = 20
TEST_DUP_EXTRA = 6
TEST_CONST = 6
DECOYS_TEST = ["noise_f4"] * 8 + ["noise_f8"] * 2 + ["randint_u2"] * 4 + ["const", "copy"] + ["oov"] * 10
DECOYS_DEV = ["noise_f4", "randint_u2", "oov"]
SCORED_DECOYS = {"noise_f4", "noise_f8", "randint_u2", "oov"}
SAMPLE_ROWS = 12
HIDDEN = "TARGET"


@lru_cache(maxsize=None)
def file_data(name: str) -> corpus.FileData:
    # v2 corpus files are bare names in data/external; other sets (e.g. the hold-out) use paths under data/
    if "/" in name:
        return corpus.decode(ROOT / "data" / name, allow_v0=True)
    return corpus.decode(corpus.EXT / name)


def usable_columns(fd: corpus.FileData) -> list[corpus.Column]:
    return [c for c in fd.columns if not c.name.startswith("duplicate")]


def task_vocabulary(task: dict) -> list[str]:
    excl = set(task.get("vocab_exclude") or [])
    return [n for n in vocab.names() if n not in excl]


def make_decoy(kind: str, fd: corpus.FileData, seed: int) -> tuple[np.ndarray, str, dict]:
    rng = np.random.default_rng(seed)
    n = fd.n_points
    meta: dict = {"kind": kind}
    if kind in ("noise_f4", "noise_f8"):
        scale = 10.0 ** rng.uniform(-4, 3)
        loc = float(rng.choice([0.0, rng.uniform(-5, 5) * scale]))
        v = rng.normal(loc, scale, n)
        dtype = "<f4" if kind == "noise_f4" else "<f8"
        v = v.astype("f4") if dtype == "<f4" else v
        meta.update(scale=scale, loc=loc)
    elif kind == "randint_u2":
        hi = int(rng.choice([16, 256, 4096, 65535]))
        v = rng.integers(0, hi, n).astype("u2")
        dtype = "<u2"
        meta.update(high=hi)
    elif kind == "const":
        val = float(rng.choice([0.0, float(f"{rng.uniform(-10, 10):.4g}")]))
        v = np.full(n, val, dtype="f4")
        dtype = "<f4"
        meta.update(value=val)
    elif kind == "copy":
        cols = [c for c in usable_columns(fd) if not features.is_constant(c.values)]
        src = cols[int(rng.integers(0, len(cols)))]
        v = np.array(src.values, copy=True)
        dtype = src.dtype
        meta.update(source_position=src.position, source_name=src.name)
    else:
        raise ValueError(kind)
    return np.asarray(v), dtype, meta


def hidden_and_known(task: dict) -> tuple[np.ndarray, str, dict[str, np.ndarray], dict[str, corpus.Column]]:
    """Arrays for a task: (hidden values, hidden dtype, {alias: values}, {alias: Column})."""
    fd = file_data(task["file"])
    cols = {c.position: c for c in usable_columns(fd)}
    d = task.get("decoy")
    if d and d["kind"] != "oov":
        x, dtype, _ = make_decoy(d["kind"], fd, d["seed"])
    else:
        h = cols[task["position"]]
        x, dtype = h.values, h.dtype
    known_vals, known_cols = {}, {}
    for alias, pos in task["aliases"].items():
        known_vals[alias] = cols[pos].values
        known_cols[alias] = cols[pos]
    return np.asarray(x), dtype, known_vals, known_cols


def dossier(task: dict) -> str:
    fd = file_data(task["file"])
    x, dtype, kv, kc = hidden_and_known(task)
    time_alias = next((a for a, c in kc.items() if c.name == "time"), None)
    lines = [
        f"Technique code from the file's settings module: {fd.technique or 'unknown (old file version)'}",
        f"Rows: {fd.n_points}",
        "",
        f"Hidden column {HIDDEN}: dtype {dtype} (decoded by code). Its name is what you must find.",
    ]
    const = features.is_constant(x)
    same = [a for a, v in kv.items() if features.identical(x, v)]
    lines.append(f"Code check - constant: {'YES' if const else 'no'}")
    lines.append(f"Code check - identical (all rows) to known column(s): {', '.join(same) if same else 'none'}")
    st = features.stats(x, dtype)
    lines.append(f"Statistics of {HIDDEN}: " + "; ".join(f"{k} {v}" for k, v in st.items()))
    rel = [] if const else features.relations(x, kv, time_alias)
    if rel:
        lines.append(f"Strongest relations found by code's generic search ({HIDDEN} = a*expr + b, least squares):")
        for r in rel:
            lines.append(f"  {r['hypothesis']}: r2 {r['r2']}, a {r['a']}, b {r['b']}, "
                         f"rows within 1% of range {r['fraction_rows_within_1pct_of_range']}")
    else:
        lines.append("Strongest relations found by code's generic search: none with r2 >= 0.5")
    lines += ["", "Known columns (named by the reference parser; shuffled order):"]
    for a, c in kc.items():
        v = features._f8(c.values)
        fin = v[np.isfinite(v)]
        rng = f"min {features.fmt(fin.min())}, max {features.fmt(fin.max())}" if fin.size else "no finite values"
        lines.append(f"  {a}: {c.name} [{c.unit or 'no unit'}], {c.dtype}, {rng}")
    n = fd.n_points
    idx = sorted(set(np.linspace(0, n - 1, min(SAMPLE_ROWS, n)).round().astype(int).tolist()))
    aliases = list(kc)
    lines += ["", f"Sample rows ({len(idx)} evenly spaced rows of {n}):", "row | " + " | ".join(aliases + [HIDDEN])]
    for i in idx:
        vals = [features.fmt(float(kv[a][i])) for a in aliases] + [features.fmt(float(x[i]))]
        lines.append(f"{i} | " + " | ".join(vals))
    return "\n".join(lines)


def sham_hole(f: str, key: str, present: set[str], truth: str | None, tech_names: dict, all_names: set[str]) -> str:
    """One canonical name to leave out of a task's vocabulary: typical for the technique, absent from the file."""
    tech = file_data(f).technique
    pool = sorted(tech_names.get(tech, set()) - present - {truth})
    if not pool:
        pool = sorted(all_names - present - {truth})
    if not pool:  # a set whose files all share one column set (hold-out 6): any other vocabulary name
        pool = sorted(set(vocab.names()) - present - {truth})
    return random.Random(f"{SEED}:{f}:{key}").choice(pool)


def _aliases(known_positions: list[int], rng: random.Random) -> dict[str, int]:
    order = list(known_positions)
    rng.shuffle(order)
    return {f"c{i}": pos for i, pos in enumerate(order)}


def is_drift(f: str, c: corpus.Column) -> bool:
    """A column whose full ID galvani 0.5.0 does not know: a real format drift."""
    return c.raw_id in file_data(f).galvani_unknown_ids


def build() -> list[dict]:
    rng = random.Random(SEED)
    files = [p.name for p in corpus.corpus_files()]
    V = vocab.names()
    candidates = []  # (file, column, bucket)
    readable = []
    for name in files:
        try:
            fd = file_data(name)
        except Exception:
            continue
        if fd.n_points >= MIN_ROWS:
            readable.append(name)
    tech_names: dict = defaultdict(set)
    for name in readable:
        tech_names[file_data(name).technique] |= {vocab.canonical(c.name) for c in usable_columns(file_data(name))}
    all_names = set().union(*tech_names.values())
    sham: dict[tuple[str, int], str] = {}
    for name in readable:
        cols = usable_columns(file_data(name))
        for c in cols:
            present = {vocab.canonical(o.name) for o in cols if o is not c}
            hole = sham_hole(name, f"col{c.position}", present, vocab.canonical(c.name), tech_names, all_names)
            sham[(name, c.position)] = hole
            if features.is_constant(c.values):
                bucket = "constant"
            elif any(features.identical(c.values, o.values) for o in cols if o is not c):
                bucket = "duplicate"
            elif elim.eliminate([o.name for o in cols if o is not c], [n for n in V if n != hole]):
                bucket = "elim"
            else:
                bucket = "main"
            candidates.append((name, c, bucket))
    drift_files = corpus.DRIFT_FILES | {f for f in readable if file_data(f).galvani_unknown_ids}

    by_name = defaultdict(list)
    for f, c, b in candidates:
        if b in ("main", "elim"):
            by_name[vocab.canonical(c.name)].append((f, c, b))
    test, pool = [], []
    for canon in sorted(by_name):
        group = by_name[canon]
        rng.shuffle(group)
        group.sort(key=lambda fcb: not is_drift(fcb[0], fcb[1]))  # drift columns first, order otherwise random
        chosen, used_files = [], set()
        for fcb in group:  # prefer distinct files
            if len(chosen) < TEST_PER_NAME and fcb[0] not in used_files:
                chosen.append(fcb)
                used_files.add(fcb[0])
        for fcb in group:
            if len(chosen) < TEST_PER_NAME and fcb not in chosen:
                chosen.append(fcb)
        test += chosen
        pool += [fcb for fcb in group if fcb not in chosen and fcb[2] == "main" and fcb[0] not in drift_files]
    dev = rng.sample(pool, min(DEV_MAIN, len(pool)))
    oov_pool = [fcb for fcb in pool if fcb not in dev]

    dups = [(f, c, b) for f, c, b in candidates if b == "duplicate"]
    dup_drift = [t for t in dups if is_drift(t[0], t[1])]
    dup_rest = [t for t in dups if t not in dup_drift]
    test += dup_drift + rng.sample(dup_rest, min(TEST_DUP_EXTRA, len(dup_rest)))
    consts = [(f, c, b) for f, c, b in candidates if b == "constant"]
    test += rng.sample(consts, min(TEST_CONST, len(consts)))

    tasks = []

    def add(split, f, c, bucket, decoy=None, vocab_exclude=None):
        fd = file_data(f)
        known = [k.position for k in usable_columns(fd) if c is None or k.position != c.position]
        t = {
            "task_id": None,
            "split": split,
            "bucket": bucket,
            "file": f,
            "technique": fd.technique,
            "position": None if c is None else c.position,
            "truth": None if (c is None or bucket == "decoy") else {
                "name": c.name, "canonical": vocab.canonical(c.name), "unit": c.unit, "raw_id": c.raw_id,
                "galvani_unknown": is_drift(f, c)},
            "decoy": decoy,
            "vocab_exclude": vocab_exclude or [],
            "aliases": _aliases(known, rng),
        }
        x, _, kv, _ = hidden_and_known(t)
        t["identical_to"] = [a for a, v in kv.items() if features.identical(x, v)]
        tasks.append(t)

    for f, c, b in test:
        add("test", f, c, b, vocab_exclude=[sham[(f, c.position)]])
    for f, c, b in dev:
        add("dev", f, c, b, vocab_exclude=[sham[(f, c.position)]])
    used_oov_names = set()
    rng.shuffle(oov_pool)
    for split, kinds in (("test", DECOYS_TEST), ("dev", DECOYS_DEV)):
        for kind in kinds:
            if kind == "oov":
                f, c, _ = next(fcb for fcb in oov_pool if vocab.canonical(fcb[1].name) not in used_oov_names)
                canon = vocab.canonical(c.name)
                used_oov_names.add(canon)
                add(split, f, c, "decoy", decoy={"kind": "oov", "source_position": c.position, "source_name": c.name},
                    vocab_exclude=[canon])
                continue
            host = rng.choice(readable)
            seed = rng.randrange(1 << 30)
            _, _, meta = make_decoy(kind, file_data(host), seed)
            present = {vocab.canonical(o.name) for o in usable_columns(file_data(host))}
            hole = sham_hole(host, f"decoy{seed}", present, None, tech_names, all_names)
            add(split, host, None, "decoy", decoy={"seed": seed, **meta}, vocab_exclude=[hole])

    names = set(V)
    missing = sorted({t["truth"]["canonical"] for t in tasks if t["truth"]} - names)
    if missing:
        raise SystemExit(f"answer-key names missing from the vocabulary: {missing}")
    rng.shuffle(tasks)
    tasks.sort(key=lambda t: t["split"] != "dev")
    for i, t in enumerate(tasks, 1):
        t["task_id"] = f"{'D' if t['split'] == 'dev' else 'T'}{i:03d}"
        t["dossier"] = dossier(t)
        t["dossier_sha256"] = hashlib.sha256(t["dossier"].encode("utf-8")).hexdigest()
    return tasks


def manifest(tasks: list[dict]) -> dict:
    allsha = hashlib.sha256("".join(t["dossier_sha256"] for t in tasks).encode()).hexdigest()
    rows = []
    for t in tasks:
        r = {k: t[k] for k in ("task_id", "split", "bucket", "file", "technique", "position", "truth", "decoy",
                               "vocab_exclude", "identical_to", "dossier_sha256")}
        rows.append(r)
    counts = defaultdict(int)
    for t in tasks:
        k = t["bucket"] + (f":{t['decoy']['kind']}" if t["decoy"] else "")
        counts[f"{t['split']}/{k}"] += 1
    return {"seed": SEED, "n_tasks": len(tasks), "counts": dict(sorted(counts.items())),
            "all_dossiers_sha256": allsha, "vocabulary_size": len(vocab.vocabulary()),
            "vocabulary_sha256": hashlib.sha256(vocab.render().encode("utf-8")).hexdigest(), "tasks": rows}


def tasks_file() -> Path:
    return OUT / "tasks.jsonl"


def load(split: str | None = None) -> list[dict]:
    path = (ROOT / "data" / split / "tasks.jsonl") if (split or "").startswith("holdout") else tasks_file()
    tasks = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [t for t in tasks if split is None or t["split"] == split]


def main() -> int:
    import warnings

    warnings.filterwarnings("ignore")
    tasks = build()
    OUT.mkdir(parents=True, exist_ok=True)
    with tasks_file().open("w", encoding="utf-8", newline="\n") as f:
        for t in tasks:
            f.write(json.dumps(t, ensure_ascii=False, default=str) + "\n")
    PREREG.mkdir(exist_ok=True)
    m = manifest(tasks)
    m["tasks_jsonl_sha256"] = hashlib.sha256(tasks_file().read_bytes()).hexdigest()
    (PREREG / "tasks_manifest.json").write_text(json.dumps(m, indent=1, ensure_ascii=False, default=str),
                                                encoding="utf-8", newline="\n")
    print(json.dumps({k: v for k, v in m.items() if k != "tasks"}, indent=1))
    lens = [len(t["dossier"]) for t in tasks]
    print(f"dossier chars: mean {np.mean(lens):.0f}, max {max(lens)}, total {sum(lens)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
