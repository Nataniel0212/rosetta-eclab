"""Evaluate the standalone namer.

  python -m namer.evaluate --pool            # development pool: every corpus column NOT used by the v2 test split
  python -m namer.evaluate --split dev       # v2 dev tasks  -> runs3/rules_dev.jsonl  (score with bench.claude_cli --score)
  python -m namer.evaluate --split test      # v2 test tasks -> runs3/rules_test.jsonl (run rarely; record in results/open_runs.md)

Pipeline per column: constant -> ABSTAIN; identical to a known column -> DUPLICATE;
name elimination (spike2/elim.py, only when the column is the only one missing; with several unknown columns only for
the one that fits a numbered gap's siblings, gap_column; in --self-check never a gap of the whole file) -> NAME;
otherwise the formula library (namer/library.py).
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
from .library import LIBRARY, current_missing, ece_ruled_out, missing_current

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
    if name == "(Q-Qo)" and ctx.alone and not {"I", "dq", "Pwe"} & set(ctx.known):
        return False  # EC-Lab computes it from the current: all 694 public files with (Q-Qo) also hold I, dq or Pwe
    if name == "Ece" and ece_ruled_out(ctx):
        return False
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


def _scale(values) -> tuple[float, float]:
    """Median magnitude and share of negative values, over the finite nonzero entries."""
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x) & (x != 0)]
    return (float(np.median(np.abs(x))), float(np.mean(x < 0))) if x.size else (0.0, 0.0)


# The members of one harmonic family agree within a factor 48.5 in public files (median 1.9). Siblings further apart
# than this are not one family's scale: in recoded files a misread split had moved another column among them.
SIBLING_SPREAD = 100


def fits_gap(gap: str, values, dtype: str, siblings: list) -> bool:
    """Could this column be the gap's own column? `siblings` are the known columns that differ from the gap only in a
    number (elim.numbered_siblings); at least two differently named ones with data, whose median magnitudes agree
    within SIBLING_SPREAD, set the scale. The column must have the storage type EC-Lab uses for the gap's quantity, a
    median magnitude within a factor 10 of the siblings' range, and at most 1 percentage point more negative values
    than the sibling with the most (a harmonic is a magnitude, so normally none)."""
    from .library import DTYPES

    sib = [(vocab.canonical(c.name), _scale(c.values)) for c in siblings]
    sib = [(n, s) for n, s in sib if s[0] > 0]  # the siblings with data
    if len({n for n, _ in sib}) < 2 or (DTYPES.get(gap) and dtype not in DTYPES[gap]):
        return False
    med, neg = _scale(values)
    lo, hi = min(s[0] for _, s in sib), max(s[0] for _, s in sib)
    if hi > SIBLING_SPREAD * lo:
        return False
    return med > 0 and lo / 10 <= med <= hi * 10 and neg <= max(s[1] for _, s in sib) + 0.01


def gap_column(known_cols: list, unknown_cols: list, vocabulary: list[str], alternatives: list = (),
               doubtful: bool = False) -> tuple[int, str, str] | None:
    """With several unknown columns: which of them fills the file's sibling gap, as (index, name, evidence), or None.

    The gap is the name that elimination finds (elim.eliminate). It must be the only hole in the file's numbered
    families (elim.numbered_gaps), so none of the other unknown columns is one of its siblings. Exactly one unknown
    column may then fit the gap's numbered siblings (fits_gap), and it must not be constant or a copy of a known
    column; constant columns and copies still count as rivals. The check trusts the reader's split of the record into
    columns, so it is repeated on every split the reader found nearly as plausible (mprfile.MprData.alternatives): if
    another column fits in any of them, the data do not say which column is the gap's. A split the reader marks as
    doubtful (another split reads more known columns as their own type) gives no name at all."""
    names = [c.name for c in known_cols]
    gap = elim.eliminate(names, vocabulary)
    if doubtful or not gap or elim.numbered_gaps(names, vocabulary) != {gap}:
        return None

    def fitting(known: list, unknown: list) -> tuple[list[int], list]:
        sib_names = elim.numbered_siblings(gap, [c.name for c in known])
        sibs = [c for c in known if c.name in sib_names]
        return [i for i, u in enumerate(unknown) if fits_gap(gap, u.values, u.dtype, sibs)], sibs

    fit, sibs = fitting(known_cols, unknown_cols)
    if len(fit) != 1:
        return None
    col = unknown_cols[fit[0]]
    if features.is_constant(col.values) or any(features.identical(col.values, c.values) for c in known_cols):
        return None
    for cols in alternatives:
        unknown = [c for c in cols if not c.known]
        if any(unknown[i].position != col.position for i in fitting([c for c in cols if c.known], unknown)[0]):
            return None
    shown = list(dict.fromkeys(c.name for c in sibs if _scale(c.values)[0] > 0))
    return fit[0], gap, (f"name elimination (sibling gap): the only one of the {len(unknown_cols)} unknown columns "
                         f"with the storage type, sign and scale of {', '.join(shown)}")


def stats() -> Stats:
    global _STATS
    if _STATS is None:
        _STATS = Stats()
    return _STATS


def answer(target, dtype, known_cols: list, technique, vocabulary: list[str], abstain_below: float = 0.6,
           exclude_group: str | None = None, noise_autocorr: float = 0.3, fallback_min_share: float = 0.5,
           alone: bool = True, file_gaps: set[str] | None = None, gap: tuple[str, str] | None = None) -> dict:
    """Name one column from the file's known columns. `alone` is False when other columns of the record are missing
    from known_cols too (several unknown columns, or --self-check on a file with unknown columns). In --self-check,
    `file_gaps` holds the sibling gaps of the whole file (elim.gaps, empty if none): no hidden known column can fill
    one, and any other gap is the hidden column's own slot, so name elimination still runs there. With several
    unknown columns the file-level check gap_column decides which one fills the sibling gap, and `gap` = (name,
    evidence) is passed for that column."""
    if features.is_constant(target):
        return {"verdict": "ABSTAIN", "name": "NONE", "confidence": 1.0, "evidence": "constant column"}
    same = [c.name for c in known_cols if features.identical(target, c.values)]
    if same:
        return {"verdict": "DUPLICATE", "name": "NONE", "duplicate_of": same[0], "confidence": 1.0,
                "evidence": f"identical to {same[0]}"}
    if gap:
        return {"verdict": "NAME", "name": gap[0], "confidence": 0.9, "evidence": gap[1]}
    # A sibling gap names the column only when it is the column's own slot: with several unknown columns every one
    # of them would get the gap's name (public files: |Ece h7| for THD/NSD/NSR Ece and the energy columns). There
    # gap_column picks the one that fits, at file level.
    e = elim.eliminate([c.name for c in known_cols], vocabulary) if alone or file_gaps is not None else None
    if e and e not in (file_gaps or ()):
        return {"verdict": "NAME", "name": e, "confidence": 0.9, "evidence": "name elimination (sibling gap)"}
    known = {}
    for c in known_cols:
        known.setdefault(vocab.canonical(c.name), Known(vocab.canonical(c.name), c.unit, np.asarray(c.values), c.dtype))
    ctx = Context(technique, known, np.asarray(target), dtype, alone)
    pr = stats().prior(technique, exclude_group)
    gl = stats().prior(None, exclude_group, all_techniques=True)
    top, gtop = (max(pr.values()) if pr else 0), (max(gl.values()) if gl else 0)
    prior = {k: (pr.get(k, 0) / top if top else 0) + 0.3 * (gl.get(k, 0) / gtop if gtop else 0) for k in set(pr) | set(gl)}
    a = name_column(ctx, LIBRARY, vocabulary, abstain_below, prior,
                    cooccur=lambda cand, sib: stats().cooccur(cand, sib, exclude_group))
    if current_missing(ctx) and missing_current(ctx) > 0 and not (a["verdict"] == "NAME" and a["name"] == "I"):
        # The file's measured current is the one missing column and this column can be it (missing_current), so
        # another name would likely be wrong. Here I was not allowed (an out-of-vocabulary current) or lost.
        ranking = a.get("ranking") or []
        return {"verdict": "ABSTAIN", "name": "NONE", "confidence": ranking[0][1] if ranking else 0.0,
                "alternatives": [r[0] for r in ranking[:2]], "ranking": ranking,
                "evidence": "files like this one hold a measured current and none is named, so the column is most "
                            "likely the current, but I does not fit or is not an allowed name"}
    if a["verdict"] == "NAME":
        return a
    # (Tried 2026-10-15: naming "the column that similar files always record and this file lacks". It names
    # decoys whenever a file naturally lacks a common column: 10/19 decoys on hold-out 4, still 6/19 when limited
    # to structured columns. Trustworthiness wins: not used. The one exception since 2026-10-27 is the measured
    # current, in the two kinds of record that hold it in all public files but one: library.missing_current.)
    # No formula or shape identifies the column: statistics fallback, unless it looks like noise.
    ac = lag1_autocorr(target)
    if ac < noise_autocorr:
        a["evidence"] = f"no formula fits and the column looks like noise (lag-1 autocorrelation {ac:.2f})"
        return a
    name, share, alts = stats().predict(target, dtype, known_cols, technique, set(vocabulary), set(known),
                                        exclude_group)
    if name and not consistent(name, ctx):
        a["evidence"] = f"statistics suggest {name}, but that contradicts its form, storage type or the file's other columns"
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


def sweep(split: str, label: str = "sweep") -> None:
    """Every usable column of every readable file in data/<split>, hidden in turn with the file's other columns
    known and the whole vocabulary allowed (constant and duplicated columns are skipped, as in --pool).
    Writes runs3/<label>_<split>.jsonl and prints correct, wrong and abstained per name. Like an official
    hold-out run it refuses to overwrite an existing result."""
    RUNS.mkdir(exist_ok=True)
    path = RUNS / f"{label}_{split}.jsonl"
    if path.exists():
        raise SystemExit(f"{path} exists; pick another --label")
    V = vocab.names()
    root = ROOT / "data" / split
    by_name, rows = defaultdict(Counter), []
    for p in sorted(root.rglob("*.mpr")):
        rel = p.relative_to(ROOT / "data").as_posix()
        try:
            fd = tasks.file_data(rel)
        except Exception:
            continue
        if fd.n_points < tasks.MIN_ROWS:
            continue
        cols = tasks.usable_columns(fd)
        for c in cols:
            if features.is_constant(c.values):
                continue
            others = [o for o in cols if o is not c]
            if any(features.identical(c.values, o.values) for o in others):
                continue
            truth = vocab.canonical(c.name)
            a = answer(c.values, c.dtype, others, fd.technique, V, exclude_group=corpus.source_group(p.name))
            kind = "correct" if a["verdict"] == "NAME" and a["name"] == truth else (
                "wrong" if a["verdict"] == "NAME" else "abstained")
            by_name[truth][kind] += 1
            rows.append({"file": rel, "position": c.position, "truth": truth, "kind": kind, **a})
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False, default=str) for r in rows) + "\n",
                    encoding="utf-8", newline="\n")
    tot = Counter()
    for c in by_name.values():
        tot.update(c)
    n = sum(tot.values())
    macro = np.mean([c["correct"] / sum(c.values()) for c in by_name.values()]) if by_name else 0.0
    print(f"{split} sweep: {n} hidden columns in {len({r['file'] for r in rows})} files, {len(by_name)} names; "
          f"correct {tot['correct']} ({tot['correct'] / max(1, n):.3f}), wrong {tot['wrong']}, "
          f"abstained {tot['abstained']}; macro {macro:.3f} -> {path}")
    for name, c in sorted(by_name.items(), key=lambda kv: (-kv[1]["wrong"], kv[0])):
        if c["wrong"] or c["abstained"]:
            print(f"  {name:28s} correct {c['correct']:4d}  wrong {c['wrong']:3d}  abstained {c['abstained']:4d}")


def bench(split: str, abstain_below: float, label: str = "rules") -> None:
    RUNS.mkdir(exist_ok=True)
    path = RUNS / f"{label}_{split}.jsonl"
    if split.startswith("holdout") and label in ("rules", "sweep") and path.exists():
        raise SystemExit(f"{path} holds the official {'run' if label == 'rules' else 'every-column sweep'} of "
                         f"{split}; write a rerun under another --label (e.g. rules-dev)")
    out = []
    for t in tasks.load(split):
        x, dtype, kv, kc = tasks.hidden_and_known(t)
        a = answer(x, dtype, list(kc.values()), t["technique"], tasks.task_vocabulary(t), abstain_below,
                   exclude_group=corpus.source_group(t["file"]))
        out.append({"task_id": t["task_id"], "split": split, "status": "ok", "arm": label, **a})
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False, default=str) for r in out) + "\n",
                    encoding="utf-8", newline="\n")
    print(f"{len(out)} answers -> {path}")


def main() -> int:
    import warnings

    warnings.filterwarnings("ignore")
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", action="store_true")
    ap.add_argument("--split", choices=["dev", "test", "holdout", "holdout2", "holdout3", "holdout4", "holdout5", "holdout6", "holdout7"])
    ap.add_argument("--abstain-below", type=float, default=0.6)
    ap.add_argument("--label", default="rules", help="run label (output runs3/<label>_<split>.jsonl)")
    ap.add_argument("--sweep", metavar="SPLIT", help="hide every usable column of every file in data/SPLIT in turn")
    ap.add_argument("-v", "--verbose", action="count", default=0)
    a = ap.parse_args()
    if a.pool:
        pool(a.verbose)
    if a.split:
        bench(a.split, a.abstain_below, a.label)
    if a.sweep:
        sweep(a.sweep, a.label if a.label != "rules" else "sweep")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
