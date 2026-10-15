"""Score arms against the pre-registered answer key and apply the decision rule.

Primary metric: top-1 accuracy on the test 'main' bucket, macro-averaged over
canonical names (each name counts once), with a 95% bootstrap interval that
resamples names, and a paired bootstrap interval for the margin over the best
baseline. Only rows with status "ok" count as answers: a failed or missing
main task is wrong, and a failed or missing decoy counts as a hallucination.

Also reported: micro and top-3 accuracy, abstention rate, the elim, duplicate
and constant buckets, decoys by kind, the real drift columns, cost and latency.

Usage:  python -m spike2.score [--split test] [--mock]
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from . import tasks, vocab

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs2"
PREREG = ROOT / "prereg2"
RESULTS = ROOT / "results"

MODEL_ARMS = ["L0", "LT", "U", "LTA", "R"]
BASELINE_ARMS = ["H_prior", "H_knn", "H_elim", "H_combo"]


def ok(ans: dict | None) -> bool:
    return bool(ans) and ans.get("status", "ok") == "ok"


def correct(task: dict, ans: dict | None) -> bool:
    return ok(ans) and ans.get("verdict") == "NAME" and vocab.canonical(ans.get("name", "")) == task["truth"]["canonical"]


def correct_top3(task: dict, ans: dict | None) -> bool:
    if not ok(ans) or ans.get("verdict") != "NAME":
        return False
    cands = [ans.get("name", "")] + list(ans.get("alternatives") or [])[:2]
    return task["truth"]["canonical"] in {vocab.canonical(c) for c in cands}


def hallucinated(task: dict, ans: dict | None) -> bool:
    """A decoy is hallucinated when the arm names it, fails, or gives no answer (an exact copy may carry its source's name)."""
    if not ok(ans):
        return True
    if ans.get("verdict") != "NAME":
        return False
    d = task["decoy"]
    if d["kind"] == "copy" and vocab.canonical(ans.get("name", "")) == vocab.canonical(d["source_name"]):
        return False
    return True


def per_name(task_list: list[dict], good: dict[str, bool]) -> dict[str, float]:
    by = defaultdict(list)
    for t in task_list:
        by[t["truth"]["canonical"]].append(good[t["task_id"]])
    return {n: float(np.mean(v)) for n, v in sorted(by.items())}


def boot_ci(values: np.ndarray, rng: np.random.Generator, n_boot: int = 4000) -> tuple[float, float]:
    if not len(values):
        return (float("nan"), float("nan"))
    b = [values[rng.integers(0, len(values), len(values))].mean() for _ in range(n_boot)]
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def _read(path: Path) -> dict[str, dict]:
    """The first final row per task (status ok or failed: both are final), else the last error row."""
    out: dict[str, dict] = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        prev = out.get(r["task_id"])
        if prev is None or prev.get("status") not in ("ok", "failed"):
            out[r["task_id"]] = r
    return out


def load_answers(split: str, suffix: str = "") -> dict[str, dict[str, dict]]:
    arms: dict[str, dict[str, dict]] = defaultdict(dict)
    b = (PREREG / "baselines_test.jsonl") if split == "test" else (RUNS / f"baselines_{split}.jsonl")
    if b.exists():
        for line in b.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                for arm in BASELINE_ARMS:
                    if arm in r:
                        arms[arm][r["task_id"]] = r[arm]
    for arm in MODEL_ARMS:
        rows = _read(RUNS / f"{arm}{suffix}_{split}.jsonl")
        if rows:
            arms[arm] = rows
    return arms


def summarise(split: str = "test", suffix: str = "") -> dict:
    ts = tasks.load(split)
    arms = load_answers(split, suffix)
    rng = np.random.default_rng(7)
    buckets = defaultdict(list)
    for t in ts:
        buckets[t["bucket"]].append(t)
    main = buckets["main"]
    decoys = buckets["decoy"]
    scored_decoys = [t for t in decoys if t["decoy"]["kind"] in tasks.SCORED_DECOYS]
    names = sorted({t["truth"]["canonical"] for t in main})
    drift = [t for t in ts if t["truth"] and t["truth"].get("galvani_unknown")]
    out = {"split": split, "n_main": len(main), "n_names": len(names), "n_scored_decoys": len(scored_decoys),
           "arms": {}}
    per_name_vec = {}
    for arm, ans in sorted(arms.items()):
        good = {t["task_id"]: correct(t, ans.get(t["task_id"])) for t in main}
        pn = per_name(main, good)
        vec = np.array([pn[n] for n in names]) if main else np.array([])
        per_name_vec[arm] = vec
        hall = [hallucinated(t, ans.get(t["task_id"])) for t in scored_decoys]
        costs = [r["cost_usd"] for r in ans.values() if isinstance(r, dict) and "cost_usd" in r]
        lat = [r["latency_s"] for r in ans.values() if isinstance(r, dict) and "latency_s" in r]

        def bucket_rate(bk, fn):
            items = buckets.get(bk, [])
            return f"{sum(fn(t, ans.get(t['task_id'])) for t in items)}/{len(items)}" if items else None

        out["arms"][arm] = {
            "answered_main": sum(ok(ans.get(t["task_id"])) for t in main),
            "answered_scored_decoys": sum(ok(ans.get(t["task_id"])) for t in scored_decoys),
            "macro_top1": round(float(vec.mean()), 4) if len(vec) else None,
            "macro_top1_ci95": [round(v, 4) for v in boot_ci(vec, rng)] if len(vec) else None,
            "micro_top1": round(float(np.mean(list(good.values()))), 4) if good else None,
            "top3_micro": round(float(np.mean([correct_top3(t, ans.get(t["task_id"])) for t in main])), 4) if main else None,
            "abstain_rate_main": round(float(np.mean([ok(ans.get(t["task_id"])) and ans[t["task_id"]].get("verdict") != "NAME" for t in main])), 4) if main else None,
            "hallucination_rate": round(float(np.mean(hall)), 4) if hall else None,
            "hallucinations": f"{sum(hall)}/{len(hall)}",
            "decoys_by_kind": {k: f"{sum(hallucinated(t, ans.get(t['task_id'])) for t in decoys if t['decoy']['kind'] == k)}"
                                  f"/{sum(1 for t in decoys if t['decoy']['kind'] == k)} named"
                               for k in sorted({t['decoy']['kind'] for t in decoys})},
            "elim_bucket_correct": bucket_rate("elim", correct),
            "duplicate_bucket_ok": bucket_rate("duplicate", lambda t, a: ok(a) and (a.get("verdict") == vocab.DUPLICATE or correct(t, a))),
            "constant_bucket_abstained": bucket_rate("constant", lambda t, a: ok(a) and a.get("verdict") == vocab.ABSTAIN),
            "drift_columns": {t["task_id"]: {"truth": t["truth"]["name"], "bucket": t["bucket"],
                                             "verdict": (ans.get(t["task_id"]) or {}).get("verdict"),
                                             "answer": (ans.get(t["task_id"]) or {}).get("name")} for t in drift},
            "cost_usd_total": round(float(sum(costs)), 4) if costs else None,
            "cost_usd_per_task": round(float(np.mean(costs)), 6) if costs else None,
            "latency_s_median": round(float(np.median(lat)), 2) if lat else None,
            "per_name": {k: round(v, 3) for k, v in pn.items()},
        }
    out["decision"] = decide(out, per_name_vec, rng)
    return out


def decide(summary: dict, vecs: dict, rng) -> dict:
    th = json.loads((PREREG / "thresholds.json").read_text(encoding="utf-8"))
    arms = summary["arms"]
    base = {a: arms[a]["macro_top1"] for a in BASELINE_ARMS if a in arms and arms[a]["macro_top1"] is not None}
    best_arm = max(base, key=base.get) if base else None
    best = base.get(best_arm)
    verdicts = {}
    for a in th["arms_eligible"]:
        if a not in arms or arms[a]["macro_top1"] is None:
            continue
        r = arms[a]
        complete = (r["answered_main"] >= th["min_answered_fraction"] * summary["n_main"]
                    and r["answered_scored_decoys"] >= th["min_answered_fraction"] * summary["n_scored_decoys"])
        diff = vecs[a] - vecs[best_arm] if best_arm else None
        margin_ci = None
        if diff is not None and len(diff):
            margin_ci = [round(v, 4) for v in boot_ci(diff, rng)]
        verdicts[a] = {
            "complete": complete,
            "accuracy_ok": r["macro_top1"] >= th["macro_top1_min"],
            "margin": None if best is None else round(r["macro_top1"] - best, 4),
            "margin_ci95_paired": margin_ci,
            "margin_ok": best is not None and r["macro_top1"] - best >= th["margin_over_best_baseline_min"],
            "hallucination_ok": r["hallucination_rate"] is not None and r["hallucination_rate"] <= th["hallucination_rate_max"],
        }
        verdicts[a]["passes"] = all(verdicts[a][k] for k in ("complete", "accuracy_ok", "margin_ok", "hallucination_ok"))
        verdicts[a]["strong"] = verdicts[a]["passes"] and r["macro_top1"] - best >= th["margin_target_strong"]
    passing = [a for a, v in verdicts.items() if v["passes"]]
    strong = [a for a, v in verdicts.items() if v["strong"]]
    decision = th["go_strong"] if strong else th["go"] if passing else th["no_go"]
    return {"best_baseline": best_arm, "best_baseline_macro_top1": best, "per_arm": verdicts, "passing_arms": passing,
            "strong_arms": strong, "decision": decision}


def main() -> int:
    import sys

    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["dev", "test"])
    ap.add_argument("--brief", action="store_true")
    ap.add_argument("--mock", action="store_true", help="score runs2/<arm>_mock_<split>.jsonl (pipeline test)")
    a = ap.parse_args()
    s = summarise(a.split, "_mock" if a.mock else "")
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"v2_score_{a.split}{'_mock' if a.mock else ''}.json").write_text(
        json.dumps(s, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"{s['split']}: {s['n_main']} main tasks over {s['n_names']} names, {s['n_scored_decoys']} scored decoys")
    for arm, r in s["arms"].items():
        print(f"{arm:8s} macro {r['macro_top1']} CI {r['macro_top1_ci95']}  micro {r['micro_top1']}  top3 {r['top3_micro']}  "
              f"abstain {r['abstain_rate_main']}  halluc {r['hallucinations']}  elim {r['elim_bucket_correct']}  "
              f"dup {r['duplicate_bucket_ok']}  const {r['constant_bucket_abstained']}  $/task {r['cost_usd_per_task']}  "
              f"answered {r['answered_main']}/{s['n_main']}")
    if not a.brief:
        print(json.dumps(s["decision"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
