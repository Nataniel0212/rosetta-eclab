"""Run model arms on a split, resumably, in parallel, under a hard budget.

  python -m spike2.run --arm LT --split dev            # dev: harness check, prompt may still change
  python -m spike2.run --arm U --split test --buckets main,decoy --max-usd 3.5
  python -m spike2.run --route --split test            # R = LTA if confidence >= tau else U (no calls)
  python -m spike2.run --arm LTA --split dev --mock oracle   # offline, no API

Results: runs2/<arm>_<split>.jsonl (one line per attempt; the scorer takes the last ok row).
Calls:   runs2/calls.jsonl (every API call: tokens, reasoning tokens, latency, $).

Rerun rule (pre-registered): tasks with status "ok" or "failed" are final.
Only status "error" (infrastructure) is retried, at most MAX_ATTEMPTS per task.
The test split runs only when prereg2/LOCK.json exists, every hashed file still
matches it, and prereg2/ and spike2/ are committed and clean in git.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from . import agent, tasks, vocab

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs2"
MAX_ATTEMPTS = 2


def results_path(arm: str, split: str) -> Path:
    return RUNS / f"{arm}_{split}.jsonl"


def attempts(path: Path) -> tuple[set[str], Counter]:
    """(final task ids, attempts per task)."""
    final, n = set(), Counter()
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                n[r["task_id"]] += 1
                if r.get("status") in ("ok", "failed"):
                    final.add(r["task_id"])
    return final, n


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()  # same normalisation as lock.py


def prereg_locked() -> tuple[bool, str]:
    lock_path = ROOT / "prereg2" / "LOCK.json"
    if not lock_path.exists():
        return False, "prereg2/LOCK.json missing (run: python -m spike2.lock)"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    required = {f"spike2/{p.name}" for p in (ROOT / "spike2").glob("*.py")} | {
        f"prereg2/{p.name}" for p in (ROOT / "prereg2").glob("*") if p.is_file() and p.name != "LOCK.json"}
    missing = sorted(required - set(lock["files"]))
    if missing:
        return False, f"LOCK.json does not cover: {missing}"
    from importlib.metadata import version
    for pkg, v in lock.get("packages", {}).items():
        if version(pkg) != v:
            return False, f"package {pkg} is {version(pkg)}, locked {v}"
    for rel, h in lock["files"].items():
        p = ROOT / rel
        if not p.exists() or _sha(p) != h:
            return False, f"{rel} differs from LOCK.json"
    if _sha(tasks.tasks_file()) != lock["tasks_jsonl_sha256"]:
        return False, "data/v2/tasks.jsonl differs from LOCK.json (rebuild with python -m spike2.tasks)"
    st = subprocess.run(["git", "status", "--porcelain", "--", "prereg2", "spike2"], cwd=ROOT,
                        capture_output=True, text=True)
    if st.returncode != 0:
        return False, f"git status failed ({st.returncode}): {st.stderr.strip()}"
    if st.stdout.strip():
        return False, f"uncommitted changes in prereg2/ or spike2/:\n{st.stdout}"
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", "prereg2/LOCK.json"], cwd=ROOT,
                             capture_output=True, text=True)
    if tracked.returncode != 0:
        return False, "prereg2/LOCK.json is not committed"
    return True, "ok"


def oracle_policy(task_by_id):
    """Mock: answers correctly on main tasks, abstains on decoys (pipeline test only)."""
    def policy(arm, messages, fmt, tag):
        t = task_by_id[tag["task_id"]]
        step = fmt["name"] == "rosetta_step"
        base = {"evidence": "mock", "unit": "", "duplicate_of": "", "alternatives": [], "confidence": 0.9}
        if step and sum(1 for m in messages if m["role"] == "assistant") == 0:
            return {"thought": "test", "action": "test", "expression": "c0 * 2", "verdict": "ABSTAIN", "name": "NONE", **base}
        if t["bucket"] in ("main", "elim"):
            ans = {"verdict": "NAME", "name": t["truth"]["canonical"]}
        elif t["bucket"] == "duplicate":
            ans = {"verdict": "DUPLICATE", "name": "NONE", "duplicate_of": t["identical_to"][0]}
        else:
            ans = {"verdict": "ABSTAIN", "name": "NONE"}
        return {**({"thought": "", "action": "answer", "expression": ""} if step else {}), **base, **ans}
    return policy


def random_policy(seed=0):
    rng = random.Random(seed)
    names = vocab.names()

    def policy(arm, messages, fmt, tag):
        if rng.random() < 0.1:
            return "</think> not json at all"
        ans = {"evidence": "mock", "verdict": rng.choice(["NAME", "NAME", "ABSTAIN"]), "name": rng.choice(names),
               "unit": "", "duplicate_of": "", "alternatives": rng.sample(names, 2), "confidence": rng.random()}
        if fmt["name"] == "rosetta_step":
            ans = {"thought": "", "action": rng.choice(["test", "answer"]),
                   "expression": rng.choice(["c0", "cumsum(c1)", "bad(", "c99", "c0/c0"]), **ans}
        return ans
    return policy


def run_arm(arm_key: str, split: str, buckets: set[str] | None, max_usd: float, workers: int, mock: str | None,
            limit: int | None, response_mode: str) -> int:
    arm = agent.ARMS[arm_key]
    ts = [t for t in tasks.load(split) if buckets is None or t["bucket"] in buckets]
    out = results_path(arm_key + ("_mock" if mock else ""), split)
    final, n_att = attempts(out)
    todo = [t for t in ts if t["task_id"] not in final and n_att[t["task_id"]] < MAX_ATTEMPTS][: limit or None]
    gave_up = [t["task_id"] for t in ts if t["task_id"] not in final and n_att[t["task_id"]] >= MAX_ATTEMPTS]
    RUNS.mkdir(exist_ok=True)
    ledger = agent.Ledger(max_usd, RUNS / ("calls_mock.jsonl" if mock else "calls.jsonl"))
    if mock:
        by_id = {t["task_id"]: t for t in tasks.load()}
        model = agent.MockModel(oracle_policy(by_id) if mock == "oracle" else random_policy())
    else:
        model = agent.TokenFactory(ledger, response_mode=response_mode)
    print(f"{arm_key} on {split}: {len(todo)} to do, {len(final)} final, {len(gave_up)} out of attempts; cap ${max_usd}",
          flush=True)
    counts = Counter()
    with ThreadPoolExecutor(max_workers=workers) as ex, out.open("a", encoding="utf-8") as f:
        futs = {ex.submit(agent.run_task, arm, t, model): t for t in todo}
        for fut in as_completed(futs):
            t = futs[fut]
            try:
                r = fut.result()
            except agent.BudgetExceeded as exc:
                counts["budget_stop"] += 1
                if counts["budget_stop"] == 1:
                    print(f"  STOP: {exc}", flush=True)
                continue  # not written: the task stays open for a later run
            except Exception as exc:
                r = {"task_id": t["task_id"], "split": split, "status": "error", "error": f"{type(exc).__name__}: {exc}"}
            r["arm"] = arm_key
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
            f.flush()
            counts[r.get("status")] += 1
            done = sum(v for k, v in counts.items() if k != "budget_stop")
            if done % 10 == 0:
                print(f"  {done}/{len(todo)} done, spent ${ledger.spent:.4f}", flush=True)
    print(f"{arm_key}: {dict(counts)}, spent ${ledger.spent:.4f} this run", flush=True)
    return 2 if ledger.stop else 0


def route(split: str, tau: float = agent.ROUTE_TAU, suffix: str = "") -> int:
    from .score import _read

    lta, u = _read(results_path("LTA" + suffix, split)), _read(results_path("U" + suffix, split))
    out = []
    for tid, a in lta.items():
        if a.get("status") == "ok" and a.get("confidence", 0) >= tau:
            r = {**a, "routed_to": "LTA"}
        elif tid in u:
            r = {**u[tid], "routed_to": "U", "cost_usd": round(a.get("cost_usd", 0) + u[tid].get("cost_usd", 0), 6),
                 "latency_s": round(a.get("latency_s", 0) + u[tid].get("latency_s", 0), 2)}
        else:  # Ultra was not run on this bucket: keep LTA's answer
            r = {**a, "routed_to": "LTA (no U)"}
        r["arm"] = "R"
        out.append(r)
    results_path("R" + suffix, split).write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in out) + "\n", encoding="utf-8")
    esc = sum(r["routed_to"] == "U" for r in out)
    print(f"R on {split}: {len(out)} tasks, escalated to Ultra {esc} ({esc / max(1, len(out)):.0%})")
    return 0


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=sorted(agent.ARMS))
    ap.add_argument("--split", choices=["dev", "test"], required=True)
    ap.add_argument("--buckets", help="comma-separated subset of main,elim,decoy,duplicate,constant")
    ap.add_argument("--max-usd", type=float, default=1.0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--mock", choices=["oracle", "random"])
    ap.add_argument("--route", action="store_true")
    ap.add_argument("--response-mode", default="json_schema", choices=["json_schema", "json_object", "none"])
    a = ap.parse_args()
    if a.route:
        return route(a.split, suffix="_mock" if a.mock else "")
    if not a.arm:
        ap.error("--arm is required unless --route")
    if a.split == "test" and not a.mock:
        ok, why = prereg_locked()
        if not ok:
            print("REFUSED: the test split is locked until prereg2/ is frozen and committed.\n" + why)
            return 3
        lock = json.loads((ROOT / "prereg2" / "LOCK.json").read_text(encoding="utf-8"))
        if a.response_mode != lock["response_mode"]:
            print(f"REFUSED: --response-mode {a.response_mode} differs from the locked {lock['response_mode']}")
            return 3
    buckets = set(a.buckets.split(",")) if a.buckets else None
    return run_arm(a.arm, a.split, buckets, a.max_usd, a.workers, a.mock, a.limit, a.response_mode)


if __name__ == "__main__":
    raise SystemExit(main())
