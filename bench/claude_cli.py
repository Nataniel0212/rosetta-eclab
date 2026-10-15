"""Run Claude models on the locked v2 naming benchmark through the Claude Code CLI (no API key).

Each task is one headless `claude -p` call in an empty working directory, with all
tools disabled, MCP servers and hooks off, and the benchmark's own system prompt
replacing Claude Code's. The model sees exactly what the Nemotron arms saw (system
prompt with the task's vocabulary, the dossier) and answers through --json-schema
with the same schema. It cannot read the answer key: no tools, no files.

  python -m bench.claude_cli --model claude-opus-5-5 --split dev
  python -m bench.claude_cli --model claude-opus-5-5 --split test --buckets main,decoy
  python -m bench.claude_cli --score --split test

Results: runs3/<label>_<split>.jsonl (git-ignored). The test split was locked for
spike v2 (prereg2/); prompt or harness changes are developed on dev only, and every
test run is listed in results/open_runs.md.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from spike2 import agent, score, tasks, vocab

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs3"
BLIND = Path(tempfile.gettempdir()) / "rosetta_blind"


def call_claude(model: str, effort: str | None, system: str, user: str, schema: dict, timeout: int = 900) -> dict:
    BLIND.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", suffix=".txt", dir=BLIND, delete=False, encoding="utf-8") as f:
        f.write(system)
        sys_path = f.name
    cmd = [shutil.which("claude") or "claude", "-p", "--tools", "", "--strict-mcp-config", "--no-session-persistence",
           "--settings", json.dumps({"disableAllHooks": True}),
           "--system-prompt-file", sys_path, "--json-schema", json.dumps(schema),
           "--output-format", "json", "--model", model]
    if effort:
        cmd += ["--effort", effort]
    t0 = time.time()
    try:
        p = subprocess.run(cmd, input=user, capture_output=True, text=True, encoding="utf-8", cwd=BLIND,
                           timeout=timeout)
    finally:
        os.unlink(sys_path)
    dt = time.time() - t0
    try:
        d = json.loads(p.stdout)
    except json.JSONDecodeError:
        return {"error": f"no JSON from claude (exit {p.returncode}): {(p.stdout or p.stderr or "")[:300]}", "latency_s": dt}
    u = d.get("usage") or {}
    return {"structured": d.get("structured_output"), "is_error": d.get("is_error"), "result": d.get("result"),
            "latency_s": round(dt, 2), "cost_usd_equiv": d.get("total_cost_usd"),
            "output_tokens": u.get("output_tokens"), "num_turns": d.get("num_turns")}


def run_task(task: dict, model: str, effort: str | None) -> dict:
    tag = {"task_id": task["task_id"], "split": task["split"]}
    names = tasks.task_vocabulary(task)
    system = agent.system_prompt(False, task.get("vocab_exclude") or [])
    user = ("Dossier written by code:\n\n" + task["dossier"] +
            f"\n\nName the hidden column {tasks.HIDDEN}. Answer with one JSON object as specified.")
    sch = agent.schema(False, names)["schema"]
    _, _, _, kc = tasks.hidden_and_known(task)
    present = {vocab.canonical(c.name): a for a, c in kc.items()}
    events, totals = [], {"latency_s": 0.0, "cost_usd_equiv": 0.0, "calls": 0}
    for attempt in range(1 + agent.MAX_PARSE_ERRORS):
        r = call_claude(model, effort, system, user, sch)
        totals["calls"] += 1
        totals["latency_s"] = round(totals["latency_s"] + r.get("latency_s", 0), 2)
        totals["cost_usd_equiv"] = round(totals["cost_usd_equiv"] + (r.get("cost_usd_equiv") or 0), 6)
        if r.get("error") or r.get("is_error") or not isinstance(r.get("structured"), dict):
            err = r.get("error") or str(r.get("result"))[:300]
            events.append({"kind": "cli_error", "error": err})
            if "limit" in err.lower() or "login" in err.lower():
                return {**tag, "status": "error", "error": err, **totals, "events": events}
            continue
        try:
            ans = agent.validate_answer(r["structured"], set(names))
            if ans["verdict"] == "NAME" and not task["identical_to"] and ans["name"] in present:
                raise ValueError(f"{ans['name']!r} is already the known column {present[ans['name']]}")
        except ValueError as exc:
            events.append({"kind": "invalid", "error": str(exc)[:300]})
            user += (f"\n\n[Your previous answer was rejected: {exc}. {tasks.HIDDEN} is a different column; "
                     "choose another vocabulary name or ABSTAIN.]")
            continue
        return {**tag, "status": "ok", **ans, **totals, "events": events}
    return {**tag, "status": "failed", "error": "no valid answer", **totals, "events": events}


def run(model: str, label: str, effort: str | None, split: str, buckets: set[str] | None, workers: int,
        limit: int | None) -> int:
    RUNS.mkdir(exist_ok=True)
    out = RUNS / f"{label}_{split}.jsonl"
    final = set()
    if out.exists():
        for line in out.read_text(encoding="utf-8").splitlines():
            if line.strip() and json.loads(line).get("status") in ("ok", "failed"):
                final.add(json.loads(line)["task_id"])
    todo = [t for t in tasks.load(split) if (buckets is None or t["bucket"] in buckets) and t["task_id"] not in final]
    todo = todo[: limit or None]
    print(f"{label} ({model}, effort {effort}) on {split}: {len(todo)} to do, {len(final)} done", flush=True)
    n = 0
    with ThreadPoolExecutor(max_workers=workers) as ex, out.open("a", encoding="utf-8", newline="\n") as f:
        futs = {ex.submit(run_task, t, model, effort): t for t in todo}
        for fut in as_completed(futs):
            r = fut.result()
            r.update(arm=label, model=model, effort=effort)
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            f.flush()
            n += 1
            if r["status"] == "error":
                print(f"  ERROR {r['task_id']}: {r['error'][:200]}", flush=True)
            if n % 10 == 0:
                print(f"  {n}/{len(todo)}", flush=True)
    return 0


def summarise(split: str, labels: list[str]) -> None:
    ts = tasks.load(split)
    main = [t for t in ts if t["bucket"] == "main"]
    decoys = [t for t in ts if t["bucket"] == "decoy" and t["decoy"]["kind"] in tasks.SCORED_DECOYS]
    names = sorted({t["truth"]["canonical"] for t in main})
    base = score.load_answers(split)
    arms = {a: base[a] for a in ("H_knn", "H_prior") if a in base}
    for label in labels:
        arms[label] = score._read(RUNS / f"{label}_{split}.jsonl")
    for arm, ans in arms.items():
        good = {t["task_id"]: score.correct(t, ans.get(t["task_id"])) for t in main}
        pn = score.per_name(main, good)
        macro = sum(pn[n] for n in names) / len(names) if names else 0
        hall = [score.hallucinated(t, ans.get(t["task_id"])) for t in decoys]
        answered = sum(score.ok(ans.get(t["task_id"])) for t in main)
        lat = [r.get("latency_s", 0) for r in ans.values() if isinstance(r, dict) and r.get("latency_s")]
        print(f"{arm:14s} macro {macro:.3f}  micro {sum(good.values()) / max(1, len(main)):.3f}  "
              f"decoys named {sum(hall)}/{len(hall)}  answered {answered}/{len(main)}"
              + (f"  median {sorted(lat)[len(lat) // 2]:.0f}s" if lat else ""))


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="claude-opus-5-5")
    ap.add_argument("--label", help="arm label (default: model name + effort)")
    ap.add_argument("--effort", choices=["low", "medium", "high", "xhigh", "max"])
    ap.add_argument("--split", choices=["dev", "test", "holdout", "holdout2", "holdout3", "holdout4", "holdout5"], required=True)
    ap.add_argument("--buckets")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--score", nargs="*", help="score these labels instead of running")
    a = ap.parse_args()
    if a.score is not None:
        labels = a.score or sorted({p.name.rsplit("_", 1)[0] for p in RUNS.glob(f"*_{a.split}.jsonl")})
        summarise(a.split, labels)
        return 0
    label = a.label or (a.model.replace("claude-", "") + (f"-{a.effort}" if a.effort else ""))
    buckets = set(a.buckets.split(",")) if a.buckets else None
    return run(a.model, label, a.effort, a.split, buckets, a.workers, a.limit)


if __name__ == "__main__":
    raise SystemExit(main())
