"""Model arms for the v2 naming benchmark, all via Nebius Token Factory.

Arms (pre-registered in prereg2/thresholds.json):
  L0   Lightning, thinking off, one shot
  LT   Lightning, thinking on, one shot
  U    Ultra, thinking on (medium effort), one shot
  LTA  Lightning, thinking on, may test up to MAX_TESTS hypotheses with code (dsl.py) before answering
  R    routed (computed from LTA and U, no extra calls): LTA's answer if its confidence >= ROUTE_TAU,
       otherwise Ultra's

Lessons from the v1 spike built in: schema-constrained JSON (json_schema with an
enum vocabulary, json_object fallback), invalid replies never enter the history,
finish_reason is checked (one retry with a larger budget), the step budget is
shown, the last step only accepts an answer, answers are validated against the
task's vocabulary, and parse errors are capped separately from tests. Every
call is logged with tokens, reasoning tokens, latency and cost; a worst-case
reservation keeps parallel workers under the hard budget cap.

Task status: "ok" (valid answer), "failed" (the model's fault: invalid replies,
repeated cut-offs; final, counts as wrong), "error" (infrastructure: may be
retried by run.py).
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from . import dsl, tasks, vocab

ROOT = Path(__file__).resolve().parents[1]
PREREG = ROOT / "prereg2"
RUNS = ROOT / "runs2"

BASE_URL = "https://api.tokenfactory.nebius.com/v1"  # hard rule: every model call goes directly to Token Factory
MODEL_DEEP = os.environ.get("ROSETTA_MODEL_DEEP", "nvidia/Nemotron-3-Ultra-550b-a55b")
MODEL_FAST = os.environ.get("ROSETTA_MODEL_FAST", "nvidia/Nemotron-3_5-Lightning")

# USD per 1M tokens (in, out), Token Factory catalogue 2026-10-13. Verify before quoting.
PRICES = {
    "nvidia/Nemotron-3-Ultra-550b-a55b": (1.00, 3.00),
    "nvidia/Nemotron-3_5-Lightning": (0.06, 0.24),
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B": (0.06, 0.24),
}

MAX_TESTS = 6
MAX_PARSE_ERRORS = 3
ROUTE_TAU = 0.7
LENGTH_RETRY_FACTOR = 1.5
MAX_TOKENS_CEILING = 12288


@dataclass(frozen=True)
class Arm:
    key: str
    model: str
    thinking: bool
    effort: str | None
    tools: bool
    max_tokens: int


ARMS = {
    "L0": Arm("L0", MODEL_FAST, False, None, False, 2048),
    "LT": Arm("LT", MODEL_FAST, True, None, False, 8192),
    "U": Arm("U", MODEL_DEEP, True, "medium_effort", False, 8192),
    "LTA": Arm("LTA", MODEL_FAST, True, None, True, 8192),
}


# --- prompts and schemas ---------------------------------------------------------

def system_prompt(tools: bool, exclude: list[str] | tuple = ()) -> str:
    text = (PREREG / "prompt.md").read_text(encoding="utf-8")
    text = text.replace("{VOCABULARY}", vocab.render(exclude))
    if tools:
        text = re.sub(r"<!-- ONESHOT -->.*?<!-- /ONESHOT -->", "", text, flags=re.S)
        text = text.replace("{DSL_DOC}", dsl.DOC).replace("{MAX_TESTS}", str(MAX_TESTS))
    else:
        text = re.sub(r"<!-- TOOLS -->.*?<!-- /TOOLS -->", "", text, flags=re.S)
    for tag in ("<!-- TOOLS -->", "<!-- /TOOLS -->", "<!-- ONESHOT -->", "<!-- /ONESHOT -->"):
        text = text.replace(tag, "")
    return text.strip()


def answer_properties(names: list[str]) -> dict:
    return {
        "evidence": {"type": "string"},
        "verdict": {"type": "string", "enum": ["NAME", vocab.DUPLICATE, vocab.ABSTAIN]},
        "name": {"type": "string", "enum": names + ["NONE"]},
        "unit": {"type": "string"},
        "duplicate_of": {"type": "string"},
        "alternatives": {"type": "array", "items": {"type": "string", "enum": names}, "maxItems": 2},
        "confidence": {"type": "number"},
    }


def schema(step: bool, names: list[str] | None = None) -> dict:
    """step=True: a hypothesis-test step (action test|answer); step=False: answer only."""
    props = answer_properties(names if names is not None else vocab.names())
    if step:
        props = {"thought": {"type": "string"}, "action": {"type": "string", "enum": ["test", "answer"]},
                 "expression": {"type": "string"}, **props}
    return {"name": "rosetta_step" if step else "rosetta_answer", "strict": True,
            "schema": {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}}


# --- parsing and validation ------------------------------------------------------

_XML_TOOL = re.compile(r"<tool_call>.*?</tool_call>", re.S)
_FENCE = re.compile(r"```(?:json)?", re.S)
_EXACT = {n: n for n in vocab.names()} | {a: vocab.canonical(a) for a in vocab.ALIAS}


def _lookup_lower() -> dict[str, str]:
    """Case-insensitive fallback, only for spellings that stay unambiguous (dq and dQ are different names)."""
    seen: dict[str, set[str]] = {}
    for k, v in _EXACT.items():
        seen.setdefault(k.lower(), set()).add(v)
    return {k: next(iter(v)) for k, v in seen.items() if len(v) == 1}


_LOWER = _lookup_lower()


def resolve_name(name: str) -> str | None:
    name = str(name or "").strip()
    return _EXACT.get(name) or _LOWER.get(name.lower())


def parse_json(text: str) -> dict:
    """First JSON object after any reasoning text (handles a lone closing </think>, fences, tool-call XML)."""
    t = text or ""
    if "</think>" in t:
        t = t.rsplit("</think>", 1)[1]
    t = _XML_TOOL.sub("", t)
    t = _FENCE.sub("", t)
    start = t.find("{")
    if start < 0:
        raise ValueError("no JSON object in reply")
    obj, _ = json.JSONDecoder().raw_decode(t[start:])
    if not isinstance(obj, dict):
        raise ValueError("reply is not a JSON object")
    return obj


def validate_answer(obj: dict, allowed: set[str]) -> dict:
    """Normalise an answer; raise ValueError with a plain-language message if it is unusable."""
    verdict = str(obj.get("verdict", "")).strip().upper()
    if verdict not in ("NAME", vocab.DUPLICATE, vocab.ABSTAIN):
        raise ValueError(f'verdict must be "NAME", "DUPLICATE" or "ABSTAIN", got {obj.get("verdict")!r}')
    name = str(obj.get("name", "") or "").strip()
    outside = False
    if verdict == "NAME":
        canon = resolve_name(name)
        if not canon:
            raise ValueError(f"name {name!r} is not in the vocabulary; copy one name exactly as listed")
        # A real EC-Lab name that this task's vocabulary leaves out is a final (wrong) answer, not a
        # formatting error: no second chance, otherwise an oov decoy could be named and then retracted.
        outside = canon not in allowed
        name = canon
    alts = [resolve_name(a) for a in (obj.get("alternatives") or [])][:2]
    conf = obj.get("confidence", 0)
    try:
        conf = max(0.0, min(1.0, float(conf)))
    except (TypeError, ValueError):
        conf = 0.0
    return {"verdict": verdict, "name": name if verdict == "NAME" else "NONE", "unit": str(obj.get("unit", "") or ""),
            "duplicate_of": str(obj.get("duplicate_of", "") or ""),
            "alternatives": [a for a in alts if a and a in allowed], "confidence": conf,
            "outside_task_vocabulary": outside, "evidence": str(obj.get("evidence", ""))[:1500]}


# --- client ----------------------------------------------------------------------

class BudgetExceeded(RuntimeError):
    pass


class Ledger:
    """Thread-safe running cost with a hard cap. Each call reserves its worst case first."""

    def __init__(self, cap_usd: float, log: Path):
        self.cap = cap_usd
        self.spent = 0.0
        self.reserved = 0.0
        self.lock = threading.Lock()
        self.log = log
        self.stop = False

    def reserve(self, worst_usd: float):
        with self.lock:
            if self.stop or self.spent + self.reserved + worst_usd > self.cap:
                self.stop = True
                raise BudgetExceeded(f"budget cap ${self.cap:.2f} reached (spent ${self.spent:.4f}, "
                                     f"reserved ${self.reserved:.4f}, next call up to ${worst_usd:.4f})")
            self.reserved += worst_usd

    def settle(self, worst_usd: float, entry: dict | None):
        with self.lock:
            self.reserved -= worst_usd
            if entry is not None:
                self.spent += entry.get("cost_usd", 0.0)
                self.log.parent.mkdir(exist_ok=True)
                with self.log.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")


def price(model: str) -> tuple[float, float]:
    if model not in PRICES:
        raise SystemExit(f"no price for model {model!r}; add it to PRICES before running (the budget cap needs it)")
    return PRICES[model]


class TokenFactory:
    """OpenAI-compatible client for Nebius Token Factory. Never via a gateway."""

    def __init__(self, ledger: Ledger, response_mode: str = "json_schema"):
        from openai import OpenAI

        assert urlparse(BASE_URL).hostname == "api.tokenfactory.nebius.com"
        key = os.environ.get("NEBIUS_API_KEY")
        if not key:
            raise SystemExit("NEBIUS_API_KEY is not set (Windows user environment variable)")
        self.client = OpenAI(base_url=BASE_URL, api_key=key, timeout=300, max_retries=0)
        self.ledger = ledger
        self.mode = response_mode  # "json_schema" | "json_object" | "none"

    def __call__(self, arm: Arm, messages: list[dict], fmt: dict, tag: dict, max_tokens: int) -> dict:
        import openai

        pin, pout = price(arm.model)
        prompt_chars = sum(len(m["content"]) for m in messages)
        worst = (prompt_chars / 2.0 * pin + max_tokens * pout) / 1e6  # generous: 2 chars per token
        self.ledger.reserve(worst)
        entry = None
        try:
            kw: dict[str, Any] = {"enable_thinking": arm.thinking, "force_nonempty_content": True}
            if arm.effort:
                kw[arm.effort] = True
            extra: dict[str, Any] = {"extra_body": {"chat_template_kwargs": kw}}
            if self.mode == "json_schema":
                extra["response_format"] = {"type": "json_schema", "json_schema": fmt}
            elif self.mode == "json_object":
                extra["response_format"] = {"type": "json_object"}
            t0 = time.time()
            for attempt in range(4):
                try:
                    raw = self.client.chat.completions.with_raw_response.create(
                        model=arm.model, messages=messages, max_tokens=max_tokens,
                        temperature=1.0, top_p=0.95, **extra)
                    break
                except openai.APIStatusError as exc:
                    if exc.status_code == 402:  # out of credit: stop everything, never retry
                        self.ledger.stop = True
                        raise BudgetExceeded("HTTP 402 from Token Factory (credit exhausted)") from exc
                    if (exc.status_code < 500 and exc.status_code != 429) or attempt == 3:
                        raise
                except (openai.APITimeoutError, openai.APIConnectionError):
                    if attempt == 3:
                        raise
                time.sleep((3, 10, 30)[attempt])
            resp = raw.parse()
            dt = time.time() - t0
            u = resp.usage
            reasoning = getattr(getattr(u, "completion_tokens_details", None), "reasoning_tokens", None)
            ch = resp.choices[0]
            cost = (u.prompt_tokens * pin + u.completion_tokens * pout) / 1e6
            out = {"content": ch.message.content or "", "finish_reason": ch.finish_reason,
                   "prompt_tokens": u.prompt_tokens, "completion_tokens": u.completion_tokens,
                   "reasoning_tokens": reasoning, "latency_s": round(dt, 2), "cost_usd": round(cost, 6),
                   "model": resp.model}
            entry = {"t": time.strftime("%Y-%m-%d %H:%M:%S"), **tag, "arm": arm.key, "mode": self.mode,
                     "max_tokens": max_tokens, **{k: v for k, v in out.items() if k != "content"},
                     "content_chars": len(out["content"])}
            return out
        finally:
            self.ledger.settle(worst, entry)


class MockModel:
    """Offline stand-in: answers from a scripted policy, no network, cost 0."""

    def __init__(self, policy):
        self.policy = policy

    def __call__(self, arm: Arm, messages: list[dict], fmt: dict, tag: dict, max_tokens: int) -> dict:
        reply = self.policy(arm, messages, fmt, tag)
        content = reply if isinstance(reply, str) else json.dumps(reply)
        return {"content": content, "finish_reason": "stop", "prompt_tokens": 0, "completion_tokens": 0,
                "reasoning_tokens": 0, "latency_s": 0.0, "cost_usd": 0.0, "model": "mock"}


# --- one task --------------------------------------------------------------------

def _acc(totals: dict, r: dict):
    for k in ("prompt_tokens", "completion_tokens", "cost_usd", "latency_s"):
        totals[k] = round(totals.get(k, 0) + (r.get(k) or 0), 6)
    totals["reasoning_tokens"] = totals.get("reasoning_tokens", 0) + (r.get("reasoning_tokens") or 0)
    totals["calls"] = totals.get("calls", 0) + 1


def run_task(arm: Arm, task: dict, model) -> dict:
    tag = {"task_id": task["task_id"], "split": task["split"]}
    names = tasks.task_vocabulary(task)
    allowed = set(names)
    totals: dict = {}
    user = ("Dossier written by code:\n\n" + task["dossier"] +
            f"\n\nName the hidden column {tasks.HIDDEN}. Answer with one JSON object as specified.")
    messages = [{"role": "system", "content": system_prompt(arm.tools, task.get("vocab_exclude") or [])},
                {"role": "user", "content": user}]
    tests: list[dict] = []
    events: list[dict] = []
    parse_errors = 0
    max_tokens = arm.max_tokens
    x, _, kv, kc = tasks.hidden_and_known(task)
    present = {vocab.canonical(c.name): a for a, c in kc.items()}
    step = 0
    while True:
        step += 1
        step_mode = arm.tools and len(tests) < MAX_TESTS  # after the last test only an answer is accepted
        r = model(arm, messages, schema(step_mode, names), tag, max_tokens)
        _acc(totals, r)
        if r["finish_reason"] == "length":
            parse_errors += 1
            events.append({"step": step, "kind": "length", "max_tokens": max_tokens})
            if parse_errors > MAX_PARSE_ERRORS:
                return {**tag, "status": "failed", "error": "repeated length cut-offs", **totals, "tests": tests, "events": events}
            max_tokens = min(MAX_TOKENS_CEILING, int(max_tokens * LENGTH_RETRY_FACTOR))
            messages.append({"role": "user", "content": "[your previous reply was cut off at the token limit and was "
                             "discarded] Reason more briefly and reply with one complete JSON object."})
            continue
        try:
            obj = parse_json(r["content"])
            action = str(obj.get("action", "answer")).lower() if step_mode else "answer"
            if action == "test":
                expr = str(obj.get("expression", ""))
                try:
                    res = dsl.fit(x, dsl.evaluate(expr, kv))
                except dsl.DSLError as exc:
                    res = {"error": str(exc)}
                tests.append({"expression": expr, **res})
                events.append({"step": step, "kind": "test", "expression": expr, "result": res})
                left = MAX_TESTS - len(tests)
                note = (f"[test {len(tests)}/{MAX_TESTS}] {tasks.HIDDEN} = a*({expr}) + b -> {json.dumps(res)}\n"
                        + (f"{left} tests left." if left else "No tests left: your next reply must be your answer."))
                messages.append({"role": "assistant", "content": json.dumps(
                    {"thought": str(obj.get("thought", ""))[:600], "action": "test", "expression": expr})})
                messages.append({"role": "user", "content": note})
                continue
            ans = validate_answer(obj, allowed)
            # Code gates the prompt's rule (added after the dev run, 2026-10-15): TARGET is not a quantity that is
            # already a known column, unless code found it identical to one. One correction, like any invalid reply.
            if ans["verdict"] == "NAME" and not task["identical_to"] and ans["name"] in present:
                raise ValueError(f"{ans['name']!r} is already the known column {present[ans['name']]}, and {tasks.HIDDEN} "
                                 "is a different column; choose another vocabulary name or ABSTAIN")
        except (ValueError, json.JSONDecodeError) as exc:
            parse_errors += 1
            events.append({"step": step, "kind": "invalid", "error": str(exc)[:300]})
            if parse_errors > MAX_PARSE_ERRORS:
                return {**tag, "status": "failed", "error": f"invalid replies: {exc}", **totals, "tests": tests, "events": events}
            # The invalid reply is not added to the history; only a short correction is.
            messages.append({"role": "user", "content": f"[previous reply discarded: {exc}] Reply with exactly one JSON object as specified."})
            continue
        return {**tag, "status": "ok", **ans, **totals, "tests": tests, "steps": step, "events": events}
