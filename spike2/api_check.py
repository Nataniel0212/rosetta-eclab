"""Smoke test for the v2 harness against Token Factory (toy dossier, no benchmark task).

Uses the real system prompt and the real answer schema (114-name enum), and checks
per model: json_schema (strict) and json_object, thinking on/off, medium_effort on
Ultra, finish_reason, usage and reasoning tokens. Rule for --response-mode
(pre-registered): json_schema only if every configuration returns a valid answer
under json_schema AND every thinking configuration reports reasoning_tokens > 0
under it (guided decoding must not switch reasoning off); otherwise json_object.
Results are appended to runs2/api_check.jsonl.

Usage:  python -m spike2.api_check
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from . import agent
from .agent import BASE_URL, MODEL_DEEP, MODEL_FAST, PRICES

ROOT = Path(__file__).resolve().parents[1]

SCHEMA = agent.schema(False)
TOY_DOSSIER = """Technique code from the file's settings module: OCV
Rows: 3601

Hidden column TARGET: dtype <f8 (decoded by code). Its name is what you must find.
Code check - constant: no
Code check - identical (all rows) to known column(s): none
Statistics of TARGET: rows 3601; min 0.0; max 3600.0; integer_valued_fraction 1.0; non_decreasing_step_fraction 1.0
Strongest relations found by code's generic search: none with r2 >= 0.5

Known columns (named by the reference parser; shuffled order):
  c0: Ewe [V], <f4, min 0.912, max 0.934

Sample rows (3 evenly spaced rows of 3601):
row | c0 | TARGET
0 | 0.912 | 0
1800 | 0.927 | 1800
3600 | 0.934 | 3600"""
PROMPT = [
    {"role": "system", "content": agent.system_prompt(False)},
    {"role": "user", "content": "Dossier written by code:\n\n" + TOY_DOSSIER
                                + "\n\nName the hidden column TARGET. Answer with one JSON object as specified."},
]


def main() -> int:
    from openai import OpenAI

    key = os.environ.get("NEBIUS_API_KEY")
    if not key:
        print("NEBIUS_API_KEY is not set")
        return 1
    client = OpenAI(base_url=BASE_URL, api_key=key, timeout=180)
    log = ROOT / "runs2" / "api_check.jsonl"
    log.parent.mkdir(exist_ok=True)
    ids = sorted(m.id for m in client.models.list().data)
    print("NVIDIA models:", [i for i in ids if "emotron" in i])
    for m in (MODEL_FAST, MODEL_DEEP):
        print(f"  {m}: {'present' if m in ids else 'MISSING'}")
    cases = []
    for model, kws in ((MODEL_FAST, [{"enable_thinking": False}, {"enable_thinking": True}]),
                       (MODEL_DEEP, [{"enable_thinking": True, "medium_effort": True}])):
        for kw in kws:
            for mode in ("json_schema", "json_object"):
                cases.append((model, kw, mode, SCHEMA, PROMPT))
    # LTA's multi-step schema (action test|answer) with its own system prompt
    step_prompt = [{"role": "system", "content": agent.system_prompt(True)}, PROMPT[1]]
    cases.append((MODEL_FAST, {"enable_thinking": True}, "json_schema", agent.schema(True), step_prompt))
    ok_schema = {}
    for model, kw, mode, sch, prompt in cases:
        rf = {"type": "json_schema", "json_schema": sch} if mode == "json_schema" else {"type": "json_object"}
        t = time.time()
        entry = {"model": model, "kwargs": kw, "mode": mode}
        try:
            r = client.chat.completions.create(model=model, messages=prompt, max_tokens=8192, temperature=1.0,
                                               top_p=0.95, response_format=rf,
                                               extra_body={"chat_template_kwargs": {**kw, "force_nonempty_content": True}})
            ch = r.choices[0]
            u = r.usage
            rt = getattr(getattr(u, "completion_tokens_details", None), "reasoning_tokens", None)
            try:
                parsed = json.loads(ch.message.content or "")
            except Exception:
                parsed = None
            valid = isinstance(parsed, dict) and parsed.get("name") in sch["schema"]["properties"]["name"]["enum"]
            thought = bool(rt) or bool(getattr(ch.message, "reasoning_content", None))
            if kw.get("enable_thinking") and not thought:
                valid = False  # reasoning was switched off (no reasoning tokens and no reasoning_content)
            pin, pout = PRICES.get(model, (0, 0))
            entry.update(ok=True, finish_reason=ch.finish_reason, content=ch.message.content, valid=valid,
                         prompt_tokens=u.prompt_tokens, completion_tokens=u.completion_tokens, reasoning_tokens=rt,
                         latency_s=round(time.time() - t, 2),
                         cost_usd=(u.prompt_tokens * pin + u.completion_tokens * pout) / 1e6)
            if mode == "json_schema":
                ok_schema[(model, json.dumps(kw), sch["name"])] = valid
            print(f"{model.split('/')[-1]:28s} {json.dumps(kw):48s} {mode:11s} finish={ch.finish_reason} valid={valid} "
                  f"reasoning={rt} {entry['latency_s']}s -> {(ch.message.content or '')[:80]!r}")
        except Exception as exc:
            entry.update(ok=False, error=f"{type(exc).__name__}: {exc}")
            if mode == "json_schema":
                ok_schema[(model, json.dumps(kw), sch["name"])] = False
            print(f"{model.split('/')[-1]:28s} {json.dumps(kw):48s} {mode:11s} FAILED {type(exc).__name__}: {str(exc)[:200]}")
        entry["t"] = time.strftime("%Y-%m-%d %H:%M:%S")
        with log.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
    mode = "json_schema" if all(ok_schema.values()) else "json_object"
    print(f"\nRecommended --response-mode: {mode}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
