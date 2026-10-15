#!/bin/sh
# Development score over everything except holdout2: pool + v2 dev + v2 test + holdout (spent).
.venv/Scripts/python -X utf8 -m namer.evaluate --pool 2>&1 | head -1
for s in dev test holdout holdout2 holdout3 holdout4; do .venv/Scripts/python -X utf8 -m namer.evaluate --split $s >/dev/null; .venv/Scripts/python -X utf8 -m bench.claude_cli --split $s --score rules | grep rules | sed "s/^/$s /"; done
