"""rosetta: name the columns of a BioLogic EC-Lab .mpr file that your parser does not know.

  rosetta FILE.mpr                      # name every column unknown to galvani (default reference parser)
  rosetta FILE.mpr --reference yadg     # ... unknown to yadg instead
  rosetta FILE.mpr --self-check         # also hide each known column in turn and report how many come back right
  rosetta FILE.mpr --json               # machine-readable output

No language model and no network access: names come from physical relations,
storage types and column shapes (namer/library.py), with a statistics fallback.
Every answer carries its evidence; "ABSTAIN" means the data does not support a name.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings

from spike2 import elim, vocab

from . import mprfile
from .evaluate import answer, gap_column


def unit_of(name: str) -> str:
    e = next((v for v in vocab.vocabulary() if v["name"] == name), None)
    return " or ".join(e["units"]) if e and e["units"] else ""


def own_group(path: str) -> str | None:
    """If the file is one of the public files behind the statistics table, leave its group out (no self-matches)."""
    from pathlib import Path

    from spike2 import corpus

    return corpus.source_group(Path(path).name)


def name_unknown(m: mprfile.MprData) -> list[dict]:
    known = [c for c in m.columns if c.known]
    several = len(m.unknown) > 1
    g = gap_column(known, m.unknown, vocab.names(), m.alternatives, m.doubtful) if several else None  # sibling gap
    out = []
    for i, c in enumerate(m.unknown):
        a = answer(c.values, c.dtype, known, m.technique, vocab.names(), exclude_group=own_group(m.path),
                   alone=not several, gap=g[1:] if g and g[0] == i else None)
        out.append({"column_id": c.raw_id, "dtype": c.dtype, "verdict": a["verdict"], "name": a.get("name"),
                    "unit": unit_of(a["name"]) if a["verdict"] == "NAME" else "", "confidence": a.get("confidence"),
                    "duplicate_of": a.get("duplicate_of", ""), "alternatives": a.get("alternatives", []),
                    "evidence": a.get("evidence", "")})
    return out


def self_check(m: mprfile.MprData) -> dict:
    known = [c for c in m.columns if c.known]
    gaps = elim.gaps([c.name for c in known], vocab.names())  # the file's own sibling gaps, if any
    rows = []
    for c in known:
        others = [o for o in known if o is not c]
        a = answer(c.values, c.dtype, others, m.technique, vocab.names(), exclude_group=own_group(m.path),
                   alone=not m.unknown, file_gaps=gaps)
        rows.append({"column": c.name, "verdict": a["verdict"], "answer": a.get("name"),
                     "correct": a["verdict"] == "NAME" and a.get("name") == vocab.canonical(c.name)})
    named = [r for r in rows if r["verdict"] == "NAME"]
    return {"hidden_one_at_a_time": len(rows), "named": len(named), "named_correctly": sum(r["correct"] for r in rows),
            "wrong": [r for r in named if not r["correct"]], "rows": rows}


def main(argv: list[str] | None = None) -> int:
    warnings.filterwarnings("ignore")
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(prog="rosetta", description=__doc__.splitlines()[0])
    ap.add_argument("file")
    ap.add_argument("--reference", choices=["galvani", "yadg"], default="galvani",
                    help="parser whose column map defines 'known' (default: galvani)")
    ap.add_argument("--self-check", action="store_true", help="hide each known column in turn and try to name it")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    try:
        m = mprfile.read(a.file, a.reference)
    except Exception as exc:
        print(f"rosetta: {exc}", file=sys.stderr)
        return 2
    result = {"file": a.file, "technique": m.technique, "rows": m.n_points, "record_bytes": m.record_size,
              "notes": m.notes, "unknown_columns": name_unknown(m)}
    if a.self_check:
        result["self_check"] = self_check(m)
    if a.json:
        print(json.dumps(result, indent=1, ensure_ascii=False, default=float))
        return 0
    print(f"{a.file}\n  technique {m.technique or 'unknown'}, {m.n_points} rows, {m.record_size} bytes per record")
    for n in m.notes:
        print(f"  {n}")
    if not result["unknown_columns"]:
        print(f"  every column ID is in {a.reference}'s column map; nothing to name")
    for u in result["unknown_columns"]:
        if u["verdict"] == "NAME":
            head = f"{u['name']} [{u['unit'] or 'no unit'}]  (confidence {u['confidence']})"
        elif u["verdict"] == "DUPLICATE":
            head = f"DUPLICATE of {u['duplicate_of']} (identical in every row)"
        else:
            head = "ABSTAIN (the data does not support a name)"
        print(f"\n  column ID {u['column_id']} ({u['dtype']}): {head}")
        print(f"    evidence: {u['evidence']}")
        if u["alternatives"]:
            print(f"    also considered: {', '.join(u['alternatives'])}")
    if a.self_check:
        s = result["self_check"]
        print(f"\n  self-check: hid each of {s['hidden_one_at_a_time']} known columns in turn; named {s['named']}, "
              f"{s['named_correctly']} correctly, {len(s['wrong'])} wrong, "
              f"{s['hidden_one_at_a_time'] - s['named']} abstained")
        for w in s["wrong"]:
            print(f"    wrong: {w['column']} -> {w['answer']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
