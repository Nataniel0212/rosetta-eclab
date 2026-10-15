"""Freeze the pre-registration: hash everything that decides the v2 spike.

Writes prereg2/LOCK.json with SHA-256 of every file in prereg2/ and spike2/,
the corpus manifest, the task file
(data/v2/tasks.jsonl, git-ignored, so its hash is the record), the dossier set,
the vocabulary, package versions and the response mode per arm. Commit
prereg2/ and spike2/ right after; run.py refuses the test split until the
hashes match and both folders are committed and clean.

Usage:  python -m spike2.lock --by "<name>" --response-mode json_schema
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(p: Path) -> str:
    """SHA-256 with CRLF normalised to LF, so a fresh clone gives the same hash (.gitattributes eol=lf)."""
    return hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--by", required=True)
    ap.add_argument("--response-mode", required=True, choices=["json_schema", "json_object", "none"])
    a = ap.parse_args()
    files = sorted([*(ROOT / "prereg2").glob("*"), *(ROOT / "spike2").glob("*.py"),
                    ROOT / "data" / "external" / "manifest.json"])
    files = [f for f in files if f.is_file() and f.name != "LOCK.json"]
    manifest = json.loads((ROOT / "prereg2" / "tasks_manifest.json").read_text(encoding="utf-8"))
    lock = {
        "locked_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "locked_by": a.by,
        "response_mode": a.response_mode,
        "all_dossiers_sha256": manifest["all_dossiers_sha256"],
        "vocabulary_sha256": manifest["vocabulary_sha256"],
        "tasks_jsonl_sha256": sha(ROOT / "data" / "v2" / "tasks.jsonl"),
        "packages": {p: version(p) for p in ("yadg", "galvani", "numpy", "openai")},
        "files": {str(f.relative_to(ROOT)).replace("\\", "/"): sha(f) for f in files},
    }
    if lock["tasks_jsonl_sha256"] != manifest.get("tasks_jsonl_sha256"):
        raise SystemExit("data/v2/tasks.jsonl does not match prereg2/tasks_manifest.json; rebuild with python -m spike2.tasks")
    lock["combined_sha256"] = hashlib.sha256(json.dumps(lock["files"], sort_keys=True).encode()).hexdigest()
    (ROOT / "prereg2" / "LOCK.json").write_text(json.dumps(lock, indent=1), encoding="utf-8", newline="\n")
    print(f"prereg2/LOCK.json written; combined sha256 {lock['combined_sha256']}")
    print('Now commit: git add prereg2 spike2 && git commit -m "prereg v2: freeze before the first test-split call"')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
