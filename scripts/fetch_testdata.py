"""Fetch third-party EC-Lab test files into data/external/ (git-ignored).

The files belong to the yadg and galvani projects (both GPL-3.0). They are
downloaded at run time from pinned commits and never committed to this repo.
Each download is checked against the size GitHub reports, its SHA-256 is
recorded, and DATA_SOURCES.md is rewritten with source, commit and licence.

Usage:  .venv/Scripts/python scripts/fetch_testdata.py [--force]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "external"

YADG_COMMIT = "61d55fe1fcf6cdd165b09a17ca99e261e1e34f50"
GALVANI_COMMIT = "d4a5f444b16eae57d2edfad6df774ead95c92fd6"

YADG_RAW = f"https://raw.githubusercontent.com/dgbowl/yadg/{YADG_COMMIT}/tests/test_x_eclab/"
# galvani stores its test data in Git LFS, so the media endpoint is needed.
GALVANI_LFS = f"https://media.githubusercontent.com/media/echemdata/galvani/{GALVANI_COMMIT}/tests/testdata/"

# (group, filename, role)
YADG_FILES = [
    # Real format drifts that break galvani 0.5.0 today (kill-test files)
    ("drift", "gcpl.issue_230.mpr", "column ID 115 unknown to galvani 0.5.0"),
    ("drift", "gcpl.issue_230.mpt", "vendor text export (hidden answer key)"),
    ("drift", "peis.issue_225.with_ece.mpr", "column ID 215 unknown to galvani 0.5.0"),
    ("drift", "peis.issue_225.with_ece.mpt", "vendor text export (hidden answer key)"),
    # Backtest #149: EC-Lab 11.50 module header change (breaks galvani 0.3.0)
    ("backtest_149", "gcpl.issue_149.mpr", "primary backtest file"),
    ("backtest_149", "gcpl.issue_149.mpt", "vendor text export"),
    ("backtest_149", "ca.issue_149.mpr", "header change"),
    ("backtest_149", "cp.issue_149.mpr", "header change"),
    ("backtest_149", "cv.issue_149.mpr", "header change"),
    ("backtest_149", "geis.issue_149.mpr", "header change"),
    ("backtest_149", "mb.issue_149.mpr", "header change"),
    ("backtest_149", "ocv.issue_149.mpr", "header change"),
    ("backtest_149", "peis.issue_149.mpr", "header change"),
    # Read by galvani 0.5.0 today; source file for injections and sanity
    ("sanity", "bcd.issue_241.mpr", "readable by galvani 0.5.0 (not a drift case)"),
    ("sanity", "bcd.issue_241.mpt", "vendor text export"),
    ("sanity", "gcpl.mpr", "readable by both parsers"),
    ("sanity", "gcpl.mpt", "vendor text export"),
]

# Spike v2 corpus: every other EC-Lab .mpr in yadg's test suite at the pinned commit,
# for the leave-one-column-out naming benchmark (columns named by yadg's map).
YADG_CORPUS = [
    "ca.issue_134.mpr", "ca.mpr", "coc.issue_185.mpr", "cov.issue_185.mpr", "cp.issue_61.mpr", "cp.mpr",
    "cv.issue_217.mpr", "cv.mpr", "cva.issue_135.mpr", "cva.issue_202.mpr", "gcpl.issue_175.mpr",
    "gcpl.issue_211.mpr", "gcpl.issue_226.CxN.mpr", "gcpl.issue_226.I.mpr", "gcpl.issue_228.mpr",
    "gcpl.pr_182.1.mpr", "gcpl.pr_182.2.mpr", "geis.mpr", "lsv.issue_195.mpr", "lsv.mpr",
    "mb.issue_218.mpr", "mb.issue_223.mpr", "mb.issue_95.mpr", "mb.mpr", "mp.issue_183.mpr", "ocv.mpr",
    "peis.issue_225.ewe_ece.mpr", "peis.issue_225.no_ece.mpr", "peis.mpr", "vsp_ocv_with.mpr",
    "vsp_ocv_wo.mpr", "vsp_peis_with.mpr", "wait.mpr", "zir.mpr",
]
YADG_FILES += [("corpus_v2", f, "spike v2 naming benchmark") for f in YADG_CORPUS]

GALVANI_FILES = [
    ("sanity", f"bio_logic{i}.{ext}", "galvani regression file")
    for i in range(1, 7)
    for ext in ("mpr", "mpt")
    if not (ext == "mpt" and i == 3)  # bio_logic3 has no .mpt upstream
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "rosetta-fetch/1"})
    with urllib.request.urlopen(req, timeout=120) as resp, dest.open("wb") as f:
        while chunk := resp.read(1 << 20):
            f.write(chunk)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true", help="re-download existing files")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    manifest = []
    failures = []
    jobs = [("yadg", YADG_RAW, g, f, r) for g, f, r in YADG_FILES] + [
        ("galvani", GALVANI_LFS, g, f, r) for g, f, r in GALVANI_FILES
    ]
    for project, base, group, name, role in jobs:
        dest = OUT / name
        url = base + name
        if args.force or not dest.exists():
            try:
                download(url, dest)
            except Exception as exc:  # network errors are reported, not fatal
                failures.append(f"{name}: {exc}")
                continue
        head = dest.read_bytes()[:64]
        if head.startswith(b"version https://git-lfs"):
            failures.append(f"{name}: got a Git LFS pointer instead of the file")
            continue
        manifest.append(
            {
                "file": name,
                "project": project,
                "group": group,
                "role": role,
                "bytes": dest.stat().st_size,
                "sha256": sha256(dest),
                "url": url,
            }
        )
        print(f"ok  {name:34s} {dest.stat().st_size:>10d} B")

    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if failures:
        print("\nFAILED:", *failures, sep="\n  ", file=sys.stderr)
        return 1
    print(f"\n{len(manifest)} files in {OUT}")
    return 0


def write_sources(manifest: list[dict]) -> None:
    lines = [
        "# Data sources",
        "",
        "Test files are downloaded at run time by `scripts/fetch_testdata.py` into",
        "`data/external/`, which is git-ignored. They are **not** part of this repository.",
        "",
        "| Project | Licence | Pinned commit |",
        "|---|---|---|",
        f"| [yadg](https://github.com/dgbowl/yadg) (`tests/test_x_eclab/`) | GPL-3.0 | `{YADG_COMMIT}` |",
        f"| [galvani](https://github.com/echemdata/galvani) (`tests/testdata/`, Git LFS) | GPL-3.0 | `{GALVANI_COMMIT}` |",
        "",
        "| File | Project | Group | Role | Bytes | SHA-256 |",
        "|---|---|---|---|---|---|",
    ]
    for m in manifest:
        lines.append(
            f"| `{m['file']}` | {m['project']} | {m['group']} | {m['role']} | {m['bytes']} | `{m['sha256'][:16]}…` |"
        )
    (ROOT / "DATA_SOURCES.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
