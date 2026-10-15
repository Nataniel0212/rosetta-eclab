"""Fetch public EC-Lab files for the hold-out benchmark (never used for development).

Sources (all openly licensed, none in yadg's or galvani's test suites):
  - Zenodo 12205268, CC BY 4.0: "Supporting data for 'How Reference Electrodes Improve Our
    Understanding of Degradation Processes in Half- and Full-Cell Potassium-Ion Battery Setups'"
    (.mpr with vendor .txt exports)
  - Battery Data Format (Battery Data Alliance), Apache-2.0: tests/data/mpr
  - navani, MIT: Example_data

Files go to data/holdout/<source>/ (git-ignored); a manifest with SHA-256 is written next to them.

Usage:  python scripts/fetch_holdout.py [holdout|holdout2]
"""

from __future__ import annotations

import hashlib
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "holdout"
MAGIC = b"BIO-LOGIC MODULAR FILE"


def get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "rosetta-holdout/1"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return r.read()


def zenodo(record: str, max_mb: float | None = None) -> list[tuple[str, str]]:
    d = json.loads(get(f"https://zenodo.org/api/records/{record}"))
    return [(f["key"], f["links"]["self"]) for f in d["files"] if f["key"].lower().endswith((".mpr", ".txt", ".mpt"))
            and (max_mb is None or f["size"] <= max_mb * 1e6)]


def github_dir(repo: str, path: str, ref: str = "HEAD") -> list[tuple[str, str]]:
    api = f"https://api.github.com/repos/{repo}/contents/{urllib.parse.quote(path)}?ref={ref}"
    items = json.loads(get(api))
    out = []
    for it in items:
        if it["type"] == "file" and it["name"].lower().endswith((".mpr", ".mpt", ".txt")):
            out.append((it["name"], it["download_url"]))
    return out


SETS = {
    "holdout": {
        "zenodo_12205268": ("CC-BY-4.0", lambda: zenodo("12205268")),
        "bdf": ("Apache-2.0", lambda: github_dir("battery-data-alliance/battery-data-format", "tests/data/mpr")),
        "navani": ("MIT", lambda: github_dir("be-smith/navani", "Example_data")),
    },
    # second hold-out (frozen before the next development round): water electrolysis (EIS, CP, LSV) and graphite
    "holdout2": {
        "zenodo_19222974": ("CC-BY-4.0", lambda: zenodo("19222974")),
        "zenodo_15747202": ("CC-BY-4.0", lambda: zenodo("15747202")),
    },
    # third hold-out: PEM fuel cells (older EC-Lab versions) and lithiation of layered conductors (GITT, GCD)
    "holdout3": {
        "zenodo_3631173": ("CC-BY-4.0", lambda: zenodo("3631173")),
        "zenodo_3631156": ("CC-BY-4.0", lambda: zenodo("3631156")),
        "zenodo_17244893": ("CC-BY-4.0", lambda: zenodo("17244893", max_mb=12)),
    },
    # fourth hold-out: commercial solid-state batteries, cycling and EIS
    "holdout4": {
        "zenodo_18925051": ("CC-BY-4.0", lambda: zenodo("18925051", max_mb=16)),
    },
    # fifth hold-out: every other CC BY Zenodo record with EC-Lab .mpr files found on 2026-10-15 (filetype:mpr)
    "holdout5": {
        "zenodo_10889611": ("CC-BY-4.0", lambda: zenodo("10889611", max_mb=30)),
        "zenodo_14809707": ("CC-BY-4.0", lambda: zenodo("14809707")),
        "zenodo_15211416": ("CC-BY-4.0", lambda: zenodo("15211416")),
        "zenodo_15263065": ("CC-BY-4.0", lambda: zenodo("15263065")),
        "zenodo_15747326": ("CC-BY-4.0", lambda: zenodo("15747326")),
        "zenodo_3676445": ("CC-BY-4.0", lambda: zenodo("3676445")),
    },
}


def main() -> int:
    global OUT
    which = sys.argv[1] if len(sys.argv) > 1 else "holdout"
    OUT = ROOT / "data" / which
    manifest = []
    for src, (lic, lister) in SETS[which].items():
        try:
            files = lister()
        except Exception as exc:
            print(f"{src}: listing failed: {exc}", file=sys.stderr)
            continue
        d = OUT / src
        d.mkdir(parents=True, exist_ok=True)
        for name, url in files:
            dest = d / name
            if not dest.exists():
                try:
                    dest.write_bytes(get(url))
                except Exception as exc:
                    print(f"  {name}: {exc}", file=sys.stderr)
                    continue
            raw = dest.read_bytes()
            if name.lower().endswith(".mpr") and not raw.startswith(MAGIC):
                print(f"  {name}: not an EC-Lab file, skipped")
                dest.unlink()
                continue
            manifest.append({"source": src, "licence": lic, "file": f"{src}/{name}", "bytes": len(raw),
                             "sha256": hashlib.sha256(raw).hexdigest(), "url": url})
        print(f"{src}: {sum(1 for m in manifest if m['source'] == src)} files")
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
