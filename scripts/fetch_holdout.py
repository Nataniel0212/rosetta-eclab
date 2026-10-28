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
import io
import json
import os
import sys
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "holdout"
MAGIC = b"BIO-LOGIC MODULAR FILE"
MAX_MB = 30  # the largest .mpr a frozen selection may take (holdout7_sources.json: max_mb); exports may be larger


_LAST: dict[str, float] = {}


def fetch(url: str, headers: dict | None = None, cap: int | None = None):
    """One polite request: at least a second between requests to a host, and retries with backoff when the
    server answers 429 or 503. Returns (status, headers, body); with `cap`, a longer body raises OSError."""
    import time
    import urllib.error

    host = urllib.parse.urlparse(url).netloc
    for attempt in range(8):
        wait = 1.0 - (time.time() - _LAST.get(host, 0.0))
        if wait > 0:
            time.sleep(wait)
        _LAST[host] = time.time()
        req = urllib.request.Request(url, headers={"User-Agent": "rosetta-holdout/1", **(headers or {})})
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                body = r.read() if cap is None else r.read(cap + 1)
                if cap is not None and len(body) > cap:
                    raise OSError(f"{url}: reply longer than {cap} bytes")
                return r.status, r.headers, body
        except urllib.error.HTTPError as exc:
            if exc.code not in (429, 503) or attempt == 7:
                raise
            retry = exc.headers.get("Retry-After") if exc.headers else None
            time.sleep(min(120.0, float(retry) if retry and retry.isdigit() else 5.0 * 2 ** attempt))
    raise OSError(f"{url}: gave up")


def get(url: str) -> bytes:
    return fetch(url)[2]


def zenodo(record: str, max_mb: float | None = None) -> list[tuple[str, str]]:
    d = json.loads(get(f"https://zenodo.org/api/records/{record}"))
    return [(f["key"], f["links"]["self"]) for f in d["files"] if f["key"].lower().endswith((".mpr", ".txt", ".mpt"))
            and (max_mb is None or f["size"] <= max_mb * 1e6)]


class RangeFile(io.RawIOBase):
    """A remote file read with HTTP Range requests, so zipfile can read single members of a large archive."""

    def __init__(self, url: str):
        self.url, self.pos = url, 0
        status, headers, _ = fetch(url, {"Range": "bytes=0-0"}, cap=1024)  # a server that ignores Range stops here
        if status != 206 or not headers.get("Content-Range"):
            raise OSError(f"{url}: no range support")
        self.size = int(headers["Content-Range"].split("/")[-1])

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else (self.pos + off if whence == 1 else self.size + off)
        return self.pos

    def readinto(self, b):
        if self.pos >= self.size or not len(b):
            return 0
        end = min(self.size, self.pos + len(b)) - 1
        status, _, data = fetch(self.url, {"Range": f"bytes={self.pos}-{end}"}, cap=end - self.pos + 1)
        if status != 206 or len(data) != end - self.pos + 1:
            raise OSError(f"{self.url}: range request answered with status {status}, {len(data)} bytes")
        b[:len(data)] = data
        self.pos += len(data)
        return len(data)


def archive_members(url: str, max_mb: float = 30, prefix: str = "") -> list[tuple[str, str, str]]:
    """EC-Lab files inside a remote .zip: (flat name, archive URL, member path). Keeps .mpr and .mpt members and
    .txt members whose stem matches an .mpr; the flat name joins the member's folders with '__'."""
    z = zipfile.ZipFile(io.BufferedReader(RangeFile(url), buffer_size=1 << 16))
    infos = [i for i in z.infolist() if not i.is_dir() and i.file_size <= max_mb * 1e6]
    stems = {PurePosixPath(i.filename).with_suffix("").as_posix().lower()
             for i in infos if i.filename.lower().endswith(".mpr")}
    out = []
    for i in infos:
        low = i.filename.lower()
        stem = PurePosixPath(i.filename).with_suffix("").as_posix().lower()
        if low.endswith((".mpr", ".mpt")) or (low.endswith(".txt") and stem in stems):
            parts = [x.replace(":", "_") for x in PurePosixPath(i.filename.replace("\\", "/")).parts
                     if x.strip("/") not in ("", ".", "..")]  # never an absolute path, a parent or a drive
            while len(parts) > 1 and len(prefix + "__".join(parts)) > 120:
                parts = parts[1:]  # drop leading folders, never the file name
            out.append((prefix + "__".join(parts), url, i.filename))
    return out


def zenodo_archives(record: str, max_mb: float = 30) -> list[tuple[str, str, str]]:
    d = json.loads(get(f"https://zenodo.org/api/records/{record}"))
    out = []
    for f in d["files"]:
        if f["key"].lower().endswith(".zip"):
            url = f"https://zenodo.org/api/records/{record}/files/{urllib.parse.quote(f['key'])}/content"
            out += archive_members(url, max_mb)
    return out


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
    # sixth hold-out: the version-0 files of hold-out 3's two PEM fuel-cell records, which could not be
    # decoded then (2026-10-27); the namer has never run on them
    "holdout6": {
        "zenodo_3631173": ("CC-BY-4.0", lambda: zenodo("3631173")),
        "zenodo_3631156": ("CC-BY-4.0", lambda: zenodo("3631156")),
    },
}



def selection(name: str) -> dict:
    """A frozen file selection, holdout/<name>_sources.json: {source: (licence, lister of its files)}."""
    path = ROOT / "holdout" / f"{name}_sources.json"
    if not path.exists():
        return {}
    sel = json.loads(path.read_text(encoding="utf-8"))
    return {r["src"]: (r["licence"], (lambda r=r: [tuple(e) for e in r["files"]])) for r in sel["records"]}


# seventh hold-out: up to three files from each openly licensed record found by a broad search on 2026-10-27
# (Zenodo archives, figshare, Dataverse, Mendeley Data, institutional repositories, GitHub); the rule and every
# chosen file are in holdout/holdout7_sources.json
SETS["holdout7"] = selection("holdout7")

def plain_name(name: str) -> bool:
    """A bare file or folder name: no path separator, drive, parent or empty part."""
    return bool(name) and name not in (".", "..") and not any(ch in name for ch in "/\\:")


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
        if not plain_name(src):
            print(f"{src}: not a plain folder name, skipped", file=sys.stderr)
            continue
        d = OUT / src
        d.mkdir(parents=True, exist_ok=True)
        archives: dict[str, zipfile.ZipFile] = {}
        for entry in files:
            name, url = entry[0], entry[1]
            member = entry[2] if len(entry) > 2 else None  # a member of a remote .zip
            if not plain_name(name):
                print(f"  {name}: not a plain file name, skipped", file=sys.stderr)
                continue
            dest = d / name
            if any(m["file"] == f"{src}/{name}" for m in manifest):
                continue  # two entries with the same name (same file name in two folders): the first one is kept
            if not dest.exists():
                part = dest.with_name(dest.name + ".part")  # an interrupted write never looks like a whole file
                try:
                    if member is None:
                        part.write_bytes(get(url))
                    else:
                        if url not in archives:
                            archives[url] = zipfile.ZipFile(io.BufferedReader(RangeFile(url), buffer_size=1 << 16))
                        if name.lower().endswith(".mpr") and archives[url].getinfo(member).file_size > MAX_MB * 1e6:
                            raise OSError(f"member larger than {MAX_MB} MB")
                        part.write_bytes(archives[url].read(member))
                    os.replace(part, dest)
                except Exception as exc:
                    print(f"  {name}: {exc}", file=sys.stderr)
                    part.unlink(missing_ok=True)
                    continue
            raw = dest.read_bytes()
            if name.lower().endswith(".mpr") and not raw.startswith(MAGIC):
                print(f"  {name}: not an EC-Lab file, skipped")
                dest.unlink()
                continue
            manifest.append({"source": src, "licence": lic, "file": f"{src}/{name}", "bytes": len(raw),
                             "sha256": hashlib.sha256(raw).hexdigest(), "url": url,
                             **({"member": member} if member else {})})
        print(f"{src}: {sum(1 for m in manifest if m['source'] == src)} files")
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
