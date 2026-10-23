"""galvani 0.5.0's column map, frozen in this repository.

The benchmarks and the namer's expected storage types use this copy, so their
answers do not depend on which galvani (if any) is installed. namer/galvani_ref.py
uses the installed galvani for "known to galvani" instead.

  python -m spike2.galvani_snapshot --write   # regenerate from an installed galvani 0.5.0
"""

from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path

SNAPSHOT = Path(__file__).resolve().parent / "galvani_0.5.0_colmap.json"
SOURCE = "galvani 0.5.0 (GPL-3.0), galvani/BioLogic.py, VMPdata_colID_dtype_map: column ID -> [name/unit, numpy dtype]"


@lru_cache(maxsize=1)
def snapshot_map() -> dict[int, tuple[str, str]]:
    d = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    return {int(k): (v[0], v[1]) for k, v in d["columns"].items()}


def write() -> int:
    from importlib.metadata import PackageNotFoundError, version

    try:
        ok = version("galvani") == "0.5.0"
    except PackageNotFoundError:
        ok = False
    if not ok:
        print("needs galvani 0.5.0 installed (pip install galvani==0.5.0)", file=sys.stderr)
        return 1
    from galvani import BioLogic

    rows = [f"  {json.dumps(str(k))}: {json.dumps([v[0], v[1]], ensure_ascii=False)}"
            for k, v in sorted(BioLogic.VMPdata_colID_dtype_map.items())]
    text = '{\n"source": ' + json.dumps(SOURCE) + ',\n"columns": {\n' + ",\n".join(rows) + "\n}\n}\n"
    SNAPSHOT.write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {SNAPSHOT} ({len(rows)} columns)")
    return 0


if __name__ == "__main__":
    raise SystemExit(write() if "--write" in sys.argv else 0)
