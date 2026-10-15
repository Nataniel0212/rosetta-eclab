"""Closed vocabulary of EC-Lab quantities and the answer key's synonym rules.

The model answers with one canonical name from this list (or abstains). The
list is yadg 7.0.1's column map plus vendor variable names seen in public
EC-Lab text exports (.mpt headers) that no parser maps yet; those act as
realistic distractors. Pre-registered merges (SYNONYMS) put names that the
vendor itself uses interchangeably, or that differ only by averaging over the
sampling interval, into one canonical entry: the data cannot tell them apart.

Scoring is on the canonical name. Units are recorded and reported separately,
because yadg's units are not always the vendor's (e.g. 'Q charge or discharge'
is C in yadg, mA.h in the vendor export).
"""

from __future__ import annotations

from functools import lru_cache

# canonical name -> aliases (all spellings that score as this canonical name)
SYNONYMS: dict[str, list[str]] = {
    "Ewe": ["Ewe", "<Ewe>", "Ecell"],
    "I": ["I", "<I>"],
    "Ece": ["Ece", "<Ece>"],
    "control": ["control", "control_V", "control_I"],
    "Pwe": ["Pwe", "P"],
    "Energy we charge": ["Energy we charge", "Energy charge"],
    "Energy we discharge": ["Energy we discharge", "Energy discharge"],
    "Q charge or discharge": ["Q charge or discharge", "Q charge/discharge"],
}

# Vendor variable names from public .mpt headers that yadg 7.0.1 does not map (distractors).
VENDOR_EXTRA: list[tuple[str, str | None]] = [
    ("Q charge/discharge", "mA·h"), ("P", "W"),
    ("x", None), ("Q discharge", "mA·h"), ("Q charge", "mA·h"), ("Capacity", "mA·h"),
    ("Efficiency", "%"), ("Pce", "W"), ("Pwe-ce", "W"), ("Ewe-Ece", "V"), ("Rce", "Ω"),
    ("Rwe-ce", "Ω"), ("Energy we-ce charge", "W·h"), ("Energy we-ce discharge", "W·h"),
    ("Re(Y)", "Ω⁻¹"), ("Im(Y)", "Ω⁻¹"), ("|Y|", "Ω⁻¹"), ("Phase(Y)", "deg"),
    ("Re(C)", "nF"), ("Im(C)", "nF"), ("|C|", "nF"), ("Phase(C)", "deg"),
    ("Tan(Delta)", None), ("Loss Angle(Delta)", "deg"), ("counter inc.", None),
]

ABSTAIN = "ABSTAIN"
DUPLICATE = "DUPLICATE"


def _alias_to_canonical() -> dict[str, str]:
    out = {}
    for canon, aliases in SYNONYMS.items():
        for a in aliases:
            out[a] = canon
    return out


ALIAS = _alias_to_canonical()


def canonical(name: str) -> str:
    return ALIAS.get(name, name)


@lru_cache(maxsize=1)
def vocabulary() -> list[dict]:
    """[{name (canonical), aliases, units}] sorted by name; stable across runs."""
    from yadg.extractors.eclab.mpr_columns import conflict_columns, data_columns, technique_dependent_ids

    entries: dict[str, dict] = {}

    def add(name: str, unit: str | None):
        c = canonical(name)
        e = entries.setdefault(c, {"name": c, "aliases": set(), "units": set()})
        if name != c:
            e["aliases"].add(name)
        if unit:
            e["units"].add(unit)

    for _dt, name, unit in data_columns.values():
        add(name, unit)
    for variants in conflict_columns.values():
        for _dt, name, unit in variants.values():
            add(name, unit)
    for cid, by_tech in technique_dependent_ids.items():
        for name in by_tech.values():
            add(name, data_columns[cid][2])
    for name, unit in VENDOR_EXTRA:
        add(name, unit)
    for canon, aliases in SYNONYMS.items():
        for a in aliases:
            add(a, None)
    out = []
    for c in sorted(entries, key=str.lower):
        e = entries[c]
        out.append({"name": c, "aliases": sorted(e["aliases"]), "units": sorted(e["units"])})
    return out


def names() -> list[str]:
    return [e["name"] for e in vocabulary()]


def render(exclude: list[str] | tuple = ()) -> str:
    """Vocabulary as shown to the model: one line per canonical name (minus excluded canonical names)."""
    lines = []
    for e in vocabulary():
        if e["name"] in exclude:
            continue
        extra = f" (also written {', '.join(e['aliases'])})" if e["aliases"] else ""
        unit = f" [{' or '.join(e['units'])}]" if e["units"] else " [no unit]"
        lines.append(f"- {e['name']}{extra}{unit}")
    return "\n".join(lines)


def normalise_unit(u: str | None) -> str:
    if not u:
        return ""
    u = u.strip().lower()
    for a, b in (("ω", "ohm"), ("·", ""), (".", ""), ("*", ""), ("^", ""), (" ", ""), ("µ", "u"), ("μ", "u")):
        u = u.replace(a, b)
    return u


if __name__ == "__main__":
    import sys

    sys.stdout.reconfigure(encoding="utf-8")
    print(len(vocabulary()), "canonical names")
    print(render())
