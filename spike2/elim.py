"""Name elimination: the code-only trick that solves a column from its siblings' names.

If the known columns include |Ewe h2|, |Ewe h3|, |Ewe h5|, ... then the
vocabulary name one token away from at least two of them (|Ewe h4|) is the
gap. That needs no data and no model, so tasks it settles are not evidence
for the model (found by the harness review, 2026-10-14). The rule:

  candidates = vocabulary names not already present among the known columns
  score(v)   = number of known column names exactly one token away from v
  answer     = the unique candidate with the highest score, if that score >= 2
"""

from __future__ import annotations

import re

from . import vocab

_TOKEN = re.compile(r"[A-Za-z]+|\d+|[^A-Za-z\d\s]")


def tokens(name: str) -> list[str]:
    return _TOKEN.findall(name)


def one_token_apart(a: str, b: str) -> bool:
    """Siblings: same token count, exactly one token differs, and they share at least one word
    (so "Ns"/"dQ" or "|C|"/"|I|", which share nothing or only punctuation, are not siblings)."""
    ta, tb = tokens(a), tokens(b)
    if len(ta) != len(tb) or len(ta) < 2:
        return False
    diff = sum(x != y for x, y in zip(ta, tb))
    shared_word = any(x == y and x.isalpha() for x, y in zip(ta, tb))
    return diff == 1 and shared_word


def eliminate(known_names: list[str], vocabulary: list[str]) -> str | None:
    present = {vocab.canonical(n) for n in known_names}
    spellings = set(known_names) | present
    best, best_score, tie = None, 0, False
    for v in vocabulary:
        if v in present:
            continue
        forms = [v] + [a for a, c in vocab.ALIAS.items() if c == v and a != v]
        score = max(sum(one_token_apart(f, k) for k in spellings) for f in forms)
        if score > best_score:
            best, best_score, tie = v, score, False
        elif score == best_score and score > 0:
            tie = True
    if best_score >= 2 and not tie:
        return best
    return None


def gaps(known_names: list[str], vocabulary: list[str]) -> set[str]:
    """Every name tied for the best score, if that score is at least 2: the file's sibling gaps (eliminate()
    returns one only when it is unique). In --self-check no hidden known column can fill any of them."""
    present = {vocab.canonical(n) for n in known_names}
    spellings = set(known_names) | present
    scores = {}
    for v in vocabulary:
        if v in present:
            continue
        forms = [v] + [a for a, c in vocab.ALIAS.items() if c == v and a != v]
        scores[v] = max(sum(one_token_apart(f, k) for k in spellings) for f in forms)
    best = max(scores.values(), default=0)
    return {v for v, s in scores.items() if s == best} if best >= 2 else set()
