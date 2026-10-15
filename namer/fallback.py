"""Statistics fallback for columns no formula can identify (e.g. Ewe in a file without power or current).

A k-nearest-neighbour vote over scale-free column features, learned from the
public EC-Lab corpus, plus a technique prior used as a tie-breaker. It can only
name quantities it has seen, so its answers carry lower confidence than a
formula match. A column that looks like noise (no autocorrelation) is never
named by the fallback.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from spike2 import corpus, features, tasks, vocab


def lag1_autocorr(x) -> float:
    y = np.asarray(x, dtype="f8")
    y = y[np.isfinite(y)]
    if y.size < 4 or np.ptp(y) == 0:
        return 0.0
    a, b = y[:-1] - y[:-1].mean(), y[1:] - y[1:].mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den else 0.0


DATA = Path(__file__).resolve().parent / "data"


class Stats:
    """Column statistics of the public corpus. Built from the downloaded files once, then read from
    namer/data/stats.* so the command-line tool works without the corpus on disk."""

    def __init__(self, use_cache: bool = True):
        meta, feats = DATA / "stats.json", DATA / "stats_features.npy"
        if use_cache and meta.exists() and feats.exists():
            m = json.loads(meta.read_text(encoding="utf-8"))
            X = np.load(feats)
            self.rows = [(g, t, n, X[i]) for i, (g, t, n) in enumerate(m["rows"])]
            techs = defaultdict(dict)
            for tech, g, f, names in m["files"]:
                techs[tech][(g, f)] = set(names)
            self._finish(techs)
            return
        self._build()
        if use_cache:
            self.save()

    def save(self):
        DATA.mkdir(exist_ok=True)
        rows = [[g, t, n] for g, t, n, _ in self.rows]
        files = [[tech, g, f, sorted(names)] for tech, d in self.techs.items() for (g, f), names in d.items()]
        (DATA / "stats.json").write_text(json.dumps({"rows": rows, "files": files}, ensure_ascii=False),
                                         encoding="utf-8", newline="\n")
        np.save(DATA / "stats_features.npy", np.stack([r[3] for r in self.rows]))

    def _finish(self, techs):
        X = np.stack([r[3] for r in self.rows])
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-9
        self.Z = (X - self.mu) / self.sd
        self.techs = techs

    def _build(self):
        rows, techs = [], defaultdict(Counter)
        for p in corpus.corpus_files():
            try:
                fd = tasks.file_data(p.name)
            except Exception:
                continue
            if fd.n_points < tasks.MIN_ROWS:
                continue
            group = corpus.source_group(p.name)
            cols = tasks.usable_columns(fd)
            tcol = next((c for c in cols if c.name == "time"), None)
            names = {vocab.canonical(c.name) for c in cols}
            techs[fd.technique][(group, p.name)] = names  # type: ignore[index]
            for c in cols:
                if features.is_constant(c.values):
                    continue
                f = features.knn_features(c.values, c.dtype, None if tcol is None or tcol is c else tcol.values,
                                          [o.values for o in cols if o is not c])
                rows.append((group, fd.technique, vocab.canonical(c.name), f))
        self.rows = rows
        self._finish(techs)

    def prior(self, technique, exclude_group: str | None, all_techniques: bool = False) -> Counter:
        cnt = Counter()
        groups = self.techs.values() if all_techniques else [self.techs.get(technique, {})]
        for files in groups:
            for (g, _f), names in files.items():
                if g != exclude_group:
                    cnt.update(names)
        return cnt

    def expected_missing(self, present: set[str], exclude_group: str | None = None, k: int = 5,
                         share: float = 0.8) -> list[str]:
        """Names recorded by at least `share` of the k public files whose column sets are most like this one,
        but missing here."""
        files = [names for t in self.techs.values() for (g, _f), names in t.items() if g != exclude_group]
        if not files or not present:
            return []
        sim = sorted(files, key=lambda n: -len(n & present) / len(n | present))[:k]
        counts = Counter(x for n in sim for x in n)
        return sorted(n for n, c in counts.items() if c >= share * len(sim) and n not in present)

    def cooccur(self, cand: str, sib: str, exclude_group: str | None = None) -> float:
        """Share of public files containing sib that also contain cand."""
        with_sib = [names for files in self.techs.values() for (g, _f), names in files.items()
                    if g != exclude_group and sib in names]
        return sum(cand in n for n in with_sib) / len(with_sib) if with_sib else 0.0

    def predict(self, target, dtype, known_cols, technique, allowed: set[str], present: set[str],
                exclude_group: str | None = None, k: int = 5):
        tcol = next((c.values for c in known_cols if c.name == "time"), None)
        f = (features.knn_features(target, dtype, tcol, [c.values for c in known_cols]) - self.mu) / self.sd
        idx = [i for i, r in enumerate(self.rows) if r[0] != exclude_group and r[2] in allowed and r[2] not in present]
        if not idx:
            return None, 0.0, []
        d = np.linalg.norm(self.Z[idx] - f, axis=1)
        votes = defaultdict(float)
        for j in np.argsort(d)[:k]:
            votes[self.rows[idx[j]][2]] += 1.0 / (d[j] + 1e-6)
        pr = self.prior(technique, exclude_group)
        total = sum(votes.values())
        ranked = sorted(votes, key=lambda n: (-votes[n], -pr.get(n, 0), n))
        return ranked[0], votes[ranked[0]] / total, ranked[1:3]
