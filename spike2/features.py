"""What code knows about one column: statistics, generic relations, kNN features.

Everything the model is shown is computed here. The relation search is
deliberately generic (each known column alone, linearly and as a time
integral); products, ratios, resets and domain formulas are left to the
model's hypotheses (dsl.py).
"""

from __future__ import annotations

import numpy as np

from . import dsl


def _f8(v) -> np.ndarray:
    return np.asarray(v, dtype="f8")


def is_constant(v) -> bool:
    x = _f8(v)
    x = x[np.isfinite(x)]
    return x.size == 0 or float(np.max(x)) == float(np.min(x))


def identical(a, b) -> bool:
    a, b = _f8(a), _f8(b)
    return a.shape == b.shape and np.allclose(a, b, rtol=1e-5, atol=1e-9, equal_nan=True)


def stats(v, dtype: str) -> dict:
    x = _f8(v)
    fin = np.isfinite(x)
    y = x[fin]
    n = len(x)
    if y.size == 0:
        return {"rows": n, "finite_fraction": 0.0}
    d = np.diff(y)
    rng = float(np.ptp(y))
    drops = int((d < -0.05 * rng).sum()) if rng > 0 else 0
    rises = int((d > 0.05 * rng).sum()) if rng > 0 else 0
    s = np.sign(y)
    out = {
        "rows": n,
        "dtype": dtype,
        "finite_fraction": round(float(fin.mean()), 4),
        "min": float(f"{y.min():.6g}"),
        "max": float(f"{y.max():.6g}"),
        "mean": float(f"{y.mean():.6g}"),
        "std": float(f"{y.std():.6g}"),
        "distinct_values": int(np.unique(y).size),
        "integer_valued_fraction": round(float(np.mean(np.isclose(y, np.round(y)))), 4),
        "fraction_positive": round(float((y > 0).mean()), 4),
        "fraction_negative": round(float((y < 0).mean()), 4),
        "fraction_zero": round(float((y == 0).mean()), 4),
        "non_decreasing_step_fraction": round(float((d >= 0).mean()), 4) if d.size else None,
        "non_increasing_step_fraction": round(float((d <= 0).mean()), 4) if d.size else None,
        "large_drops (>5% of range)": drops,
        "large_rises (>5% of range)": rises,
        "sign_changes": int((s[1:] * s[:-1] < 0).sum()),
    }
    pos = y[y > 0]
    if pos.size > 3 and (pos.size == y.size):
        r = pos[1:] / pos[:-1]
        med = float(np.median(r))
        if med != 1.0:
            out["geometric_spacing_fraction"] = round(float(np.mean(np.abs(r / med - 1) < 0.05)), 4)
    return out


def relations(x, known: dict[str, np.ndarray], time_alias: str | None, top: int = 6) -> list[dict]:
    """Generic search: x ~ a*c + b and x ~ a*cumsum(c*dt) + b for every known column c."""
    out = []
    dt = dsl._diff0(_f8(known[time_alias])) if time_alias else None
    for alias, c in known.items():
        c = _f8(c)
        if is_constant(c):
            continue
        f = dsl.fit(x, c)
        if "r2" in f:
            out.append({"hypothesis": f"{alias}", **f})
        if dt is not None and alias != time_alias:
            g = dsl.fit(x, np.cumsum(np.nan_to_num(c * dt)))
            if "r2" in g:
                out.append({"hypothesis": f"cumsum({alias}*diff0({time_alias}))", **g})
    out.sort(key=lambda r: -r["r2"])
    return [r for r in out[:top] if r["r2"] >= 0.5]


# --- kNN features (scale-free) -------------------------------------------------

DTYPES = ["<f8", "<f4", "<u2", "<u4"]


def knn_features(v, dtype: str, time_values=None, known_values: list | None = None) -> np.ndarray:
    x = _f8(v)
    y = x[np.isfinite(x)]
    if y.size < 2:
        y = np.zeros(2)
    d = np.diff(y)
    rng = float(np.ptp(y))
    amax = float(np.abs(y).max())
    feats = [1.0 if dtype == t else 0.0 for t in DTYPES]
    feats += [
        float(np.mean(np.isclose(y, np.round(y)))),
        float((y > 0).mean()), float((y < 0).mean()), float((y == 0).mean()),
        float((d >= 0).mean()), float((d <= 0).mean()),
        float(np.unique(y).size / y.size),
        float(np.log10(amax + 1e-12)),
        float(np.log10(rng + 1e-12)),
        float((d < -0.05 * rng).sum() / y.size) if rng > 0 else 0.0,
    ]
    corr_t = 0.0
    if time_values is not None and not is_constant(time_values) and rng > 0:
        t = _f8(time_values)
        m = np.isfinite(t) & np.isfinite(x)
        if m.sum() > 2:
            corr_t = float(np.corrcoef(t[m], x[m])[0, 1])
    feats.append(corr_t if np.isfinite(corr_t) else 0.0)
    best = 0.0
    for k in known_values or []:
        k = _f8(k)
        if is_constant(k) or rng == 0:
            continue
        m = np.isfinite(k) & np.isfinite(x)
        if m.sum() > 2:
            r = np.corrcoef(k[m], x[m])[0, 1]
            if np.isfinite(r):
                best = max(best, abs(float(r)))
    feats.append(best)
    return np.asarray(feats, dtype="f8")


def fmt(v: float) -> str:
    if v is None or not np.isfinite(v):
        return "nan"
    return f"{v:.6g}"
