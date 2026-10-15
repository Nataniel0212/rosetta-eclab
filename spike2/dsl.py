"""Safe expression language for hypotheses about a column.

The model writes an expression over the file's known columns (aliases c0, c1,
...) and code evaluates it with numpy: nothing is exec'd, only a whitelisted
AST (numbers, column aliases, + - * / **, unary minus and the functions
below). Code then fits target ~ a * expr + b and reports how much of the
column the hypothesis explains, so the model never has to compute a number.
"""

from __future__ import annotations

import ast

import numpy as np

MAX_LEN = 300


def _diff0(x):
    """First difference with a leading 0, same length as x."""
    return np.concatenate([[0.0], np.diff(x)])


def _cumsum_by(x, key):
    """Cumulative sum that restarts whenever key changes value."""
    x = np.nan_to_num(np.asarray(x, dtype="f8"))
    key = np.asarray(key)
    out = np.empty_like(x)
    acc = 0.0
    for i in range(len(x)):
        if i and key[i] != key[i - 1]:
            acc = 0.0
        acc += x[i]
        out[i] = acc
    return out


def _since_change(t, key):
    """t minus the value of t at the most recent change of key (e.g. time since step start)."""
    t = np.asarray(t, dtype="f8")
    key = np.asarray(key)
    out = np.empty_like(t)
    start = t[0] if len(t) else 0.0
    for i in range(len(t)):
        if i and key[i] != key[i - 1]:
            start = t[i]
        out[i] = t[i] - start
    return out


def _shift(x, k=1):
    k = int(k)
    x = np.asarray(x, dtype="f8")
    if k == 0:
        return x.copy()
    out = np.full_like(x, np.nan)
    if k > 0:
        out[k:] = x[:-k]
    else:
        out[:k] = x[-k:]
    return out


FUNCS = {
    "abs": np.abs, "sqrt": np.sqrt, "log": np.log, "log10": np.log10, "exp": np.exp,
    "sin": np.sin, "cos": np.cos, "tan": np.tan, "arctan": np.arctan, "arctan2": np.arctan2,
    "degrees": np.degrees, "radians": np.radians, "sign": np.sign, "square": np.square,
    "cumsum": lambda x: np.cumsum(np.nan_to_num(np.asarray(x, dtype="f8"))),
    "diff0": _diff0, "cumsum_by": _cumsum_by, "since_change": _since_change, "shift": _shift,
    "where_pos": lambda x, a, b: np.where(np.asarray(x) > 0, a, b),
    "maximum": np.maximum, "minimum": np.minimum,
}
CONSTS = {"pi": np.pi}

DOC = (
    "Expression language: column aliases (c0, c1, ...), numbers, + - * / **, unary minus, constant pi, "
    "and functions abs sqrt log log10 exp sin cos tan arctan arctan2 degrees radians sign square maximum minimum, "
    "cumsum(x), diff0(x) (first difference, leading 0), cumsum_by(x, key) (cumulative sum restarting when key "
    "changes), since_change(t, key) (t minus t at the last change of key), shift(x, k), where_pos(x, a, b). "
    "Code fits TARGET = a*expr + b, so constant factors and offsets need not be exact."
)


class DSLError(ValueError):
    pass


def evaluate(expr: str, cols: dict[str, np.ndarray]) -> np.ndarray:
    if not isinstance(expr, str) or not expr.strip():
        raise DSLError("empty expression")
    if len(expr) > MAX_LEN:
        raise DSLError(f"expression longer than {MAX_LEN} characters")
    try:
        tree = ast.parse(expr.strip(), mode="eval")
    except SyntaxError as exc:
        raise DSLError(f"syntax error: {exc.msg}") from None

    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return float(node.value)
        if isinstance(node, ast.Name):
            if node.id in cols:
                return np.asarray(cols[node.id], dtype="f8")
            if node.id in CONSTS:
                return CONSTS[node.id]
            raise DSLError(f"unknown name {node.id!r}; use column aliases {sorted(cols)[:8]}... or pi")
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            v = ev(node.operand)
            return -v if isinstance(node.op, ast.USub) else v
        if isinstance(node, ast.BinOp):
            a, b = ev(node.left), ev(node.right)
            ops = {ast.Add: np.add, ast.Sub: np.subtract, ast.Mult: np.multiply, ast.Div: np.divide,
                   ast.Pow: np.power}
            for t, f in ops.items():
                if isinstance(node.op, t):
                    return f(a, b)
            raise DSLError(f"operator {type(node.op).__name__} not allowed")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in FUNCS:
                name = getattr(node.func, "id", "?")
                raise DSLError(f"function {name!r} not allowed; allowed: {sorted(FUNCS)}")
            if node.keywords:
                raise DSLError("keyword arguments are not allowed")
            return FUNCS[node.func.id](*[ev(a) for a in node.args])
        raise DSLError(f"{type(node).__name__} not allowed")

    with np.errstate(all="ignore"):
        try:
            out = ev(tree)
        except DSLError:
            raise
        except Exception as exc:
            raise DSLError(f"{type(exc).__name__}: {exc}") from None
    n = len(next(iter(cols.values())))
    out = np.broadcast_to(np.asarray(out, dtype="f8"), (n,)).copy()
    return out


def fit(target: np.ndarray, pred: np.ndarray) -> dict:
    """Least-squares target ~ a*pred + b over rows where both are finite."""
    y = np.asarray(target, dtype="f8")
    x = np.asarray(pred, dtype="f8")
    m = np.isfinite(y) & np.isfinite(x)
    n = int(m.sum())
    if n < 3:
        return {"rows_used": n, "error": "fewer than 3 finite rows"}
    y, x = y[m], x[m]
    if np.ptp(y) == 0:
        return {"rows_used": n, "error": "TARGET is constant, so no hypothesis can be tested against it"}
    if np.ptp(x) == 0:
        return {"rows_used": n, "r2": 0.0, "error": "the expression is constant over these rows, so it explains nothing"}
    rng = float(np.ptp(y))
    a, b = np.polyfit(x, y, 1)
    res = y - (a * x + b)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float((res ** 2).sum()) / ss_tot
    return {
        "rows_used": n,
        "rows_total": int(len(target)),
        "a": float(f"{a:.6g}"),
        "b": float(f"{b:.6g}"),
        "r2": round(r2, 5),
        "max_abs_residual_over_range": round(float(np.abs(res).max()) / rng, 5),
        "fraction_rows_within_1pct_of_range": round(float((np.abs(res) <= 0.01 * rng).mean()), 4),
    }
