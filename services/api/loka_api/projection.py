"""Controlled projection — a generic KB.METHODS predictor (the professor's METHODS / EcoFormer slot).

Fits ``outcome ~ f(dial) + controls`` on a panel via pure-Python OLS, then projects the outcome
when the dial moves to a new value — holding the target's own control values fixed and ANCHORED to
the target's actual current outcome. Anchoring means: at the current dial the projection equals the
target's real value, and moving the dial moves along the fitted slope. That avoids a global model
mis-pricing a specific country (e.g. predicting Thailand's child mortality from the world average).

Returns a point estimate + 95% interval, labelled ``observational`` — an association-based
projection, not an identified causal effect. Generic: pass any panel and which columns are
outcome / dial / controls; no per-scenario code. ``log_cols`` names columns entered in log scale.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from typing import Any


def _solve(A: list[list[float]], b: list[float]) -> list[float]:
    """Solve A x = b (Gauss-Jordan, partial pivoting). A is p×p, b length p."""
    n = len(A)
    m = [list(A[i]) + [b[i]] for i in range(n)]
    for c in range(n):
        piv = max(range(c, n), key=lambda r: abs(m[r][c]))
        if abs(m[piv][c]) < 1e-12:
            raise ValueError("singular matrix (collinear predictors?)")
        m[c], m[piv] = m[piv], m[c]
        pv = m[c][c]
        m[c] = [v / pv for v in m[c]]
        for r in range(n):
            if r != c and m[r][c] != 0.0:
                f = m[r][c]
                m[r] = [m[r][k] - f * m[c][k] for k in range(n + 1)]
    return [m[i][n] for i in range(n)]


def _num(v: Any) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def controlled_projection(
    panel: Sequence[Mapping[str, Any]],
    *,
    outcome: str,
    dial: str,
    controls: Sequence[str],
    target: Mapping[str, Any],
    new_dial: float,
    log_cols: Sequence[str] = (),
    clamp_min: float | None = 0.0,
) -> dict[str, Any]:
    """Project ``outcome`` for ``target`` when ``dial`` -> ``new_dial``, controls held fixed."""
    log = set(log_cols)
    cols = [dial, *controls]

    def _t(col: str, v: float) -> float:
        return math.log(v) if col in log else v

    def feat(row: Mapping[str, Any], dial_val: float | None = None) -> list[float] | None:
        x = [1.0]
        for col in cols:
            raw = dial_val if (col == dial and dial_val is not None) else _num(row.get(col))
            if raw is None or (col in log and raw <= 0):
                return None
            x.append(_t(col, float(raw)))
        return x

    # assemble complete-case design matrix
    Xs: list[list[float]] = []
    ys: list[float] = []
    for row in panel:
        y = _num(row.get(outcome))
        x = feat(row)
        if y is not None and x is not None:
            Xs.append(x)
            ys.append(y)
    n, p = len(Xs), len(cols) + 1
    if n <= p:
        raise ValueError(f"not enough rows ({n}) to fit {p} parameters")

    # Fingerprint of the sample the model actually fitted on — the complete-case rows, not the
    # source file. Two runs over the same data produce the same digest; a data revision produces
    # a different one, so an audit record can tell "same answer" from "same answer by coincidence".
    digest = hashlib.sha256()
    digest.update(f"{outcome}|{dial}|{','.join(controls)}|{','.join(sorted(log))}".encode())
    for x, y in zip(Xs, ys, strict=True):
        digest.update(f"{y:.6g}|{'|'.join(f'{v:.6g}' for v in x)}\n".encode())
    sample_digest = digest.hexdigest()[:16]

    # OLS: beta = (X'X)^-1 X'y
    XtX = [[0.0] * p for _ in range(p)]
    Xty = [0.0] * p
    for x, y in zip(Xs, ys):
        for i in range(p):
            Xty[i] += x[i] * y
            for j in range(p):
                XtX[i][j] += x[i] * x[j]
    beta = _solve(XtX, Xty)

    def predict(x: list[float]) -> float:
        return sum(bi * xi for bi, xi in zip(beta, x))

    # fit quality
    rss = sum((y - predict(x)) ** 2 for x, y in zip(Xs, ys))
    my = sum(ys) / n
    tss = sum((y - my) ** 2 for y in ys) or 1.0
    r2 = 1.0 - rss / tss
    s2 = rss / (n - p)

    # anchor to the target's real current value
    x_cur = feat(target)
    y_cur = _num(target.get(outcome))
    if x_cur is None or y_cur is None:
        raise ValueError("target is missing the dial/controls/outcome needed to project")
    anchor = y_cur - predict(x_cur)

    x_new = feat(target, dial_val=float(new_dial))
    if x_new is None:
        raise ValueError("new_dial is invalid (non-positive under log?)")
    point = predict(x_new) + anchor

    # The decision question is the *effect* of moving the dial, not the level. Because the
    # projection is anchored, point == y_cur + beta_dial * delta_t exactly, so the uncertainty
    # that matters is Var(beta_dial) — not the residual spread across the panel.
    cur_dial_val = _num(target.get(dial))
    assert cur_dial_val is not None  # feat(target) succeeded, so the dial is present and valid
    delta_t = _t(dial, float(new_dial)) - _t(dial, cur_dial_val)
    e_dial = [0.0] * p
    e_dial[1] = 1.0  # column 1 is the dial (column 0 is the intercept)
    var_beta_dial = s2 * _solve(XtX, e_dial)[1]
    se_effect = math.sqrt(max(var_beta_dial, 0.0)) * abs(delta_t)
    effect = beta[1] * delta_t
    eff_lo, eff_hi = effect - 1.96 * se_effect, effect + 1.96 * se_effect
    lo, hi = point - 1.96 * se_effect, point + 1.96 * se_effect

    # A prediction interval for the *level* — how far one country can sit from the fitted
    # surface. Reported for context; it is not the interval the decision should be read against.
    z = _solve(XtX, x_new)
    leverage = sum(a * b for a, b in zip(x_new, z))
    se_level = math.sqrt(max(s2 * (1.0 + leverage), 0.0))
    lvl_lo, lvl_hi = point - 1.96 * se_level, point + 1.96 * se_level

    if clamp_min is not None:
        point, lo, hi = max(point, clamp_min), max(lo, clamp_min), max(hi, clamp_min)
        lvl_lo, lvl_hi = max(lvl_lo, clamp_min), max(lvl_hi, clamp_min)

    return {
        "outcome": outcome,
        "dial": dial,
        "current_dial": cur_dial_val,
        "new_dial": float(new_dial),
        "current_outcome": round(y_cur, 3),
        "projected_outcome": round(point, 3),
        "interval_95": [round(lo, 3), round(hi, 3)],
        "effect": round(effect, 3),
        "effect_interval_95": [round(eff_lo, 3), round(eff_hi, 3)],
        "effect_se": round(se_effect, 4),
        "level_prediction_interval_95": [round(lvl_lo, 3), round(lvl_hi, 3)],
        "controls_held_fixed": {c: _num(target.get(c)) for c in controls},
        "fit": {"n": n, "params": p, "r2": round(r2, 3), "sample_digest": sample_digest},
        "identification": "observational",
        "note": (
            "association-based projection, anchored to the target's current value and holding its "
            "controls fixed; NOT an identified causal effect (residual confounding may remain)"
        ),
        "interval_note": (
            "interval_95 is the 95% CI of the projected level implied by the effect's standard "
            "error (Var(beta_dial)) — the interval the decision is read against. "
            "level_prediction_interval_95 is the much wider prediction interval for where a "
            "single country's level may sit relative to the fitted surface; it answers a "
            "different question and must not be read as the uncertainty of the change."
        ),
    }
