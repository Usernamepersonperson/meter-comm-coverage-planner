"""Gateway site selection: greedy max-coverage, swap local search, and an exact ILP check.

Objective for a site set S, with c_b = number of selected sites that can hear building b and
w_b = number of meters in b:

    f(S) = sum_b w_b * ( 1[c_b >= 1] + lam * 1[c_b >= 2] )

lam < 1 makes the first cover count more than the second, so the greedy order first reaches
coverage and then buys redundancy. f is monotone submodular, so greedy has the usual (1 - 1/e) bound.
"""

from __future__ import annotations

import time

import numpy as np
from scipy import sparse
from scipy.optimize import Bounds, LinearConstraint, milp

LAM = 0.5


def objective(A: np.ndarray, w: np.ndarray, sel, lam: float = LAM) -> float:
    c = A[list(sel)].sum(axis=0) if len(sel) else np.zeros(A.shape[1])
    return float((w * ((c >= 1) + lam * (c >= 2))).sum())


def _gain(A32: np.ndarray, w: np.ndarray, counts: np.ndarray, lam: float) -> np.ndarray:
    val = w * ((counts == 0) + lam * (counts == 1))
    return A32 @ val.astype(np.float32)


def greedy(A: np.ndarray, w: np.ndarray, k_max: int, lam: float = LAM) -> list[int]:
    """Greedy order of up to k_max sites (stops early when nothing adds value)."""
    A32 = A.astype(np.float32)
    counts = np.zeros(A.shape[1], dtype=np.int32)
    chosen: list[int] = []
    for _ in range(min(k_max, A.shape[0])):
        g = _gain(A32, w, counts, lam)
        g[chosen] = -1.0
        j = int(np.argmax(g))
        if g[j] <= 1e-9:
            break
        chosen.append(j)
        counts += A[j]
    return chosen


def local_search(A: np.ndarray, w: np.ndarray, start: list[int], lam: float = LAM, max_rounds: int = 20) -> list[int]:
    """Best-improvement 1-swap search: replace one chosen site by any unused candidate."""
    A32 = A.astype(np.float32)
    sel = list(start)
    if not sel:
        return sel
    best_f = objective(A, w, sel, lam)
    for _ in range(max_rounds):
        improved = False
        for pos in range(len(sel)):
            rest = sel[:pos] + sel[pos + 1:]
            counts = A[rest].sum(axis=0).astype(np.int32) if rest else np.zeros(A.shape[1], dtype=np.int32)
            g = _gain(A32, w, counts, lam)
            g[rest] = -1.0
            j = int(np.argmax(g))
            f_new = objective(A, w, rest + [j], lam)
            if f_new > best_f + 1e-9:
                sel[pos] = j
                best_f = f_new
                improved = True
        if not improved:
            break
    return sel


def ilp(A: np.ndarray, w: np.ndarray, k: int, lam: float = LAM, time_limit_s: float = 60.0):
    """Exact (or time-limited) optimum of f for exactly k sites via HiGHS.

    Identical coverage patterns are merged first (their weights add). Returns (sites, info).
    """
    t0 = time.perf_counter()
    cols = np.where(A.any(axis=1))[0]
    if len(cols) == 0 or k <= 0:
        return [], {"status": "empty", "gap": None, "seconds": 0.0, "rows": 0, "upper_bound": None}
    sub = A[cols]  # (C', B)
    keep = sub.any(axis=0)
    packed = np.packbits(sub[:, keep], axis=0)
    _, inv = np.unique(packed.T, axis=0, return_inverse=True)
    inv = np.asarray(inv).ravel()
    n_u = int(inv.max()) + 1
    wu = np.bincount(inv, weights=w[keep], minlength=n_u)
    first = np.zeros(n_u, dtype=int)
    first[inv[::-1]] = np.arange(len(inv))[::-1]
    M = sub[:, keep][:, first].T.astype(np.float64)  # (n_u, C')
    C = len(cols)
    # variables: x (C) | y1 (n_u) | y2 (n_u)
    nv = C + 2 * n_u
    cost = np.concatenate([np.zeros(C), -wu, -lam * wu])
    Ms = sparse.csr_matrix(M)
    eye = sparse.identity(n_u, format="csr")
    # y1 <= sum a x ;  2 y2 <= sum a x
    r1 = sparse.hstack([-Ms, eye, sparse.csr_matrix((n_u, n_u))])
    r2 = sparse.hstack([-Ms, sparse.csr_matrix((n_u, n_u)), 2 * eye])
    card = sparse.csr_matrix(np.concatenate([np.ones(C), np.zeros(2 * n_u)])[None, :])
    cons = [LinearConstraint(sparse.vstack([r1, r2]), -np.inf, 0.0), LinearConstraint(card, k, k)]
    res = milp(cost, constraints=cons, integrality=np.ones(nv), bounds=Bounds(0, 1),
               options={"time_limit": float(time_limit_s), "mip_rel_gap": 1e-4})
    info = {"status": str(res.message), "seconds": time.perf_counter() - t0, "rows": n_u,
            "gap": getattr(res, "mip_gap", None),
            "upper_bound": -res.mip_dual_bound if getattr(res, "mip_dual_bound", None) is not None else None}
    if res.x is None:
        return [], info
    sites = [int(cols[i]) for i in np.where(res.x[:C] > 0.5)[0]]
    return sites, info


def coverage_stats(A: np.ndarray, w: np.ndarray, sel) -> dict:
    """Meter-weighted share covered once / at least twice."""
    tot = float(w.sum())
    if len(sel) == 0:
        return {"cover1": 0.0, "cover2": 0.0}
    c = A[list(sel)].sum(axis=0)
    return {"cover1": float(w[c >= 1].sum() / tot), "cover2": float(w[c >= 2].sum() / tot)}


def curve(A: np.ndarray, w: np.ndarray, order: list[int]) -> list[dict]:
    """Coverage after the first k sites of a greedy order, k = 1..len(order)."""
    return [{"k": k, **coverage_stats(A, w, order[:k])} for k in range(1, len(order) + 1)]


def min_sites(curve_rows: list[dict], target: float, key: str = "cover1") -> int | None:
    for r in curve_rows:
        if r[key] >= target:
            return r["k"]
    return None
