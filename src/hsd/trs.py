"""Exact extrema of ``||a + B u||`` over the ball ``||u|| <= R``.

These are trust-region subproblems.  They are solved exactly (up to the
tolerance of a one-dimensional root find) via an eigendecomposition and the
classical secular equation, including the "hard case".  Using the exact extrema
rather than the triangle inequality ``||a|| +/- R ||B||`` makes the certificates
of ``certify.py`` tighter and therefore more powerful, at no cost in validity.
"""

from __future__ import annotations

import numpy as np

__all__ = ["max_norm_over_ball", "min_norm_over_ball"]

_EPS = 1e-12


def _prepare(a: np.ndarray, B: np.ndarray):
    Q = B.T @ B
    Q = (Q + Q.T) / 2.0
    c = B.T @ a
    lam, V = np.linalg.eigh(Q)
    lam = np.clip(lam, 0.0, None)
    ct = V.T @ c
    return lam, V, ct, float(a @ a)


def _norm_from(lam, ct, mu, sign):
    """``||u||^2`` for ``u = (sign*mu I + ...)`` solution of the secular equation."""
    d = sign * (mu - lam)
    return np.sum((ct / d) ** 2)


def max_norm_over_ball(a: np.ndarray, B: np.ndarray, R: float) -> float:
    """``max_{||u|| <= R} ||a + B u||`` (exact)."""
    if R <= 0:
        return float(np.linalg.norm(a))
    lam, V, ct, aa = _prepare(a, B)
    lmax = lam.max() if lam.size else 0.0
    # Interior stationary points cannot be maxima (the objective is convex), so
    # the maximiser lies on the sphere: (Q - mu I) u = -c with mu > lmax.
    top = np.abs(ct) > _EPS * max(1.0, np.abs(ct).max(initial=0.0))
    hard = not np.any(top & (np.abs(lam - lmax) <= 1e-12 * max(1.0, lmax)))
    if hard:
        # No component of c along the top eigenspace: try mu = lmax.
        mask = np.abs(lam - lmax) > 1e-12 * max(1.0, lmax)
        if mask.any():
            nrm2 = np.sum((ct[mask] / (lmax - lam[mask])) ** 2)
        else:
            nrm2 = 0.0
        if nrm2 <= R * R:
            u = np.zeros_like(lam)
            u[mask] = ct[mask] / (lmax - lam[mask])
            extra = np.sqrt(max(R * R - nrm2, 0.0))
            j = int(np.argmax(lam))
            u[j] += extra
            return float(np.linalg.norm(a + B @ (V @ u)))
    # Standard case: solve ||u(mu)|| = R for mu > lmax by bisection.
    lo = lmax + 1e-14 * max(1.0, lmax)
    scale = max(np.linalg.norm(ct) / max(R, _EPS), 1.0)
    hi = lmax + scale + 1.0
    for _ in range(200):
        if _norm_from(lam, ct, hi, +1) <= R * R:
            break
        hi *= 2.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if _norm_from(lam, ct, mid, +1) > R * R:
            lo = mid
        else:
            hi = mid
    mu = 0.5 * (lo + hi)
    u = V @ (ct / (mu - lam))
    nu = np.linalg.norm(u)
    if nu > 0:
        u *= min(1.0, R / nu)
    return float(np.linalg.norm(a + B @ u))


def min_norm_over_ball(a: np.ndarray, B: np.ndarray, R: float) -> float:
    """``min_{||u|| <= R} ||a + B u||`` (exact)."""
    if R <= 0:
        return float(np.linalg.norm(a))
    lam, V, ct, aa = _prepare(a, B)
    # Unconstrained minimiser of the convex quadratic, if it is inside the ball.
    pos = lam > _EPS * max(1.0, lam.max(initial=0.0))
    u = np.zeros_like(lam)
    u[pos] = -ct[pos] / lam[pos]
    if np.all(pos | (np.abs(ct) <= _EPS)) and np.linalg.norm(u) <= R:
        return float(np.linalg.norm(a + B @ (V @ u)))
    # Otherwise the minimiser is on the sphere: (Q + nu I) u = -c, nu > -lmin.
    lmin = lam.min()
    lo = max(0.0, -lmin) + 1e-14
    hi = lo + max(np.linalg.norm(ct) / max(R, _EPS), 1.0) + 1.0
    for _ in range(200):
        if np.sum((ct / (lam + hi)) ** 2) <= R * R:
            break
        hi *= 2.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if np.sum((ct / (lam + mid)) ** 2) > R * R:
            lo = mid
        else:
            hi = mid
    nu = 0.5 * (lo + hi)
    u = -ct / (lam + nu)
    n_u = np.linalg.norm(u)
    if n_u > R:
        u *= R / n_u
    return float(np.linalg.norm(a + B @ (V @ u)))
