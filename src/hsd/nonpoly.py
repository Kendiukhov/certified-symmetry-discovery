"""Non-polynomial reference systems and Monte-Carlo ground-truth defects.

Used only to *evaluate* procedures when the model class cannot contain the
truth.  The defect of a candidate generator for a smooth non-polynomial field is
computed by high-accuracy Monte-Carlo integration of the analytic bracket, with
the integration error reported so that it is never mistaken for a real effect.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

__all__ = ["SmoothSystem", "SMOOTH_SYSTEMS", "mc_defect", "sup_mc_defect"]


@dataclass
class SmoothSystem:
    name: str
    n: int
    rhs: Callable[[np.ndarray], np.ndarray]
    jac: Callable[[np.ndarray], np.ndarray]
    description: str


def _pendulum_rhs(X):
    return np.stack([X[:, 1], -np.sin(X[:, 0]) - 0.1 * X[:, 1]], axis=1)


def _pendulum_jac(X):
    J = np.zeros((X.shape[0], 2, 2))
    J[:, 0, 1] = 1.0
    J[:, 1, 0] = -np.cos(X[:, 0])
    J[:, 1, 1] = -0.1
    return J


def _rot_rhs(X):
    """dr/dt = r(1-r), dphi/dt = 1/(1+r^2): exactly SO(2)-symmetric, and not a
    polynomial vector field in Cartesian coordinates."""
    r2 = np.sum(X ** 2, axis=1)
    r = np.sqrt(np.maximum(r2, 1e-300))
    a = (1.0 - r)                      # radial rate divided by r
    w = 1.0 / (1.0 + r2)
    return np.stack([a * X[:, 0] - w * X[:, 1], a * X[:, 1] + w * X[:, 0]], axis=1)


def _rot_jac(X, h=1e-6):
    J = np.zeros((X.shape[0], 2, 2))
    for j in range(2):
        e = np.zeros(2)
        e[j] = h
        J[:, :, j] = (_rot_rhs(X + e) - _rot_rhs(X - e)) / (2 * h)
    return J


SMOOTH_SYSTEMS = {
    "pendulum": SmoothSystem("pendulum", 2, _pendulum_rhs, _pendulum_jac,
                             "Damped pendulum; no continuous affine symmetry."),
    "rot_nonpoly": SmoothSystem("rot_nonpoly", 2, _rot_rhs, _rot_jac,
                                "Non-polynomial planar flow with exact SO(2) symmetry."),
}


def mc_defect(smooth: SmoothSystem, prob, theta: np.ndarray, rng, n_mc: int = 2_000_000):
    """Monte-Carlo estimate of ``rho(theta)`` for a smooth vector field.

    Returns ``(rho, standard_error)``.
    """
    n = smooth.n
    X = rng.uniform(-prob.scale, prob.scale, size=(n_mc, n))
    xi = np.einsum("p,pij->ij", theta, prob.gen_basis)
    Xi = prob.alg.eval(xi, X)                          # (N, n)
    Dxi = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            e = [0] * n
            e[j] = 1
            Dxi[i, j] = xi[i, prob.alg.index[tuple(e)]]
    F = smooth.rhs(X)
    J = smooth.jac(X)
    t1 = np.einsum("kij,kj->ki", J, Xi)                # DF xi
    t2 = F @ Dxi.T                                     # D xi F (xi is affine)
    num = np.sum((t1 - t2) ** 2, axis=1)
    den = np.sum(t1 ** 2, axis=1) + np.sum(t2 ** 2, axis=1)
    mn, md = num.mean(), den.mean()
    rho = float(np.sqrt(mn / md))
    se_n = num.std(ddof=1) / np.sqrt(n_mc)
    se_d = den.std(ddof=1) / np.sqrt(n_mc)
    se = 0.5 * rho * np.sqrt((se_n / mn) ** 2 + (se_d / md) ** 2)
    return rho, float(se)


def sup_mc_defect(smooth: SmoothSystem, prob, V: np.ndarray, rng, n_dirs: int = 400,
                  n_mc: int = 200_000):
    """Largest Monte-Carlo defect over the unit sphere of a subspace."""
    V = np.atleast_2d(V.T).T
    if V.size == 0:
        return 0.0
    best = 0.0
    d = V.shape[1]
    for _ in range(n_dirs if d > 1 else 1):
        c = rng.normal(size=d) if d > 1 else np.ones(1)
        best = max(best, mc_defect(smooth, prob, V @ (c / np.linalg.norm(c)), rng, n_mc)[0])
    return best
