"""Trajectory designs: many solution segments integrated at once.

Rows sampled along solutions arrive in time order, so neighbouring rows are
dependent and finite differences are meaningful only inside a contiguous
stretch.  :func:`segments` integrates a batch of initial conditions with a
fixed-step fourth-order Runge--Kutta scheme (four sub-steps per sample, error
far below any noise level used) and cuts each solution into the maximal runs
that stay inside the target box, so that no difference is ever taken across a
gap where a solution left the domain.
"""

from __future__ import annotations

import numpy as np

__all__ = ["integrate", "segments"]


def integrate(rhs, X0: np.ndarray, T: float, dt: float, substeps: int = 4) -> np.ndarray:
    """Solutions of ``x' = rhs(x)`` from the rows of ``X0`` sampled every ``dt``
    on ``[0, T)``; returns ``(n_steps, n_traj, n)``."""
    n_steps = int(round(T / dt))
    h = dt / substeps
    X = np.array(X0, dtype=float)
    out = np.empty((n_steps,) + X.shape)
    for t in range(n_steps):
        out[t] = X
        for _ in range(substeps):
            k1 = rhs(X)
            k2 = rhs(X + 0.5 * h * k1)
            k3 = rhs(X + 0.5 * h * k2)
            k4 = rhs(X + h * k3)
            X = X + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return out


def segments(sys_, rng, n_traj: int, T: float, dt: float, scale,
             min_len: int = 5, x0: np.ndarray | None = None):
    """Contiguous in-domain runs of ``n_traj`` solutions of ``sys_``.

    Returns ``(runs, ids)``: a list of ``(len, n)`` arrays and the index of the
    solution each run came from (runs of one solution are not independent and
    should share a cluster label).
    """
    n = sys_.n
    if x0 is None:
        x0 = rng.uniform(-1.0, 1.0, size=(n_traj, n)) * np.asarray(scale)
    sol = integrate(sys_.rhs, x0, T, dt)
    inside = np.all(np.abs(sol) <= np.asarray(scale), axis=2)        # (steps, traj)
    runs, ids = [], []
    for j in range(n_traj):
        ok = inside[:, j]
        edges = np.flatnonzero(np.diff(np.concatenate([[0], ok.astype(int), [0]])))
        for a, b in zip(edges[::2], edges[1::2]):
            if b - a >= min_len:
                runs.append(sol[a:b, j])
                ids.append(j)
    return runs, ids
