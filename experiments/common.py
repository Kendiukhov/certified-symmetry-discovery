"""Shared harness for the experiments: protocols, metrics and parallel driving.

Every experiment reports the same two quantities, defined once here so that all
methods are scored identically:

**false certification** -- the declared subspace contains a generator whose
*true* defect exceeds the tolerance ``delta``.  This is the error the
certificates are designed to control, and it is evaluated against exact ground
truth (the true system's coefficients are known in closed form).

**detection** -- the declared dimension is at least the dimension of the true
symmetry algebra, so that the procedure has found all the symmetry that is
there without over-claiming.
"""

from __future__ import annotations

import json
import os
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import scipy.linalg as sla

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

ALPHA = 0.05
DELTA = 0.05


# ---------------------------------------------------------------- metrics
def sup_true_defect(prob, V: np.ndarray, beta_true: np.ndarray) -> float:
    """``sup_{theta in span(V), theta != 0} rho(theta; beta_true)``.

    Exactly the largest generalised eigenvalue of the true numerator and
    denominator forms restricted to ``V``, so no sampling of directions is
    involved.
    """
    V = np.asarray(V, dtype=float)
    if V.ndim == 1:
        V = V[:, None]
    if V.shape[1] == 0:
        return 0.0
    C = V.T @ prob.C(beta_true) @ V
    D = V.T @ prob.D(beta_true) @ V
    D = (D + D.T) / 2 + 1e-14 * np.trace(D) / D.shape[0] * np.eye(D.shape[0])
    w = sla.eigh((C + C.T) / 2, D, eigvals_only=True)
    return float(np.sqrt(max(w.max(), 0.0)))


def principal_angle(V: np.ndarray, W: np.ndarray) -> float:
    """Largest principal angle (degrees) between two subspaces."""
    if V.size == 0 or W.size == 0:
        return 90.0
    Qv, _ = np.linalg.qr(V)
    Qw, _ = np.linalg.qr(W)
    s = np.linalg.svd(Qv.T @ Qw, compute_uv=False)
    return float(np.degrees(np.arccos(np.clip(s.min(), -1.0, 1.0))))


def rms_field(sys_, rng, scale, sampler="box", n_mc=200_000):
    """Root-mean-square size of the vector field over the target domain, used to
    express the noise level as a fraction of the signal."""
    n = sys_.n
    if sampler == "box":
        X = rng.uniform(-scale, scale, size=(n_mc, n))
    else:
        X = rng.normal(0, scale, size=(n_mc, n))
    return float(np.sqrt(np.mean(np.sum(sys_.rhs(X) ** 2, axis=1))))


# ------------------------------------------------------------- parallelism
def run_cells(fn, cells, workers=None, chunk=1, desc=""):
    """Map ``fn`` over ``cells`` in worker processes, returning a list."""
    workers = workers or max(1, (os.cpu_count() or 2) - 2)
    t0 = time.time()
    out = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, r in enumerate(ex.map(fn, cells, chunksize=chunk)):
            out.append(r)
            if desc and (i + 1) % max(1, len(cells) // 20) == 0:
                el = time.time() - t0
                print(f"  [{desc}] {i+1}/{len(cells)} cells, {el:.0f}s elapsed",
                      flush=True)
    return out


def save(name: str, payload: dict, cpu_seconds: float | None = None):
    payload = dict(payload)
    payload["_meta"] = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "cpu_seconds": cpu_seconds,
        "alpha": ALPHA,
        "delta": DELTA,
    }
    path = RESULTS / f"{name}.json"
    path.write_text(json.dumps(payload, indent=1, default=_default))
    print(f"wrote {path}")
    return path


def _default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def load(name: str) -> dict:
    return json.loads((RESULTS / f"{name}.json").read_text())
