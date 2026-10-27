"""Shared harness for the experiments: protocols, metrics and parallel driving.

Every experiment reports the same two quantities, defined once here so that all
methods are scored identically:

**false certification** -- the declared subspace contains a generator whose
*true* defect exceeds the tolerance ``delta``.  This is the error the
certificates are designed to control, and it is evaluated against exact ground
truth (the true system's coefficients are known in closed form).

**detection** -- the declared dimension is at least the dimension of the true
symmetry algebra.

**recovery** -- detection, and in addition the declared subspace contains the
true algebra up to a largest principal angle of 10 degrees, measured in the
``L2(nu)`` geometry of the generator fields (so it does not depend on how the
candidate basis is scaled).  A rule that declares a large or badly oriented
subspace can detect without recovering.

Monte Carlo uncertainty of a rate estimated from ``k`` events in ``T``
replications is reported as an exact Clopper--Pearson interval
(:func:`clopper_pearson`).
"""

from __future__ import annotations

import json
import os
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor

# One BLAS thread per worker process: the experiments parallelise over cells,
# and nested BLAS threading on small matrices only oversubscribes the machine.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")
from pathlib import Path

import numpy as np
import scipy.linalg as sla

ROOT = Path(__file__).resolve().parents[1]
# HSD_RESULTS redirects output (used by the smoke test); HSD_SMOKE caps the
# number of replications so that every experiment can be exercised quickly.
RESULTS = Path(os.environ.get("HSD_RESULTS", ROOT / "results"))
RESULTS.mkdir(exist_ok=True)
SMOKE = int(os.environ.get("HSD_SMOKE", "0"))


def trials(n: int) -> int:
    """Number of replications: ``n``, or the smoke-test cap if one is set."""
    return min(n, SMOKE) if SMOKE else n

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


def _generator_sqrt_gram(prob) -> np.ndarray:
    if getattr(prob, "_Gxi", None) is None:
        prob._build_generator_grams()
    w, U = np.linalg.eigh((prob._Gxi + prob._Gxi.T) / 2)
    return (U * np.sqrt(np.clip(w, 0.0, None))) @ U.T


def containment_angle(prob, S: np.ndarray, V: np.ndarray) -> float:
    """Largest principal angle (degrees) between the true algebra ``S`` and the
    declared subspace ``V``, in the ``L2(nu)`` geometry of generator fields:
    0 when ``S`` lies inside ``V``, 90 when ``V`` is too small to contain it."""
    S = np.atleast_2d(S.T).T if S.ndim == 1 else S
    V = np.atleast_2d(V.T).T if V.ndim == 1 else V
    if S.shape[1] == 0:
        return 0.0
    if V.shape[1] < S.shape[1]:
        return 90.0
    Gh = _generator_sqrt_gram(prob)
    Qs, _ = np.linalg.qr(Gh @ S)
    Qv, _ = np.linalg.qr(Gh @ V)
    s = np.linalg.svd(Qs.T @ Qv, compute_uv=False)
    return float(np.degrees(np.arccos(np.clip(s.min(), -1.0, 1.0))))


def recovered(prob, S: np.ndarray, V: np.ndarray, tol_deg: float = 10.0) -> bool:
    """Detection plus containment of the true algebra within ``tol_deg``."""
    if S.shape[1] == 0:
        return True
    return V.shape[1] >= S.shape[1] and containment_angle(prob, S, V) < tol_deg


def clopper_pearson(k: int, T: int, level: float = 0.95) -> tuple[float, float]:
    """Exact two-sided binomial confidence interval for ``k`` events in ``T``."""
    from scipy import stats
    a = 1.0 - level
    lo = 0.0 if k == 0 else float(stats.beta.ppf(a / 2, k, T - k + 1))
    hi = 1.0 if k == T else float(stats.beta.ppf(1 - a / 2, k + 1, T - k))
    return lo, hi


def one_sided_upper(k: int, T: int, level: float = 0.95) -> float:
    """Exact one-sided upper confidence bound for a binomial rate."""
    from scipy import stats
    return 1.0 if k == T else float(stats.beta.ppf(level, k + 1, T - k))


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
