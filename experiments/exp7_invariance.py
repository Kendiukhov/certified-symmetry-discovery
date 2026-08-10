"""Experiment 7 -- invariances of a learned model, not of a dynamical system.

Otto et al. (2025) apply the same nullspace construction to trained models.  For
a scalar basis-function regression the Lie derivative is ``grad f . xi``, so the
defect is the root-mean-square cosine of the angle between the model's gradient
and the generator.  We fit models to noisy samples of

* a rotation-invariant target,
* a target that is invariant only up to a controlled amount,
* a target with no continuous invariance,

and ask each method which invariances of the *target* may be declared.  The
restricted-support variant trains on an angular sector, the situation in which a
model can appear invariant simply because it was never asked about the rest of
the plane.
"""

from __future__ import annotations

import time
from itertools import product

import numpy as np

from common import ALPHA, DELTA, run_cells, save
from hsd.certify import certified_dimension, certify_direction
from hsd.estimation import fit_ols_scalar
from hsd.invariance import InvarianceProblem

DEG = 6
TRIALS = 300
NS = [1000, 4000, 16000, 64000]
SIGMAS = [0.01, 0.05, 0.2]


def target_coefs(alg, kind: str, eps: float = 0.0):
    """Coefficients of the target function in the monomial basis."""
    f = np.zeros(alg.m)
    if kind in ("invariant", "near_invariant"):
        # f = (x^2+y^2) - 0.5 (x^2+y^2)^2  (+ eps * x^2 y^2 style breaking term)
        f[alg.index[(2, 0)]] += 1.0
        f[alg.index[(0, 2)]] += 1.0
        for a, c in [((4, 0), 1.0), ((2, 2), 2.0), ((0, 4), 1.0)]:
            f[alg.index[a]] += -0.5 * c
        if kind == "near_invariant":
            f[alg.index[(3, 0)]] += eps
    elif kind == "anisotropic":
        f[alg.index[(2, 0)]] += 1.0
        f[alg.index[(0, 2)]] += 0.6
        f[alg.index[(4, 0)]] += -0.3
    else:
        raise ValueError(kind)
    return f


def one_cell(args):
    kind, eps, sector, N, srel, seed = args
    rng = np.random.default_rng(seed)
    prob = InvarianceProblem.build(2, DEG, "linear", "box", scale=1.0)
    alg = prob.alg
    f = target_coefs(alg, kind, eps)
    beta0 = prob.beta_of_function(f)
    rot = np.array([0.0, 1.0, -1.0, 0.0])
    rot /= np.linalg.norm(rot)
    true_rot = prob.rho(rot, beta0)
    true_min = float(prob.spectrum(beta0)[0][0])

    Xr = rng.uniform(-1, 1, size=(200_000, 2))
    rms = float(np.sqrt(np.mean(alg.eval(f[None, :], Xr) ** 2)))
    acc = dict(fc_naive_t=0.0, fc_naive_d=0.0, fc_sym=0.0,
               dim_naive_t=0.0, dim_naive_d=0.0, dim_sym=0.0, u_rot=0.0)
    for _ in range(TRIALS):
        X = rng.uniform(-1, 1, size=(N, 2))
        if sector:
            ang = np.arctan2(X[:, 1], X[:, 0])
            keep = np.abs(ang) < np.pi / 3
            while keep.sum() < N:
                Z = rng.uniform(-1, 1, size=(N, 2))
                a2 = np.arctan2(Z[:, 1], Z[:, 0])
                X = np.vstack([X[keep], Z[np.abs(a2) < np.pi / 3]])
                keep = np.ones(len(X), bool)
            X = X[:N]
        y = alg.eval(f[None, :], X)[:, 0] + srel * rms * rng.normal(size=N)
        fit = fit_ols_scalar(prob, X, y)
        rho_t, Th_t = prob.spectrum(fit.beta)
        d_t = int(np.sum(rho_t <= DELTA))
        acc["dim_naive_t"] += d_t
        acc["fc_naive_t"] += _sup_true(prob, Th_t[:, :d_t], beta0) > DELTA
        pe = prob.empirical(X)
        rho_d, Th_d = pe.spectrum(fit.beta)
        d_d = int(np.sum(rho_d <= DELTA))
        acc["dim_naive_d"] += d_d
        acc["fc_naive_d"] += _sup_true(prob, Th_d[:, :d_d], beta0) > DELTA
        dc, Vc = certified_dimension(prob, fit, DELTA, ALPHA)
        acc["dim_sym"] += dc
        acc["fc_sym"] += _sup_true(prob, Vc, beta0) > DELTA
        acc["u_rot"] += certify_direction(prob, rot, fit, ALPHA).upper
    out = {k: v / TRIALS for k, v in acc.items()}
    out.update(kind=kind, eps=eps, sector=sector, N=N, sigma_rel=srel,
               true_rot_defect=float(true_rot), true_min_defect=true_min)
    return out


def _sup_true(prob, V, beta0):
    import scipy.linalg as sla
    if V.size == 0:
        return 0.0
    C = V.T @ prob.C(beta0) @ V
    D = V.T @ prob.D(beta0) @ V
    D = (D + D.T) / 2 + 1e-14 * np.trace(D) / D.shape[0] * np.eye(D.shape[0])
    return float(np.sqrt(max(sla.eigh((C + C.T) / 2, D, eigvals_only=True).max(), 0.0)))


def main():
    specs = ([("invariant", 0.0, s) for s in (False, True)]
             + [("near_invariant", e, False) for e in (0.02, 0.05, 0.2)]
             + [("anisotropic", 0.0, s) for s in (False, True)])
    cells = [(k, e, s, N, sg, 7000 + 1637 * i)
             for i, ((k, e, s), N, sg) in enumerate(product(specs, NS, SIGMAS))]
    w0 = time.time()
    rows = run_cells(one_cell, cells, desc="exp7")
    save("exp7_invariance", {"rows": rows, "trials": TRIALS, "degree": DEG})
    print(f"exp7 wall {time.time()-w0:.0f}s")


if __name__ == "__main__":
    main()
