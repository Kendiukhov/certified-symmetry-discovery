"""Experiment 16 -- non-polynomial systems with a dictionary that contains them.

Two families (``hsd.dict_systems``):

* the radial flow ``r' = r(1 - r)``, ``phi' = 1/(1 + r^2)`` plus
  ``eps x^2 d/dx`` on the unit disc, fitted with a library of quadratic
  polynomials and four radial functions; at ``eps = 0`` the rotation is an
  exact symmetry, and its defect grows with ``eps``;
* the damped pendulum on ``[-1.5, 1.5]^2`` fitted with the library
  ``{1, theta, omega, theta^2, theta omega, omega^2, sin theta, cos theta}``;
  no affine generator is a symmetry (smallest true defect 0.204).

The library contains the truth in both cases, so the certificate applies with
no model-error term.  Ground truth: true defects from the same exact (or
rounding-level) quadrature as the certificate, and the true algebra as the
numerical nullspace of the bracket map (the smallest non-zero singular value is
reported to show the gap).  For each setting we record the false-certification
rate, detection and recovery for SymCert and for a fixed threshold at the
tolerance, and the median certified bound on the minimal-defect generator.
"""

from __future__ import annotations

import time
from itertools import product

import numpy as np

from common import ALPHA, DELTA, run_cells, save, sup_true_defect
from common import trials
from hsd import fit_ols
from hsd.certify import certified_dimension, certify_direction, refute_all
from hsd.dict_systems import (box_sample, disc_sample, pendulum_beta, pendulum_problem,
                              radial_beta, radial_problem)

TRIALS = trials(300)
NS = [200, 800, 3200]
SIGMAS = [0.03, 0.1, 0.3]
RADIAL_EPS = [0.0, 0.03, 0.06, 0.12]


def _setup(family, eps):
    if family == "radial":
        prob = radial_problem()
        return prob, radial_beta(prob, eps), disc_sample
    prob = pendulum_problem()
    return prob, pendulum_beta(prob), box_sample


def _true_algebra(prob, beta):
    U, s, Vt = np.linalg.svd(prob.A(beta), full_matrices=True)
    s_full = np.concatenate([s, np.zeros(prob.P - s.size)])
    null = s_full <= 1e-10 * max(s_full.max(), 1e-300)
    gap = float(s_full[~null].min()) if (~null).any() else float("nan")
    return Vt[null].T, gap


def one_cell(args):
    family, eps, N, srel, seed = args
    rng = np.random.default_rng(seed)
    prob, beta0, sampler = _setup(family, eps)
    S, gap = _true_algebra(prob, beta0)
    dstar = S.shape[1]
    Xr = sampler(np.random.default_rng(0), 200_000)
    rms = float(np.sqrt(np.mean(np.sum(prob.rhs(beta0, Xr) ** 2, axis=1))))
    sigma = srel * rms
    acc = dict(fc_sym=0, det_sym=0, dim_sym=0, fc_thr=0, det_thr=0, dim_thr=0,
               refute=0)
    uppers = []
    for _ in range(TRIALS):
        X = sampler(rng, N)
        Y = prob.rhs(beta0, X) + sigma * rng.normal(size=X.shape)
        fit = fit_ols(prob, X, Y)
        rho_hat, Theta = prob.spectrum(fit.beta)
        d = int(np.sum(rho_hat <= DELTA))
        acc["dim_thr"] += d
        acc["fc_thr"] += sup_true_defect(prob, Theta[:, :d], beta0) > DELTA
        acc["det_thr"] += d >= dstar
        dc, Vc = certified_dimension(prob, fit, DELTA, ALPHA)
        acc["dim_sym"] += dc
        acc["fc_sym"] += sup_true_defect(prob, Vc, beta0) > DELTA
        acc["det_sym"] += dc >= dstar
        uppers.append(certify_direction(prob, Theta[:, 0], fit, ALPHA).upper)
        acc["refute"] += refute_all(prob, fit, ALPHA) > DELTA
    out = {k: v / TRIALS for k, v in acc.items()}
    out.update(count_fc_sym=int(acc["fc_sym"]), count_fc_thr=int(acc["fc_thr"]))
    rot = np.zeros(prob.P)
    rot[1], rot[2] = 1.0, -1.0
    out.update(family=family, eps=eps, N=N, sigma_rel=srel, d_true=dstar,
               singular_gap=gap, true_min_defect=float(prob.spectrum(beta0)[0][0]),
               true_rot_defect=float(prob.rho(rot, beta0)) if family == "radial" else None,
               upper_med=float(np.median(uppers)), trials=TRIALS)
    return out


def main():
    cells = []
    i = 0
    for (fam, eps), N, s in product([("radial", e) for e in RADIAL_EPS] + [("pendulum", 0.0)],
                                    NS, SIGMAS):
        cells.append((fam, eps, N, s, 16000 + 401 * i))
        i += 1
    w0 = time.time()
    rows = run_cells(one_cell, cells, desc="exp16")
    save("exp16_dictionary", {"rows": rows, "trials": TRIALS, "Ns": NS,
                              "sigmas": SIGMAS, "radial_eps": RADIAL_EPS})
    print(f"exp16 wall {time.time()-w0:.0f}s")


if __name__ == "__main__":
    main()
