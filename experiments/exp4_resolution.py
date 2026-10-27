"""Experiment 4 -- how small a symmetry violation can be certified away.

Exact symmetry can never be confirmed from noisy data: a system whose defect is
zero and one whose defect is a hair above zero produce almost the same data.
The right question is therefore quantitative -- what is the smallest tolerance
``delta`` at which a procedure may honestly say "this generator is a
``delta``-approximate symmetry"?

We answer it two ways on the one-parameter family ``broken_hopf(eps)``:

* empirically, by recording the certified tolerance as a function of ``N`` and
  the noise level;
* theoretically, by a non-asymptotic Le Cam two-point bound giving the smallest
  tolerance any procedure with false-certification rate ``alpha`` and power
  ``gamma`` can attain, evaluated on the actual design of each replication.

The comparison is made at a fixed model class (degree 3, ``Q = 20``); a second
sweep fits the same symmetric system in classes of degree 3, 5 and 7 to show
how the gap between the certificate and the bound grows with ``Q``.
"""

from __future__ import annotations

import time
from itertools import product

import numpy as np

from common import ALPHA, run_cells, rms_field, save
from common import trials
from hsd import DefectProblem, fit_ols, simulate_design
from hsd.certify import certify_direction
from hsd.systems import broken_hopf

NS = [100, 200, 400, 800, 1600, 3200, 6400, 12800]
SIGMAS = [0.03, 0.1, 0.3]
EPS = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4]
TRIALS = trials(200)
QDEGREES = [3, 5, 7]


def lecam_min_tolerance(prob, theta, beta0, sigma, X, alpha=ALPHA, gamma=0.5):
    """Smallest tolerance at which *any* procedure can reach power ``gamma``.

    Non-asymptotic two-point bound (Proposition on the certifiable tolerance)
    for the fixed design ``X``: with ``Delta`` the top right singular vector of
    the bracket map in the ``L2(mu_N)`` geometry of the design, normalised so
    that ``||F_Delta||_{L2(mu_N)} = 1``,

        delta_min = 2 sigma s_max (gamma - alpha)
                    / ( sqrt(N) ||M beta0|| + 2 sigma (gamma - alpha) ||M Delta|| ).
    """
    L = prob.L_theta(theta)
    M = prob.M_theta(theta)
    Phi = prob.features(X)
    N = X.shape[0]
    Gb = np.kron(np.eye(prob.n), Phi.T @ Phi / N)
    w, U = np.linalg.eigh((Gb + Gb.T) / 2)
    Gih = (U / np.sqrt(np.clip(w, 1e-300, None))) @ U.T
    _, s, Vt = np.linalg.svd(L @ Gih)
    s_max = float(s[0])
    Delta = Gih @ Vt[0]                               # ||F_Delta||_{L2(mu_N)} = 1
    c = 2.0 * sigma * (gamma - alpha)
    return c * s_max / (np.sqrt(N) * float(np.linalg.norm(M @ beta0))
                        + c * float(np.linalg.norm(M @ Delta)))


def _rot(prob):
    rot = np.zeros(prob.P)
    rot[1], rot[2] = 1.0, -1.0
    return rot / np.linalg.norm(rot)


def one_cell(args):
    eps, N, srel, degF, seed = args
    rng = np.random.default_rng(seed)
    sys_ = broken_hopf(eps)
    prob = DefectProblem.build(2, degF, "affine", "box", scale=1.0)
    beta0 = sys_.beta(prob)
    rot = _rot(prob)
    true_rho = prob.rho(rot, beta0)
    sigma = srel * rms_field(sys_, np.random.default_rng(0), 1.0)

    uppers, lowers, plugs, uppers_pt, uppers_sep, lecam = [], [], [], [], [], []
    for _ in range(TRIALS):
        X, Y = simulate_design(sys_, rng, N=N, sigma=sigma, sampler="box", scale=1.0)
        fit = fit_ols(prob, X, Y)
        c = certify_direction(prob, rot, fit, ALPHA, "simultaneous")
        uppers.append(c.upper)
        lowers.append(c.lower)
        plugs.append(c.plug_in)
        uppers_sep.append(certify_direction(prob, rot, fit, ALPHA, ratio="separate").upper)
        # The rotation is fixed in advance here, so the tighter pointwise bound
        # is legitimate without splitting; it isolates the cost of simultaneity.
        uppers_pt.append(certify_direction(prob, rot, fit, ALPHA, "pointwise").upper)
        if eps == 0.0:
            lecam.append(lecam_min_tolerance(prob, rot, beta0, sigma, X))
    uppers, lowers, plugs = map(np.asarray, (uppers, lowers, plugs))
    uppers_pt = np.asarray(uppers_pt)
    return dict(eps=eps, N=N, sigma_rel=srel, degF=degF, Q=prob.Q,
                true_defect=float(true_rho),
                upper_med=float(np.median(uppers)),
                upper_separate_med=float(np.median(uppers_sep)),
                upper_pointwise_med=float(np.median(uppers_pt)),
                upper_q90=float(np.quantile(uppers, 0.9)),
                lower_med=float(np.median(lowers)),
                plug_med=float(np.median(plugs)),
                coverage=float(np.mean((lowers <= true_rho + 1e-12)
                                       & (true_rho <= uppers + 1e-12))),
                refute_rate=float(np.mean(lowers > 1e-9)),
                refute_count=int(np.sum(lowers > 1e-9)),
                lecam=float(np.median(lecam)) if lecam else float("nan"))


def main():
    cells = [(e, N, s, 3, 4000 + 3931 * i)
             for i, (e, N, s) in enumerate(product(EPS, NS, SIGMAS))]
    # Dependence on the model class: the same symmetric system fitted in
    # classes of degree 3, 5 and 7 (Q = 20, 42, 72).
    qcells = [(0.0, N, 0.1, d, 4700 + 811 * i)
              for i, (d, N) in enumerate(product(QDEGREES, NS))]
    w0 = time.time()
    rows = run_cells(one_cell, cells, desc="exp4")
    qrows = run_cells(one_cell, qcells, desc="exp4q")
    save("exp4_resolution", {"rows": rows, "qsweep": qrows, "trials": TRIALS,
                             "Ns": NS, "sigmas": SIGMAS, "eps": EPS,
                             "qdegrees": QDEGREES})
    print(f"exp4 wall {time.time()-w0:.0f}s")


if __name__ == "__main__":
    main()
