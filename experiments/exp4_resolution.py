"""Experiment 4 -- how small a symmetry violation can be certified away.

Exact symmetry can never be confirmed from noisy data: a system whose defect is
zero and one whose defect is a hair above zero produce almost the same data.
The right question is therefore quantitative -- what is the smallest tolerance
``delta`` at which a procedure may honestly say "this generator is a
``delta``-approximate symmetry"?

We answer it two ways on the one-parameter family ``broken_hopf(eps)``:

* empirically, by recording the certified tolerance as a function of ``N`` and
  the noise level;
* theoretically, by a Le Cam two-point bound giving the smallest tolerance any
  procedure with false-certification rate ``alpha`` and power ``gamma`` can
  attain.  Both constants are computable in closed form for this model, so the
  comparison is a like-for-like efficiency ratio rather than a rate statement.
"""

from __future__ import annotations

import time
from itertools import product

import numpy as np

from common import ALPHA, run_cells, rms_field, save
from hsd import DefectProblem, fit_ols, simulate_design
from hsd.certify import certify_direction
from hsd.polynomials import moment_matrix
from hsd.systems import broken_hopf

NS = [100, 200, 400, 800, 1600, 3200, 6400, 12800]
SIGMAS = [0.03, 0.1, 0.3]
EPS = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4]
TRIALS = 200


def lecam_min_tolerance(prob, theta, beta0, sigma, N, alpha=ALPHA, gamma=0.5):
    """Smallest tolerance at which *any* procedure can reach power ``gamma``.

    Two-point argument.  Let ``beta0`` have zero defect along ``theta`` and let
    ``beta1 = beta0 + t D`` have defect ``delta``.  Pinsker's inequality gives
    ``gamma <= alpha + sqrt(N/(4 sigma^2)) t ||D||_{L2(mu)}``, and the direction
    ``D`` that makes ``t`` smallest for a given ``delta`` is the top right
    singular vector of the bracket map in the ``L2(mu)`` geometry.  Solving for
    ``delta`` gives the bound returned here.
    """
    L = prob.L_theta(theta)
    M = prob.M_theta(theta)
    g = moment_matrix(prob.alg, prob.measure, prob.scale, sub=prob.beta_idx)
    Gb = np.kron(np.eye(prob.n), g)
    w, U = np.linalg.eigh((Gb + Gb.T) / 2)
    Gih = (U / np.sqrt(np.clip(w, 1e-300, None))) @ U.T
    s_max = float(np.linalg.svd(L @ Gih, compute_uv=False)[0])
    denom = float(np.linalg.norm(M @ beta0))
    return 2.0 * sigma * s_max * (gamma - alpha) / (denom * np.sqrt(N))


def one_cell(args):
    eps, N, srel, seed = args
    rng = np.random.default_rng(seed)
    sys_ = broken_hopf(eps)
    prob = DefectProblem.build(2, 3, "affine", "box", scale=1.0)
    beta0 = sys_.beta(prob)
    rot = np.zeros(prob.P)
    rot[1], rot[2] = 1.0, -1.0
    rot /= np.linalg.norm(rot)
    true_rho = prob.rho(rot, beta0)
    sigma = srel * rms_field(sys_, np.random.default_rng(0), 1.0)

    uppers, lowers, plugs, uppers_pt = [], [], [], []
    for _ in range(TRIALS):
        X, Y = simulate_design(sys_, rng, N=N, sigma=sigma, sampler="box", scale=1.0)
        fit = fit_ols(prob, X, Y)
        c = certify_direction(prob, rot, fit, ALPHA, "simultaneous")
        uppers.append(c.upper)
        lowers.append(c.lower)
        plugs.append(c.plug_in)
        # The rotation is fixed in advance here, so the tighter pointwise bound
        # is legitimate without splitting; it isolates the cost of simultaneity.
        uppers_pt.append(certify_direction(prob, rot, fit, ALPHA, "pointwise").upper)
    uppers, lowers, plugs = map(np.asarray, (uppers, lowers, plugs))
    uppers_pt = np.asarray(uppers_pt)
    return dict(eps=eps, N=N, sigma_rel=srel, true_defect=float(true_rho),
                upper_med=float(np.median(uppers)),
                upper_pointwise_med=float(np.median(uppers_pt)),
                upper_q90=float(np.quantile(uppers, 0.9)),
                lower_med=float(np.median(lowers)),
                plug_med=float(np.median(plugs)),
                coverage=float(np.mean((lowers <= true_rho + 1e-12)
                                       & (true_rho <= uppers + 1e-12))),
                refute_rate=float(np.mean(lowers > 1e-9)),
                lecam=float(lecam_min_tolerance(prob, rot, beta0, sigma, N)))


def main():
    cells = [(e, N, s, 4000 + 3931 * i)
             for i, (e, N, s) in enumerate(product(EPS, NS, SIGMAS))]
    w0 = time.time()
    rows = run_cells(one_cell, cells, desc="exp4")
    save("exp4_resolution", {"rows": rows, "trials": TRIALS, "Ns": NS,
                             "sigmas": SIGMAS, "eps": EPS})
    print(f"exp4 wall {time.time()-w0:.0f}s")


if __name__ == "__main__":
    main()
