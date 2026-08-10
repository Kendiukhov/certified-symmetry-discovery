"""Experiment 6 -- problem dimension, and the price of sample splitting.

Otto et al. (2025) list "the effects of problem dimension, noise level and
amount of data" as an open question.  The width of the simultaneous certificate
is driven by the number of model coefficients ``Q`` through the radius of the
confidence ellipsoid, while the split certificate pays instead an effective
dimension that depends only on the generator being tested -- but sees half the
data.  We measure which wins as the model class grows.
"""

from __future__ import annotations

import time
from itertools import product

import numpy as np

from common import ALPHA, DELTA, rms_field, run_cells, save
from hsd import DefectProblem, exact_symmetry_algebra, fit_ols, get_system, simulate_design
from hsd.certify import certify_direction

TRIALS = 300
CONFIGS = [
    # (system, model degree, candidate class, target half-width)
    ("hopf", 3, "affine", 1.0),
    ("hopf", 5, "affine", 1.0),
    ("hopf", 3, "quadratic", 1.0),
    ("sphere_flow3", 3, "affine", 1.0),
    ("sphere_flow3", 3, "quadratic", 1.0),
    ("lin_rot2", 1, "affine", 1.0),
    ("lin_rot2", 3, "affine", 1.0),
    ("rigid_sym", 2, "affine", 1.0),
    ("rigid_sym", 4, "affine", 1.0),
]
NS = [200, 400, 800, 1600, 3200, 6400]


def one_cell(args):
    name, degF, gens, scale, N, srel, seed = args
    rng = np.random.default_rng(seed)
    sys_ = get_system(name)
    prob = DefectProblem.build(sys_.n, degF, gens, "box", scale=scale)
    beta0 = sys_.beta(prob)
    Strue = exact_symmetry_algebra(sys_, prob)
    dstar = Strue.shape[1]
    sigma = srel * rms_field(sys_, np.random.default_rng(0), scale)
    us, up, det_s, det_p, fc_s, fc_p = [], [], 0, 0, 0, 0
    for _ in range(TRIALS):
        X, Y = simulate_design(sys_, rng, N=N, sigma=sigma, sampler="box", scale=scale)
        fit = fit_ols(prob, X, Y)
        _, Theta = prob.spectrum(fit.beta)
        cs = certify_direction(prob, Theta[:, 0], fit, ALPHA, "simultaneous")
        us.append(cs.upper)
        det_s += cs.upper <= DELTA
        fc_s += (cs.upper <= DELTA) and prob.rho(Theta[:, 0], beta0) > DELTA

        half = N // 2
        fA = fit_ols(prob, X[:half], Y[:half])
        fB = fit_ols(prob, X[half:], Y[half:])
        _, ThA = prob.spectrum(fA.beta)
        cp = certify_direction(prob, ThA[:, 0], fB, ALPHA, "pointwise")
        up.append(cp.upper)
        det_p += cp.upper <= DELTA
        fc_p += (cp.upper <= DELTA) and prob.rho(ThA[:, 0], beta0) > DELTA
    return dict(system=name, degF=degF, gens=gens, N=N, sigma_rel=srel,
                P=prob.P, Q=prob.Q, d_true=dstar,
                upper_sim=float(np.median(us)), upper_split=float(np.median(up)),
                detect_sim=det_s / TRIALS, detect_split=det_p / TRIALS,
                fc_sim=fc_s / TRIALS, fc_split=fc_p / TRIALS)


def main():
    cells = [(nm, dg, gn, sc, N, 0.05, 6000 + 2179 * i)
             for i, ((nm, dg, gn, sc), N) in enumerate(product(CONFIGS, NS))]
    w0 = time.time()
    rows = run_cells(one_cell, cells, desc="exp6")
    save("exp6_dimension", {"rows": rows, "trials": TRIALS, "Ns": NS})
    print(f"exp6 wall {time.time()-w0:.0f}s")


if __name__ == "__main__":
    main()
