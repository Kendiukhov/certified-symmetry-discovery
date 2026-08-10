"""Experiment 13 -- does the conclusion depend on the tolerance?

The tolerance ``delta`` is a user choice, and a comparison made at one value of
it proves little.  We repeat the calibration comparison across two decades of
``delta``, on a subset of the benchmark systems, recording for each the
worst-case false-certification rate and the detection rate of both the
threshold rule matched to the same ``delta`` and of SymCert.
"""

from __future__ import annotations

import time
from itertools import product

import numpy as np

from common import ALPHA, rms_field, run_cells, save, sup_true_defect
from hsd import DefectProblem, exact_symmetry_algebra, fit_ols, get_system, simulate_design
from hsd.certify import certified_dimension

DELTAS = [0.01, 0.02, 0.05, 0.10, 0.20, 0.40]
SYSTEMS = [("hopf", 3, 1.0), ("lin_rot2", 1, 1.0), ("rigid_sym", 2, 1.0),
           ("vanderpol", 3, 1.0), ("lotka_volterra", 2, 1.0), ("rigid_asym", 2, 1.0)]
NS = [400, 3200]
SIGMAS = [0.03, 0.10]
TRIALS = 300


def one_cell(args):
    name, degF, scale, N, srel, seed = args
    rng = np.random.default_rng(seed)
    sys_ = get_system(name)
    prob = DefectProblem.build(sys_.n, degF, "affine", "box", scale=scale)
    beta0 = sys_.beta(prob)
    dstar = exact_symmetry_algebra(sys_, prob).shape[1]
    sigma = srel * rms_field(sys_, np.random.default_rng(0), scale)
    acc = {f"{k}_{d}": 0.0 for d in DELTAS
           for k in ("symfc", "symdet", "symdim", "taufc", "taudet", "taudim")}
    for _ in range(TRIALS):
        X, Y = simulate_design(sys_, rng, N=N, sigma=sigma, sampler="box", scale=scale)
        fit = fit_ols(prob, X, Y)
        rho_hat, Theta = prob.spectrum(fit.beta)
        for d in DELTAS:
            dc, Vc = certified_dimension(prob, fit, d, ALPHA)
            acc[f"symdim_{d}"] += dc
            acc[f"symfc_{d}"] += sup_true_defect(prob, Vc, beta0) > d
            acc[f"symdet_{d}"] += dc >= dstar
            dt = int(np.sum(rho_hat <= d))
            acc[f"taudim_{d}"] += dt
            acc[f"taufc_{d}"] += sup_true_defect(prob, Theta[:, :dt], beta0) > d
            acc[f"taudet_{d}"] += dt >= dstar
    out = {k: v / TRIALS for k, v in acc.items()}
    out.update(system=name, N=N, sigma_rel=srel, d_true=dstar)
    return out


def main():
    cells = [(nm, dg, sc, N, s, 13_000 + 4093 * i)
             for i, ((nm, dg, sc), N, s) in enumerate(product(SYSTEMS, NS, SIGMAS))]
    w0 = time.time()
    rows = run_cells(one_cell, cells, desc="exp13")
    save("exp13_tolerance", {"rows": rows, "deltas": DELTAS, "trials": TRIALS})
    print(f"exp13 wall {time.time()-w0:.0f}s")


if __name__ == "__main__":
    main()
