"""Experiment 10 -- what noise actually does to the plug-in defect.

Two measurements that the paper quotes directly.

**(a) Direction of the bias.** Additive noise on the observed right-hand side
inflates both parts of the defect ratio, and inflates the numerator far more for
a direction close to a true symmetry.  We therefore expect the smallest plug-in
defect to be biased *upward*, so that noise causes missed detections rather than
false discoveries.  We push the noise to a half of the signal and the sample
size down to 30, and record the whole lower tail.

**(b) Selection optimism.** The direction chosen to minimise the plug-in defect
has a true defect larger than its plug-in value.  We measure that gap as a
function of the sample size; it is the reason a test evaluated on the data that
chose the direction is invalid.
"""

from __future__ import annotations

import time
from itertools import product

import numpy as np

from common import DELTA, run_cells, rms_field, save
from hsd import DefectProblem, fit_ols, get_system, simulate_design

TRIALS = 500
SYSTEMS = [("vanderpol", 3, 1.0), ("lotka_volterra", 2, 1.0), ("duffing", 3, 2.0),
           ("rigid_asym", 2, 1.0)]


def bias_cell(args):
    name, degF, scale, N, srel, seed = args
    rng = np.random.default_rng(seed)
    sys_ = get_system(name)
    prob = DefectProblem.build(sys_.n, degF, "affine", "box", scale=scale)
    beta0 = sys_.beta(prob)
    true_min = float(prob.spectrum(beta0)[0][0])
    sigma = srel * rms_field(sys_, np.random.default_rng(0), scale)
    mins = np.empty(TRIALS)
    for t in range(TRIALS):
        X, Y = simulate_design(sys_, rng, N=N, sigma=sigma, sampler="box", scale=scale)
        mins[t] = prob.spectrum(fit_ols(prob, X, Y).beta)[0][0]
    return dict(system=name, N=N, sigma_rel=srel, true_min_defect=true_min,
                median=float(np.median(mins)), q05=float(np.quantile(mins, 0.05)),
                q01=float(np.quantile(mins, 0.01)), minimum=float(mins.min()),
                frac_below_delta=float(np.mean(mins <= DELTA)),
                frac_below_true=float(np.mean(mins < true_min)))


def optimism_cell(args):
    name, degF, scale, N, srel, seed = args
    rng = np.random.default_rng(seed)
    sys_ = get_system(name)
    prob = DefectProblem.build(sys_.n, degF, "affine", "box", scale=scale)
    beta0 = sys_.beta(prob)
    sigma = srel * rms_field(sys_, np.random.default_rng(0), scale)
    gaps = np.empty(TRIALS)
    for t in range(TRIALS):
        X, Y = simulate_design(sys_, rng, N=N, sigma=sigma, sampler="box", scale=scale)
        rho, Theta = prob.spectrum(fit_ols(prob, X, Y).beta)
        gaps[t] = prob.rho(Theta[:, 0], beta0) - rho[0]
    return dict(system=name, N=N, sigma_rel=srel, gap_mean=float(gaps.mean()),
                gap_se=float(gaps.std(ddof=1) / np.sqrt(TRIALS)),
                gap_median=float(np.median(gaps)),
                frac_positive=float(np.mean(gaps > 0)))


def main():
    w0 = time.time()
    bcells = [(nm, dg, sc, N, s, 10_000 + 1013 * i)
              for i, ((nm, dg, sc), N, s) in enumerate(product(
                  SYSTEMS, [30, 60, 120, 400], [0.1, 0.3, 0.5, 1.0]))]
    bias = run_cells(bias_cell, bcells, desc="exp10a")
    ocells = [(nm, dg, sc, N, 0.2, 10_500 + 733 * i)
              for i, ((nm, dg, sc), N) in enumerate(product(
                  SYSTEMS, [100, 200, 400, 800, 1600, 3200, 6400]))]
    opt = run_cells(optimism_cell, ocells, desc="exp10b")
    save("exp10_noise", {"bias": bias, "optimism": opt, "trials": TRIALS})
    print(f"exp10 wall {time.time()-w0:.0f}s")


if __name__ == "__main__":
    main()
