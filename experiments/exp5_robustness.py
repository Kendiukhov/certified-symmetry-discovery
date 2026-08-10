"""Experiment 5 -- what the certificate needs, and what happens when it is missing.

The exact finite-sample statement assumes homoskedastic Gaussian errors on the
right-hand side.  We check three departures:

1. **Non-Gaussian errors** (Student-t with 4 degrees of freedom, Laplace) and
   **heteroskedastic** errors, with the exact-Gaussian ellipsoid and with a
   sandwich covariance.
2. **Trajectory data**, where states are sampled along solutions rather than
   from a design, so the rows are strongly dependent.
3. **Derivatives estimated from noisy states** by finite differences, which
   violates the model in the worst way: the regressors themselves are noisy, so
   the estimator is biased and no covariance correction can repair it.

Coverage of the certificate -- the fraction of trials in which the reported
interval contains the true defect -- is reported for every case, including the
one where it fails.
"""

from __future__ import annotations

import time
from itertools import product

import numpy as np
from scipy.integrate import solve_ivp

from common import ALPHA, DELTA, run_cells, rms_field, save, sup_true_defect
from hsd import DefectProblem, fit_ols, get_system, simulate_design
from hsd.certify import certified_dimension, certify_direction

TRIALS = 400
SYSTEMS = [("hopf", 3, 1.0), ("vanderpol", 3, 1.0), ("rigid_asym", 2, 1.0)]


def coverage_cell(args):
    name, degF, scale, noise, hetero, robust, N, srel, seed = args
    rng = np.random.default_rng(seed)
    sys_ = get_system(name)
    prob = DefectProblem.build(sys_.n, degF, "affine", "box", scale=scale)
    beta0 = sys_.beta(prob)
    sigma = srel * rms_field(sys_, np.random.default_rng(0), scale)
    cov_hits = 0
    fc = 0
    dims = 0
    for _ in range(TRIALS):
        X, Y = simulate_design(sys_, rng, N=N, sigma=sigma, sampler="box",
                               scale=scale, noise=noise, hetero=hetero)
        fit = fit_ols(prob, X, Y, robust=robust)
        _, Theta = prob.spectrum(fit.beta)
        ok = True
        for k in range(prob.P):
            c = certify_direction(prob, Theta[:, k], fit, ALPHA)
            t = prob.rho(Theta[:, k], beta0)
            if not (c.lower - 1e-12 <= t <= c.upper + 1e-12):
                ok = False
        cov_hits += ok
        dc, Vc = certified_dimension(prob, fit, DELTA, ALPHA)
        dims += dc
        fc += sup_true_defect(prob, Vc, beta0) > DELTA
    return dict(system=name, noise=noise, hetero=hetero, robust=robust, N=N,
                sigma_rel=srel, coverage=cov_hits / TRIALS,
                fc=fc / TRIALS, dim=dims / TRIALS, nominal=1 - ALPHA)


def trajectory_cell(args):
    """Dependent rows and, optionally, noisy states with finite-difference
    derivatives."""
    name, degF, scale, mode, srel, seed = args
    rng = np.random.default_rng(seed)
    sys_ = get_system(name)
    prob = DefectProblem.build(sys_.n, degF, "affine", "box", scale=scale)
    beta0 = sys_.beta(prob)
    sigma = srel * rms_field(sys_, np.random.default_rng(0), scale)
    n = sys_.n
    dt = 0.01
    cov_hits = fc = dims = 0
    trials = TRIALS // 4
    for _ in range(trials):
        Xs, Ys = [], []
        for _ in range(12):
            x0 = rng.uniform(-scale, scale, size=n)
            sol = solve_ivp(lambda t, y: sys_.rhs(y[None, :])[0], [0, 2.0], x0,
                            t_eval=np.arange(0, 2.0, dt), rtol=1e-11, atol=1e-13)
            traj = sol.y.T
            keep = np.all(np.abs(traj) <= scale, axis=1)
            traj = traj[keep]
            if len(traj) < 5:
                continue
            if mode == "clean_states":
                Xs.append(traj)
                Ys.append(sys_.rhs(traj) + sigma * rng.normal(size=traj.shape))
            else:                                  # noisy states + central differences
                obs = traj + sigma * dt * rng.normal(size=traj.shape)
                d = (obs[2:] - obs[:-2]) / (2 * dt)
                Xs.append(obs[1:-1])
                Ys.append(d)
        X, Y = np.vstack(Xs), np.vstack(Ys)
        fit = fit_ols(prob, X, Y, robust=True)
        _, Theta = prob.spectrum(fit.beta)
        ok = True
        for k in range(prob.P):
            c = certify_direction(prob, Theta[:, k], fit, ALPHA)
            t = prob.rho(Theta[:, k], beta0)
            if not (c.lower - 1e-12 <= t <= c.upper + 1e-12):
                ok = False
        cov_hits += ok
        dc, Vc = certified_dimension(prob, fit, DELTA, ALPHA)
        dims += dc
        fc += sup_true_defect(prob, Vc, beta0) > DELTA
    return dict(system=name, mode=mode, sigma_rel=srel, coverage=cov_hits / trials,
                fc=fc / trials, dim=dims / trials, nominal=1 - ALPHA, n_traj_trials=trials)


def main():
    w0 = time.time()
    cells = []
    i = 0
    # Two regimes: one where nothing is certifiable (so coverage is the only
    # thing being tested) and one where the true algebra is recovered, so that
    # the table also shows the method still works under the wrong error model.
    for (nm, dg, sc), (noise, het), (N, srel) in product(
            SYSTEMS, [("gauss", 0.0), ("t", 0.0), ("laplace", 0.0), ("gauss", 1.0)],
            [(400, 0.1), (3200, 0.03)]):
        for robust in (False, True):
            cells.append((nm, dg, sc, noise, het, robust, N, srel, 5000 + 811 * i))
            i += 1
    rows = run_cells(coverage_cell, cells, desc="exp5a")
    # Sweep the state-noise level as well: errors in the regressors bias the
    # estimator, and no covariance correction repairs that, so we must show
    # where the guarantee actually breaks rather than assert that it does not.
    tcells = [(nm, dg, sc, mode, sr, 5500 + 313 * j)
              for j, ((nm, dg, sc), mode, sr) in enumerate(
                  product(SYSTEMS, ["clean_states", "noisy_states_fd"],
                          [0.01, 0.05, 0.2, 0.5]))]
    trows = run_cells(trajectory_cell, tcells, desc="exp5b")
    save("exp5_robustness", {"design": rows, "trajectory": trows, "trials": TRIALS})
    print(f"exp5 wall {time.time()-w0:.0f}s")


if __name__ == "__main__":
    main()
