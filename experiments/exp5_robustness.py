"""Experiment 5 -- what the certificate needs, and what happens when it is missing.

The exact finite-sample statement assumes homoskedastic Gaussian errors on the
right-hand side, independent of the states.  Every other situation is handled
by supplying a different confidence region for the coefficients, and the
certificate inherits that region's coverage and nothing more.  We therefore
report two coverages throughout:

* **region coverage** -- how often the coefficient region contains the true
  coefficients, the primitive event on which every certificate rests;
* **certificate coverage** -- how often the reported interval contains the true
  defect of every plug-in direction.

Two studies.

1. **Error distribution** on an i.i.d. design: Gaussian, Student-t with 4
   degrees of freedom, Laplace, and heteroskedastic Gaussian errors, each with
   the exact-Gaussian region (``cov="ols"``) and White's sandwich
   (``cov="hc0"``).
2. **Trajectory data**: 48 solution segments per replication, in two modes --
   exact states with noisy right-hand sides, and noisy states with derivatives
   taken by central differences *within each contiguous segment*.  The second
   mode puts error in the regressors and correlates neighbouring rows, which
   no covariance correction repairs; it is reported so that the failure is
   visible at the level of the region, whatever the certificate does.  Each
   replication is analysed with White's sandwich and with the cluster-robust
   sandwich over segments.  Serially correlated errors along trajectories are
   studied separately (``exp15_dependence``).
"""

from __future__ import annotations

import time
from itertools import product

import numpy as np

from common import ALPHA, DELTA, run_cells, rms_field, save, sup_true_defect
from common import trials
from hsd import DefectProblem, fit_ols, get_system, simulate_design
from hsd.certify import certified_dimension, certify_direction
from hsd.estimation import region_covers
from hsd.trajectories import segments

TRIALS = trials(400)
SYSTEMS = [("hopf", 3, 1.0), ("vanderpol", 3, 1.0), ("rigid_asym", 2, 1.0)]
N_SEGMENTS = 48
T_SEGMENT = 1.0
DT = 0.01
STATE_NOISE = [0.0005, 0.002, 0.005]     # absolute state-noise s.d. (box half-width 1)


def _interval_coverage(prob, fit, beta0):
    _, Theta = prob.spectrum(fit.beta)
    for k in range(prob.P):
        c = certify_direction(prob, Theta[:, k], fit, ALPHA)
        t = prob.rho(Theta[:, k], beta0)
        if not (c.lower - 1e-12 <= t <= c.upper + 1e-12):
            return False
    return True


def coverage_cell(args):
    name, degF, scale, noise, hetero, cov, N, srel, seed = args
    rng = np.random.default_rng(seed)
    sys_ = get_system(name)
    prob = DefectProblem.build(sys_.n, degF, "affine", "box", scale=scale)
    beta0 = sys_.beta(prob)
    sigma = srel * rms_field(sys_, np.random.default_rng(0), scale)
    reg = cert = fc = dims = 0
    for _ in range(TRIALS):
        X, Y = simulate_design(sys_, rng, N=N, sigma=sigma, sampler="box",
                               scale=scale, noise=noise, hetero=hetero)
        fit = fit_ols(prob, X, Y, cov=cov)
        reg += region_covers(fit, beta0, ALPHA)
        cert += _interval_coverage(prob, fit, beta0)
        dc, Vc = certified_dimension(prob, fit, DELTA, ALPHA)
        dims += dc
        fc += sup_true_defect(prob, Vc, beta0) > DELTA
    return dict(system=name, noise=noise, hetero=hetero, cov=cov,
                robust=cov != "ols", N=N, sigma_rel=srel,
                region_coverage=reg / TRIALS, coverage=cert / TRIALS,
                fc=fc / TRIALS, fc_count=int(fc), dim=dims / TRIALS,
                nominal=1 - ALPHA, trials=TRIALS)


def trajectory_cell(args):
    name, degF, scale, mode, level, seed = args
    rng = np.random.default_rng(seed)
    sys_ = get_system(name)
    prob = DefectProblem.build(sys_.n, degF, "affine", "box", scale=scale)
    beta0 = sys_.beta(prob)
    sigma = 0.05 * rms_field(sys_, np.random.default_rng(0), scale)
    trials = max(1, TRIALS // 4)
    acc = {f"{k}_{c}": 0 for k in ("reg", "cert", "fc", "dim") for c in ("hc0", "cluster")}
    npts = []
    for _ in range(trials):
        segs, ids = segments(sys_, rng, N_SEGMENTS, T_SEGMENT, DT, scale)
        Xs, Ys, G = [], [], []
        for g, tr in zip(ids, segs):
            if mode == "clean_states":
                Xs.append(tr)
                Ys.append(sys_.rhs(tr) + sigma * rng.normal(size=tr.shape))
                G.append(np.full(len(tr), g))
            else:                                  # noisy states + central differences
                obs = tr + level * rng.normal(size=tr.shape)
                Xs.append(obs[1:-1])
                Ys.append((obs[2:] - obs[:-2]) / (2 * DT))
                G.append(np.full(len(tr) - 2, g))
        X, Y, G = np.vstack(Xs), np.vstack(Ys), np.concatenate(G)
        npts.append(len(X))
        for c in ("hc0", "cluster"):
            fit = fit_ols(prob, X, Y, cov=c, groups=G)
            acc[f"reg_{c}"] += region_covers(fit, beta0, ALPHA)
            acc[f"cert_{c}"] += _interval_coverage(prob, fit, beta0)
            dc, Vc = certified_dimension(prob, fit, DELTA, ALPHA)
            acc[f"dim_{c}"] += dc
            acc[f"fc_{c}"] += sup_true_defect(prob, Vc, beta0) > DELTA
    out = {k: v / trials for k, v in acc.items()}
    out.update(system=name, mode=mode, level=level, n_traj_trials=trials,
               median_points=float(np.median(npts)), nominal=1 - ALPHA,
               fc_count_hc0=int(acc["fc_hc0"]), fc_count_cluster=int(acc["fc_cluster"]))
    return out


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
        for cov in ("ols", "hc0"):
            cells.append((nm, dg, sc, noise, het, cov, N, srel, 5000 + 811 * i))
            i += 1
    rows = run_cells(coverage_cell, cells, desc="exp5a")
    tcells = [(nm, dg, sc, mode, lv, 5500 + 313 * j)
              for j, ((nm, dg, sc), (mode, lv)) in enumerate(
                  product(SYSTEMS, [("clean_states", 0.0)]
                          + [("noisy_states_fd", s) for s in STATE_NOISE]))]
    trows = run_cells(trajectory_cell, tcells, desc="exp5b")
    save("exp5_robustness", {"design": rows, "trajectory": trows, "trials": TRIALS,
                             "n_segments": N_SEGMENTS, "t_segment": T_SEGMENT,
                             "dt": DT, "state_noise": STATE_NOISE})
    print(f"exp5 wall {time.time()-w0:.0f}s")


if __name__ == "__main__":
    main()
