"""Experiment 2 -- incomplete coverage of the state space.

The defect is defined on a target domain that the user cares about.  When the
data occupy only part of it, brackets that are large on the target domain can be
small on the data, and a nullspace computed on the sample "discovers" symmetries
that do not exist.  We sweep the size of the observed region, measure how often
that happens, and check the closed-form extrapolation factor ``kappa`` against
the observed onset.
"""

from __future__ import annotations

import time
from itertools import product

import numpy as np
from scipy.integrate import solve_ivp

from common import ALPHA, DELTA, run_cells, rms_field, save, sup_true_defect
from hsd import DefectProblem, exact_symmetry_algebra, fit_ols, get_system, simulate_design
from hsd.certify import certified_dimension
from hsd.coverage import extrapolation_factor

SYSTEMS = [("hopf", 3, 1.0), ("vanderpol", 3, 1.0), ("lotka_volterra", 2, 1.0),
           ("rigid_asym", 2, 1.0), ("rigid_sym", 2, 1.0)]
RADII = [1.0, 0.8, 0.6, 0.5, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15]
TRIALS = 200
N = 2000
SIGMA_REL = 0.02


def one_cell(args):
    name, degF, scale, r, seed = args
    rng = np.random.default_rng(seed)
    sys_ = get_system(name)
    prob = DefectProblem.build(sys_.n, degF, "affine", "box", scale=scale)
    beta0 = sys_.beta(prob)
    dstar = exact_symmetry_algebra(sys_, prob).shape[1]
    sigma = SIGMA_REL * rms_field(sys_, np.random.default_rng(0), scale)
    acc = dict(fc_naive_data=0.0, fc_naive_target=0.0, fc_sym=0.0,
               dim_naive_data=0.0, dim_naive_target=0.0, dim_sym=0.0,
               kappa=0.0, log_kappa=0.0)
    for _ in range(TRIALS):
        X, Y = simulate_design(sys_, rng, N=N, sigma=sigma, sampler="box", scale=r)
        fit = fit_ols(prob, X, Y)
        pe = prob.empirical(X)
        rhoD, ThD = pe.spectrum(fit.beta)
        dD = int(np.sum(rhoD <= DELTA))
        acc["dim_naive_data"] += dD
        acc["fc_naive_data"] += sup_true_defect(prob, ThD[:, :dD], beta0) > DELTA

        rhoT, ThT = prob.spectrum(fit.beta)
        dT = int(np.sum(rhoT <= DELTA))
        acc["dim_naive_target"] += dT
        acc["fc_naive_target"] += sup_true_defect(prob, ThT[:, :dT], beta0) > DELTA

        dc, Vc = certified_dimension(prob, fit, DELTA, ALPHA)
        acc["dim_sym"] += dc
        acc["fc_sym"] += sup_true_defect(prob, Vc, beta0) > DELTA

        k = extrapolation_factor(prob, X)["bound"]
        acc["kappa"] += min(k, 1e12)
        acc["log_kappa"] += np.log10(min(k, 1e12))
    out = {k: v / TRIALS for k, v in acc.items()}
    rho_true = prob.spectrum(beta0)[0]
    above = rho_true[rho_true > DELTA]
    # Prop. 8: a spurious delta-symmetry needs the extrapolation bound
    # sqrt(kappa kappa') to reach rho*/delta, where rho* is the smallest true
    # defect that is not already within tolerance.
    out.update(system=name, radius=r, d_true=dstar,
               true_min_defect=float(rho_true[0]),
               rho_star=float(above[0]) if above.size else float("inf"),
               bound_critical=float(above[0] / DELTA) if above.size else float("inf"))
    return out


def trajectory_study():
    """Same question, with data drawn from trajectories rather than a design."""
    rng = np.random.default_rng(11)
    out = []
    for name, degF, scale in [("vanderpol", 3, 3.0), ("hopf", 3, 1.5), ("lorenz", 2, 20.0)]:
        sys_ = get_system(name)
        prob = DefectProblem.build(sys_.n, degF, "affine", "box", scale=scale)
        beta0 = sys_.beta(prob)
        sigma = 0.02 * rms_field(sys_, np.random.default_rng(0), scale)

        def integrate(x0, T, npts):
            sol = solve_ivp(lambda t, y: sys_.rhs(y[None, :])[0], [0, T], x0,
                            t_eval=np.linspace(0, T, npts), rtol=1e-10, atol=1e-12)
            return sol.y.T

        designs = {}
        n = sys_.n
        x0 = np.full(n, 0.7 * scale) if name != "lorenz" else np.array([1.0, 1.0, 20.0])
        tr = integrate(x0, 60 if name != "lorenz" else 40, 4000)
        designs["single trajectory"] = tr[1000:]
        starts = rng.uniform(-scale, scale, size=(8, n))
        designs["8 trajectories"] = np.vstack([integrate(s, 8, 400) for s in starts])
        designs["uniform design"] = rng.uniform(-scale, scale, size=(3000, n))
        for label, X in designs.items():
            X = X[np.all(np.abs(X) <= 3 * scale, axis=1)]
            Y = sys_.rhs(X) + sigma * rng.normal(size=X.shape)
            fit = fit_ols(prob, X, Y)
            pe = prob.empirical(X)
            rhoD, ThD = pe.spectrum(fit.beta)
            dD = int(np.sum(rhoD <= DELTA))
            rhoT, ThT = prob.spectrum(fit.beta)
            dT = int(np.sum(rhoT <= DELTA))
            dc, Vc = certified_dimension(prob, fit, DELTA, ALPHA)
            out.append(dict(system=name, design=label, n_points=int(X.shape[0]),
                            kappa=float(extrapolation_factor(prob, X)["bound"]),
                            dim_naive_data=dD, dim_naive_target=dT, dim_sym=dc,
                            fc_naive_data=bool(sup_true_defect(prob, ThD[:, :dD], beta0) > DELTA),
                            fc_naive_target=bool(sup_true_defect(prob, ThT[:, :dT], beta0) > DELTA),
                            fc_sym=bool(sup_true_defect(prob, Vc, beta0) > DELTA),
                            d_true=exact_symmetry_algebra(sys_, prob).shape[1]))
    return out


def main():
    cells = [(nm, dg, sc, r, 2000 + 6421 * i)
             for i, ((nm, dg, sc), r) in enumerate(product(SYSTEMS, RADII))]
    w0 = time.time()
    rows = run_cells(one_cell, cells, desc="exp2")
    traj = trajectory_study()
    save("exp2_coverage", {"rows": rows, "trajectories": traj, "trials": TRIALS,
                           "N": N, "sigma_rel": SIGMA_REL, "radii": RADII})
    print(f"exp2 wall {time.time()-w0:.0f}s")


if __name__ == "__main__":
    main()
