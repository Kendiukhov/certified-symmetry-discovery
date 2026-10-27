"""Experiment 15 -- serially dependent errors along trajectories.

States are observed exactly along ``G`` solution segments of the system and
the right-hand side is observed with errors that follow a stationary AR(1)
process within each segment,

    e_t = phi e_{t-1} + sqrt(1 - phi^2) sigma z_t ,   z_t ~ N(0, I),

independent across segments.  The rows are then dependent in their errors, not
only in their design, and the ordinary-least-squares region and White's
sandwich both understate the variance of the coefficients by roughly the
factor ``(1 + phi) / (1 - phi)``.  Two estimators are designed for this
situation: the cluster-robust sandwich over segments (valid as the number of
segments grows) and the Newey--West sandwich within segments with Andrews'
automatic bandwidth (valid as the segments grow long).

For each ``(system, phi, G)`` we record, for each estimator, the coverage of
the coefficient region (the primitive event), the coverage of the certificate
interval over all plug-in directions, the false-certification rate and the
certified dimension.
"""

from __future__ import annotations

import time
from itertools import product

import numpy as np

from common import ALPHA, DELTA, rms_field, run_cells, save, sup_true_defect
from common import trials
from hsd import DefectProblem, fit_ols, get_system
from hsd.certify import certified_dimension, certify_direction
from hsd.estimation import region_covers
from hsd.trajectories import segments

TRIALS = trials(200)
SYSTEMS = [("hopf", 3, 1.0), ("rigid_asym", 2, 1.0)]
PHIS = [0.0, 0.5, 0.9, 0.98]
GROUPS = [12, 48, 192]
T_SEGMENT = 1.0
DT = 0.01
SIGMA_REL = 0.05
ESTIMATORS = ("ols", "hc0", "cluster", "hac")


def _ar1(rng, T, n, phi, sigma):
    e = np.empty((T, n))
    e[0] = sigma * rng.normal(size=n)
    s = sigma * np.sqrt(1.0 - phi ** 2)
    for t in range(1, T):
        e[t] = phi * e[t - 1] + s * rng.normal(size=n)
    return e


def _interval_coverage(prob, fit, beta0):
    _, Theta = prob.spectrum(fit.beta)
    for k in range(prob.P):
        c = certify_direction(prob, Theta[:, k], fit, ALPHA)
        t = prob.rho(Theta[:, k], beta0)
        if not (c.lower - 1e-12 <= t <= c.upper + 1e-12):
            return False
    return True


def one_cell(args):
    name, degF, scale, phi, G, seed = args
    rng = np.random.default_rng(seed)
    sys_ = get_system(name)
    prob = DefectProblem.build(sys_.n, degF, "affine", "box", scale=scale)
    beta0 = sys_.beta(prob)
    sigma = SIGMA_REL * rms_field(sys_, np.random.default_rng(0), scale)
    acc = {f"{m}_{e}": 0 for m in ("reg", "cert", "fc", "dim") for e in ESTIMATORS}
    finite = {e: 0 for e in ESTIMATORS}
    npts = []
    for _ in range(TRIALS):
        runs, ids = segments(sys_, rng, G, T_SEGMENT, DT, scale)
        X = np.vstack(runs)
        Y = np.vstack([sys_.rhs(r) + _ar1(rng, len(r), sys_.n, phi, sigma) for r in runs])
        groups = np.concatenate([np.full(len(r), g) for g, r in zip(ids, runs)])
        npts.append(len(X))
        for e in ESTIMATORS:
            fit = fit_ols(prob, X, Y, cov=e, groups=groups)
            acc[f"reg_{e}"] += region_covers(fit, beta0, ALPHA)
            acc[f"cert_{e}"] += _interval_coverage(prob, fit, beta0)
            dc, Vc = certified_dimension(prob, fit, DELTA, ALPHA)
            acc[f"dim_{e}"] += dc
            acc[f"fc_{e}"] += sup_true_defect(prob, Vc, beta0) > DELTA
            finite[e] += fit.null_basis is None or fit.null_basis.size == 0
    out = {k: v / TRIALS for k, v in acc.items()}
    out.update({f"count_fc_{e}": int(acc[f"fc_{e}"]) for e in ESTIMATORS})
    out.update({f"identified_{e}": finite[e] / TRIALS for e in ESTIMATORS})
    out.update(system=name, phi=phi, G=G, Q=prob.Q, trials=TRIALS,
               median_points=float(np.median(npts)), nominal=1 - ALPHA)
    return out


def main():
    cells = [(nm, dg, sc, phi, G, 15000 + 211 * i)
             for i, ((nm, dg, sc), phi, G) in enumerate(product(SYSTEMS, PHIS, GROUPS))]
    w0 = time.time()
    rows = run_cells(one_cell, cells, desc="exp15")
    save("exp15_dependence", {"rows": rows, "trials": TRIALS, "phis": PHIS,
                              "groups": GROUPS, "t_segment": T_SEGMENT, "dt": DT,
                              "sigma_rel": SIGMA_REL, "estimators": list(ESTIMATORS)})
    print(f"exp15 wall {time.time()-w0:.0f}s")


if __name__ == "__main__":
    main()
