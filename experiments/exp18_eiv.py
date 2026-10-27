"""Experiment 18 -- noisy states (errors in variables) break the guarantee.

The certificate assumes the regressors -- the states -- are observed exactly.
Here the states are observed with independent Gaussian noise of standard
deviation ``sigma_x`` while the right-hand side is measured separately with
small independent noise.  Least squares then estimates a *pseudo-true* field:
measurement error attenuates the nonlinear terms more than the linear ones, so
the fitted field looks more linear and, since linear fields have large
symmetry algebras, more symmetric than the system is.  For the hardening
Duffing oscillator ``x' = y, y' = -x - 0.3 x^3 - 0.2 y`` (smallest true
defect 0.102) the pseudo-true smallest defect falls below the tolerance for a
range of ``sigma_x``; as ``N`` grows the confidence region concentrates on the
pseudo-true coefficients and certification of a false symmetry becomes
likely.  The symmetric Hopf normal form is run alongside.

Recorded per setting: coverage of the coefficient region (against the true
coefficients), false-certification rate, certified dimension, the median upper
bound on the minimal-defect generator, and the pseudo-true smallest defect
computed from a reference fit on ``10^6`` noisy states.
"""

from __future__ import annotations

import time
from fractions import Fraction
from itertools import product

import numpy as np

from common import ALPHA, DELTA, rms_field, run_cells, save, sup_true_defect
from common import trials
from hsd import DefectProblem, exact_symmetry_algebra, fit_ols, get_system
from hsd.certify import certified_dimension, certify_direction
from hsd.estimation import region_covers
from hsd.systems import System

TRIALS = trials(200)
STATE_NOISE = [0.0, 0.04, 0.08, 0.12]
NS = [3200, 25600, 204800]
SIGMA_Y_REL = 0.02


def hard_duffing() -> System:
    return System("hard_duffing", 2, 3,
                  {(0, (0, 1)): Fraction(1), (1, (1, 0)): Fraction(-1),
                   (1, (3, 0)): Fraction(-3, 10), (1, (0, 1)): Fraction(-1, 5)},
                  "Hardening Duffing oscillator x' = y, y' = -x - 0.3 x^3 - 0.2 y.")


SYSTEMS = {"hard_duffing": hard_duffing, "hopf": lambda: get_system("hopf")}


def pseudo_true_min_defect(sys_, prob, sx, rng):
    X = rng.uniform(-1.0, 1.0, size=(1_000_000, 2))
    Xt = X + sx * rng.normal(size=X.shape)
    return float(prob.spectrum(fit_ols(prob, Xt, sys_.rhs(X)).beta)[0][0])


def one_cell(args):
    name, sx, N, seed = args
    rng = np.random.default_rng(seed)
    sys_ = SYSTEMS[name]()
    prob = DefectProblem.build(2, 3, "affine", "box", scale=1.0)
    beta0 = sys_.beta(prob)
    dstar = exact_symmetry_algebra(sys_, prob).shape[1]
    sy = SIGMA_Y_REL * rms_field(sys_, np.random.default_rng(0), 1.0)
    reg = fc = fc_thr = 0
    dims, uppers, dims_thr = [], [], []
    for _ in range(TRIALS):
        X = rng.uniform(-1.0, 1.0, size=(N, 2))
        Y = sys_.rhs(X) + sy * rng.normal(size=X.shape)
        Xt = X + sx * rng.normal(size=X.shape)
        fit = fit_ols(prob, Xt, Y)
        reg += region_covers(fit, beta0, ALPHA)
        dc, Vc = certified_dimension(prob, fit, DELTA, ALPHA)
        dims.append(dc)
        fc += sup_true_defect(prob, Vc, beta0) > DELTA
        rho_hat, Theta = prob.spectrum(fit.beta)
        uppers.append(certify_direction(prob, Theta[:, 0], fit, ALPHA).upper)
        d = int(np.sum(rho_hat <= DELTA))          # fixed threshold at the tolerance
        dims_thr.append(d)
        fc_thr += sup_true_defect(prob, Theta[:, :d], beta0) > DELTA
    return dict(system=name, state_noise=sx, N=N, d_true=dstar, trials=TRIALS,
                true_min_defect=float(prob.spectrum(beta0)[0][0]),
                pseudo_true_min_defect=pseudo_true_min_defect(
                    sys_, prob, sx, np.random.default_rng(seed + 1)),
                region_coverage=reg / TRIALS, fc=fc / TRIALS, count_fc=int(fc),
                fc_threshold=fc_thr / TRIALS, mean_dim_threshold=float(np.mean(dims_thr)),
                mean_dim=float(np.mean(dims)), detect=float(np.mean(np.array(dims) >= dstar)),
                upper_med=float(np.median(uppers)))


def main():
    cells = [(nm, sx, N, 18000 + 157 * i)
             for i, (nm, sx, N) in enumerate(product(list(SYSTEMS), STATE_NOISE, NS))]
    w0 = time.time()
    rows = run_cells(one_cell, cells, desc="exp18")
    save("exp18_eiv", {"rows": rows, "trials": TRIALS, "state_noise": STATE_NOISE,
                       "Ns": NS, "sigma_y_rel": SIGMA_Y_REL})
    print(f"exp18 wall {time.time()-w0:.0f}s")


if __name__ == "__main__":
    main()
