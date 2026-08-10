"""Experiment 9 -- paying honestly for approximation error.

When the true vector field is not in the fitted class, validity has to be bought
with an explicit bound on the approximation error.  We compare the two forms
implemented in :func:`hsd.certify.certify_direction`:

* a supremum-norm bound valid for *any* smooth error, which is safe but spends
  worst-case norms;
* an ``L2`` bound valid when the truth can be bracketed by a richer polynomial
  class, which is far tighter.

and we sweep the degree of the fitted class to show how the price falls as the
class grows.  The test systems are genuinely non-polynomial: one with an exact
rotational symmetry, one with none.
"""

from __future__ import annotations

import time

import numpy as np

from common import ALPHA, save
from hsd import DefectProblem, fit_ols
from hsd.certify import certify_direction
from hsd.defect import _deriv_field
from hsd.nonpoly import SMOOTH_SYSTEMS, mc_defect

DEGREES = [3, 5, 7, 9]
N = 40_000
SIGMA_REL = 0.02


def measure_model_error(prob, fit, sm, rng, scale, n_mc=400_000):
    """Supremum and ``L2(nu)`` sizes of the fitted model's approximation error."""
    X = rng.uniform(-scale, scale, size=(n_mc, prob.n))
    F = prob.field_of_beta(fit.beta)
    h = sm.rhs(X) - prob.alg.eval(F, X)
    eta0 = float(np.linalg.norm(h, axis=1).max())
    J = np.stack([prob.alg.eval(_deriv_field(prob.alg, F, k), X)
                  for k in range(prob.n)], axis=2)
    eta1 = float(np.linalg.norm(sm.jac(X) - J, axis=(1, 2)).max())
    eta2 = float(np.sqrt(np.mean(np.sum(h ** 2, axis=1))))
    return eta0, eta1, eta2


def main():
    rng = np.random.default_rng(90)
    rows = []
    w0 = time.time()
    for key, scale in [("rot_nonpoly", 1.0), ("pendulum", 1.5)]:
        sm = SMOOTH_SYSTEMS[key]
        Xr = rng.uniform(-scale, scale, size=(200_000, 2))
        rms = float(np.sqrt(np.mean(np.sum(sm.rhs(Xr) ** 2, axis=1))))
        for deg in DEGREES:
            prob = DefectProblem.build(2, deg, "affine", "box", scale=scale)
            big = DefectProblem.build(2, deg + 4, "affine", "box", scale=scale)
            X = rng.uniform(-scale, scale, size=(N, 2))
            Y = sm.rhs(X) + SIGMA_REL * rms * rng.normal(size=X.shape)
            fit = fit_ols(prob, X, Y)
            _, Theta = prob.spectrum(fit.beta)
            theta = Theta[:, 0]
            true_rho, se = mc_defect(sm, prob, theta, rng, n_mc=2_000_000)
            eta0, eta1, eta2 = measure_model_error(prob, fit, sm, rng, scale)
            c_none = certify_direction(prob, theta, fit, ALPHA)
            c_sup = certify_direction(prob, theta, fit, ALPHA, eta=(eta0, eta1))
            c_l2 = certify_direction(prob, theta, fit, ALPHA, eta_l2=(eta2, big))
            rows.append(dict(system=key, degree=deg, Q=prob.Q, N=N,
                             true_defect=true_rho, true_defect_se=se,
                             plug_in=c_none.plug_in,
                             upper_no_model_error=c_none.upper,
                             upper_sup_norm=c_sup.upper,
                             upper_l2=c_l2.upper,
                             eta0=eta0, eta1=eta1, eta2=eta2,
                             rel_l2_error=eta2 / rms))
            print(rows[-1], flush=True)
    save("exp9_modelerror", {"rows": rows, "N": N, "sigma_rel": SIGMA_REL,
                             "degrees": DEGREES})
    print(f"exp9 wall {time.time()-w0:.0f}s")


if __name__ == "__main__":
    main()
