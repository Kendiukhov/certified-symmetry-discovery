"""Experiment 12 -- a case study on real measurements.

The Hudson's Bay Company lynx and hare pelt records for 1900--1920 are the
canonical real data set for exactly the kind of model discovery this paper is
about: twenty-one annual observations of a two-dimensional predator-prey system.
We run the whole pipeline on them and report what an honest analysis can and
cannot say.

The regression assumptions are only approximately met here -- derivatives are
differenced from noisy annual counts, so the regressors themselves carry error
(see the limitations).  We therefore present this as a case study rather than as
a validity claim, and it is instructive precisely because the honest answer is a
refusal.

Data source: pelt counts (thousands) tabulated by Howard (2009) and distributed
with the Stan example models, https://github.com/stan-dev/example-models
(knitr/lotka-volterra/hudson-bay-lynx-hare.csv), cross-checked against
independent reproductions of the same table.
"""

from __future__ import annotations

import time

import numpy as np

from common import ALPHA, DELTA, save
from hsd import DefectProblem, fit_ols
from hsd.certify import certified_dimension, certify_direction, refute_all
from hsd.coverage import extrapolation_factor

YEAR = np.arange(1900, 1921)
HARE = np.array([30.0, 47.2, 70.2, 77.4, 36.3, 20.6, 18.1, 21.4, 22.0, 25.4, 27.1,
                 40.3, 57.0, 76.6, 52.3, 19.5, 11.2, 7.6, 14.6, 16.2, 24.7])
LYNX = np.array([4.0, 6.1, 9.8, 35.2, 59.4, 41.7, 19.0, 13.0, 8.3, 9.1, 7.4,
                 8.0, 12.3, 19.5, 45.7, 51.1, 29.7, 15.8, 9.7, 10.1, 8.6])


def main():
    w0 = time.time()
    Z = np.stack([HARE, LYNX], axis=1)
    scale = float(np.abs(Z).max())
    Zs = Z / scale                              # work in units of the largest count
    # central differences for the interior years (dt = 1 year)
    X = Zs[1:-1]
    Y = (Zs[2:] - Zs[:-2]) / 2.0
    prob = DefectProblem.build(2, 2, "affine", "box", scale=1.0)
    fit = fit_ols(prob, X, Y, robust=True)
    rho_hat, Theta = prob.spectrum(fit.beta)
    cov = extrapolation_factor(prob, X)
    pe = prob.empirical(X)
    rho_emp, Theta_emp = pe.spectrum(fit.beta)

    rows = []
    for k in range(prob.P):
        c = certify_direction(prob, Theta[:, k], fit, ALPHA)
        rows.append(dict(index=k, plug_in=float(rho_hat[k]),
                         upper=float(c.upper), lower=float(c.lower),
                         generator={n: round(float(v), 3)
                                    for n, v in zip(prob.gen_names, Theta[:, k])
                                    if abs(v) > 1e-3}))
    naive_dim_target = int(np.sum(rho_hat <= DELTA))
    naive_dim_data = int(np.sum(rho_emp <= DELTA))
    cert_dim, _ = certified_dimension(prob, fit, DELTA, ALPHA)
    out = dict(n_points=int(X.shape[0]), scale=scale,
               design_condition=float(fit.design_cond),
               sigma_hat=float(np.sqrt(fit.sigma2)),
               kappa_bound=float(cov["bound"]),
               plug_in_spectrum_target=[float(v) for v in rho_hat],
               plug_in_spectrum_data=[float(v) for v in rho_emp],
               naive_dim_target=naive_dim_target, naive_dim_data=naive_dim_data,
               certified_dim=cert_dim,
               smallest_certified_tolerance=float(min(r["upper"] for r in rows)),
               refutation_lower_bound=float(refute_all(prob, fit, ALPHA)),
               directions=rows)
    save("exp12_realdata", out)
    for k, v in out.items():
        if k != "directions":
            print(f"  {k}: {v}")
    print(f"exp12 wall {time.time()-w0:.1f}s")


if __name__ == "__main__":
    main()
