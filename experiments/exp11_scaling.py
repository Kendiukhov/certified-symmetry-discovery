"""Experiment 11 -- how the certificate behaves as the problem grows.

The state dimension enters twice: through the number of model coefficients ``Q``,
which sets the radius of the confidence ellipsoid, and through the number of
candidate generators ``P``, which sets how many directions are searched.  Only
the first should matter.  We use linear systems, whose exact symmetry algebra is
the commutant of the coefficient matrix and whose dimension we therefore know in
closed form for every ``n``, and we record wall-clock cost alongside statistical
behaviour so that the computational claim is checkable too.
"""

from __future__ import annotations

import time
from itertools import product

import numpy as np

from common import ALPHA, DELTA, rms_field, run_cells, save, sup_true_defect
from hsd import DefectProblem, exact_symmetry_algebra, fit_ols, simulate_design
from hsd.certify import certified_dimension, certify_direction
from hsd.systems import System

TRIALS = 200
DIMS = [2, 3, 4, 5, 6]
NS = [200, 800, 3200, 12800]
SIGMA = 0.05


def rotation_block_system(n: int) -> System:
    """A linear system built from rotation blocks (and one real eigenvalue when
    ``n`` is odd), so that the symmetry algebra is nontrivial and its dimension
    grows with ``n`` in a way we can verify exactly."""
    from fractions import Fraction
    terms = {}
    for b in range(n // 2):
        i, j = 2 * b, 2 * b + 1
        w = Fraction(b + 1)             # distinct frequencies -> known commutant
        e_i, e_j = [0] * n, [0] * n
        e_i[i], e_j[j] = 1, 1
        terms[(i, tuple(e_j))] = -w
        terms[(j, tuple(e_i))] = w
    if n % 2 == 1:
        e = [0] * n
        e[n - 1] = 1
        terms[(n - 1, tuple(e))] = Fraction(-1)
    return System(f"rot_blocks{n}", n, 1, terms,
                  "Block-rotation linear system used for the dimension sweep.",
                  tags=("linear", "symmetric"))


def one_cell(args):
    n, N, seed = args
    rng = np.random.default_rng(seed)
    sys_ = rotation_block_system(n)
    t_build = time.process_time()
    prob = DefectProblem.build(n, 1, "linear", "box", scale=1.0)
    t_build = time.process_time() - t_build
    beta0 = sys_.beta(prob)
    dstar = exact_symmetry_algebra(sys_, prob).shape[1]
    sigma = SIGMA * rms_field(sys_, np.random.default_rng(0), 1.0)
    det = fc = 0
    uppers, t_cert = [], 0.0
    for _ in range(TRIALS):
        X, Y = simulate_design(sys_, rng, N=N, sigma=sigma, sampler="box", scale=1.0)
        fit = fit_ols(prob, X, Y)
        # CPU time, not wall time: the machine may be shared, and the cost we
        # want to report is the algorithm's, not the queue's.
        t0 = time.process_time()
        dc, Vc = certified_dimension(prob, fit, DELTA, ALPHA)
        _, Theta = prob.spectrum(fit.beta)
        uppers.append(certify_direction(prob, Theta[:, 0], fit, ALPHA).upper)
        t_cert += time.process_time() - t0
        det += dc >= dstar
        fc += sup_true_defect(prob, Vc, beta0) > DELTA
    return dict(n=n, N=N, P=prob.P, Q=prob.Q, d_true=dstar,
                detect=det / TRIALS, fc=fc / TRIALS,
                upper_med=float(np.median(uppers)),
                cpu_seconds_per_certificate=t_cert / TRIALS,
                seconds_to_build=t_build)


def main():
    cells = [(n, N, 11_000 + 617 * i)
             for i, (n, N) in enumerate(product(DIMS, NS))]
    w0 = time.time()
    rows = run_cells(one_cell, cells, desc="exp11")
    save("exp11_scaling", {"rows": rows, "trials": TRIALS, "sigma_rel": SIGMA})
    print(f"exp11 wall {time.time()-w0:.0f}s")


if __name__ == "__main__":
    main()
