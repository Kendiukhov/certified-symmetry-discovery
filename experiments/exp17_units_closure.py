"""Experiment 17 -- units of the state, and closure of certified subspaces.

**(a) Units.**  The relative defect is free of the scale of the vector field,
of the generator and of the candidate basis, but with the Euclidean output
metric it is not free of the *units of the state*: rescaling one coordinate
changes it.  We take the broken-Hopf system (rotation defect 0.064 in its own
coordinates) and Lotka--Volterra, express them in coordinates ``x = S z`` with
``S = diag(c, 1)`` (one variable measured in different units), and compute the
defect of the transported generators -- and the smallest defect over the whole
candidate class -- with (i) the Euclidean metric in the new coordinates, (ii)
the transported metric ``W = S'S`` and (iii) the nondimensionalising metric
``W = diag(1/s_i^2)`` with ``s_i`` the root-mean-square size of coordinate
``i`` over the target domain.  A simulation then certifies in the new units
with the Euclidean and the nondimensional metric and judges every
certification by the unit-free defect of the physical system.

**(b) Closure.**  For the systems whose symmetry algebra has dimension at
least two -- ``so(3)`` of the radial sphere flow (non-abelian), the symmetric
top and the planar rotation-invariant linear system (both abelian) -- we
measure how far each certified subspace of dimension at least two is from
closure under the bracket (``hsd.closure.closure_report``) and whether the
commutators of its generators are themselves certified.
"""

from __future__ import annotations

import time
from fractions import Fraction
from itertools import product

import numpy as np

from common import ALPHA, DELTA, rms_field, run_cells, save, sup_true_defect
from common import trials
from hsd import DefectProblem, fit_ols, get_system, simulate_design
from hsd.certify import certified_dimension, certify_direction
from hsd.closure import closure_report
from hsd.systems import System, broken_hopf

UNITS = [Fraction(1, 100), Fraction(1, 10), Fraction(1, 2), Fraction(1), Fraction(2),
         Fraction(10), Fraction(100)]
SIM_UNITS = [Fraction(1, 100), Fraction(1), Fraction(100)]
SIM_SYSTEMS = {"broken_hopf": (lambda: broken_hopf(0.1), 3),
               "lotka_volterra": (lambda: get_system("lotka_volterra"), 2)}
TRIALS = trials(300)
def rescale(sys_: System, s) -> System:
    """``F~(z) = S^{-1} F(S z)`` for ``S = diag(s)`` (exact rationals)."""
    terms = {}
    for (i, a), c in sys_.terms.items():
        f = Fraction(1)
        for sj, aj in zip(s, a):
            f *= Fraction(sj) ** aj
        terms[(i, a)] = c * f / Fraction(s[i])
    return System(sys_.name + "_units", sys_.n, sys_.deg, terms, "rescaled")


def transport_generator(theta, s, n):
    """Affine generator ``xi(x) = M x + b`` -> ``xi~(z) = S^-1 M S z + S^-1 b``."""
    M, b = theta[: n * n].reshape(n, n), theta[n * n:]
    S = np.diag([float(v) for v in s])
    Si = np.linalg.inv(S)
    return np.concatenate([(Si @ M @ S).ravel(), Si @ b])


def units_table():
    rows = []
    for name, sys_, degF in [("broken_hopf", broken_hopf(0.1), 3),
                             ("lotka_volterra", get_system("lotka_volterra"), 2)]:
        base = DefectProblem.build(2, degF, "affine", "box", scale=1.0)
        b0 = sys_.beta(base)
        rho, Theta = base.spectrum(b0)
        theta = Theta[:, 0] / np.linalg.norm(Theta[:, 0])
        if name == "broken_hopf":
            theta = np.zeros(base.P)
            theta[1], theta[2] = 1.0, -1.0
        ref = base.rho(theta, b0)
        for c in UNITS:
            s = (c, Fraction(1))
            sv = np.array([float(v) for v in s])
            zsys = rescale(sys_, s)
            half = 1.0 / sv                                   # target box in z units
            rms_coord = half / np.sqrt(3.0)                    # uniform on [-h, h]
            pe = DefectProblem.build(2, degF, "affine", "box", scale=half)
            pw = DefectProblem.build(2, degF, "affine", "box", scale=half,
                                     metric=np.diag(sv ** 2))
            pn = DefectProblem.build(2, degF, "affine", "box", scale=half, metric=rms_coord)
            bz = zsys.beta(pe)
            tz = transport_generator(theta, s, 2)
            rows.append(dict(system=name, unit_factor=float(c), defect_reference=float(ref),
                             defect_euclidean=float(pe.rho(tz, bz)),
                             defect_transported=float(pw.rho(tz, bz)),
                             defect_nondimensional=float(pn.rho(tz, bz)),
                             min_defect_euclidean=float(pe.spectrum(bz)[0][0]),
                             min_defect_nondimensional=float(pn.spectrum(bz)[0][0])))
    return rows


def units_sim_cell(args):
    name, c, metric_kind, seed = args
    rng = np.random.default_rng(seed)
    s = (c, Fraction(1))
    sv = np.array([float(v) for v in s])
    make, degF = SIM_SYSTEMS[name]
    sys_ = make()
    zsys = rescale(sys_, s)
    half = 1.0 / sv
    metric = None if metric_kind == "euclidean" else half / np.sqrt(3.0)
    prob = DefectProblem.build(2, degF, "affine", "box", scale=half, metric=metric)
    judge = DefectProblem.build(2, degF, "affine", "box", scale=half,
                                metric=half / np.sqrt(3.0))
    bz = zsys.beta(prob)
    sigma = 0.05 * rms_field(zsys, np.random.default_rng(0), half)
    cert = fc = 0
    for _ in range(TRIALS):
        X, Y = simulate_design(zsys, rng, N=12800, sigma=sigma, sampler="box", scale=half)
        fit = fit_ols(prob, X, Y)
        dc, Vc = certified_dimension(prob, fit, DELTA, ALPHA)
        cert += dc > 0
        # Judged by the unit-free (nondimensional) defect of the physical system.
        fc += dc > 0 and sup_true_defect(judge, Vc, bz) > DELTA
    return dict(system=name, unit_factor=float(c), metric=metric_kind,
                certify_rate=cert / TRIALS,
                fc=fc / TRIALS, count_fc=int(fc), trials=TRIALS)


CLOSURE_SYSTEMS = [("sphere_flow3", 3, 1.0), ("rigid_sym", 2, 1.0), ("lin_rot2", 1, 1.0)]


def closure_cell(args):
    name, degF, scale, N, srel, seed = args
    rng = np.random.default_rng(seed)
    sys_ = get_system(name)
    prob = DefectProblem.build(sys_.n, degF, "affine", "box", scale=scale)
    sigma = srel * rms_field(sys_, np.random.default_rng(0), scale)
    closures, comm_ok, dims = [], 0, []
    n2 = n_with_comm = 0
    for _ in range(max(1, TRIALS // 3)):
        X, Y = simulate_design(sys_, rng, N=N, sigma=sigma, sampler="box", scale=scale)
        fit = fit_ols(prob, X, Y)
        dc, Vc = certified_dimension(prob, fit, DELTA, ALPHA)
        dims.append(dc)
        if dc >= 2:
            rep = closure_report(prob, Vc, fit, ALPHA)
            n2 += 1
            closures.append(rep["max_closure"])
            if rep["n_commutators"]:
                n_with_comm += 1
                comm_ok += rep["commutator_upper"] <= DELTA
    return dict(system=name, N=N, sigma_rel=srel, runs=max(1, TRIALS // 3),
                runs_dim_ge2=n2, mean_dim=float(np.mean(dims)),
                median_closure=float(np.median(closures)) if closures else None,
                max_closure=float(np.max(closures)) if closures else None,
                runs_with_commutators=n_with_comm,
                commutators_certified=(comm_ok / n_with_comm) if n_with_comm else None)


def main():
    w0 = time.time()
    table = units_table()
    sims = run_cells(units_sim_cell, [(nm, c, m, 17000 + 97 * i)
                                      for i, (nm, c, m) in enumerate(product(
                                          list(SIM_SYSTEMS), SIM_UNITS,
                                          ["euclidean", "nondimensional"]))], desc="exp17a")
    clos = run_cells(closure_cell, [(nm, dg, sc, N, s, 17500 + 131 * i)
                                    for i, ((nm, dg, sc), N, s) in enumerate(
                                        product(CLOSURE_SYSTEMS, [800, 3200, 12800],
                                                [0.01, 0.03]))], desc="exp17b")
    save("exp17_units_closure", {"units": table, "units_sim": sims, "closure": clos,
                                 "trials": TRIALS})
    print(f"exp17 wall {time.time()-w0:.0f}s")


if __name__ == "__main__":
    main()
