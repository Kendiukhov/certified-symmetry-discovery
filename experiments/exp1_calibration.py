"""Experiment 1 -- calibration and power under noise and finite samples.

Ten benchmark systems (four with a nonzero symmetry algebra, six without) are
observed through the standard noisy-derivative design.  Every method must
answer the same question: which generators may be declared ``delta``-approximate
symmetries?  We record how often that declaration is wrong (false certification)
and how often the true symmetry algebra is recovered.
"""

from __future__ import annotations

import time
from itertools import product

import numpy as np

from common import (ALPHA, DELTA, principal_angle, rms_field, run_cells, save,
                    sup_true_defect)
from hsd import DefectProblem, exact_symmetry_algebra, fit_ols, get_system, simulate_design
from hsd.baselines import (eigengap, rank_test_dimension, split_significance,
                           weyl_certificate)
from hsd.certify import certified_dimension, certify_subspace, refute_all

SYSTEMS = [
    # (name, degree of the model class, target-domain half-width)
    ("hopf", 3, 1.0),
    ("lin_rot2", 1, 1.0),
    ("rigid_sym", 2, 1.0),
    ("sphere_flow3", 3, 1.0),
    ("vanderpol", 3, 1.0),
    ("duffing", 3, 2.0),
    ("lotka_volterra", 2, 1.0),
    ("selkov", 3, 2.0),
    ("rigid_asym", 2, 1.0),
    ("lorenz", 2, 20.0),
]
NS = [100, 200, 400, 800, 1600, 3200]
SIGMAS = [0.01, 0.03, 0.10, 0.30]
TAUS = [0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.40]
TRIALS = 300


def one_cell(args):
    name, degF, scale, N, srel, seed = args
    rng = np.random.default_rng(seed)
    sys_ = get_system(name)
    prob = DefectProblem.build(sys_.n, degF, "affine", "box", scale=scale)
    beta0 = sys_.beta(prob)
    Strue = exact_symmetry_algebra(sys_, prob)
    dstar = Strue.shape[1]
    sigma = srel * rms_field(sys_, np.random.default_rng(0), scale)

    acc = {k: 0.0 for k in
           ["fc_sym", "det_sym", "dim_sym", "fc_gap", "det_gap", "dim_gap",
            "fc_rank", "det_rank", "dim_rank", "fc_weyl", "det_weyl", "dim_weyl",
            "fc_split", "det_split", "fc_symp", "det_symp", "dim_symp",
            "refute", "angle_ok", "u_first"]}
    acc.update({f"fc_tau{t}": 0.0 for t in TAUS})
    acc.update({f"det_tau{t}": 0.0 for t in TAUS})
    acc.update({f"dim_tau{t}": 0.0 for t in TAUS})

    for _ in range(TRIALS):
        X, Y = simulate_design(sys_, rng, N=N, sigma=sigma, sampler="box", scale=scale)
        fit = fit_ols(prob, X, Y)
        rho_hat, Theta = prob.spectrum(fit.beta)

        # ---- fixed-threshold rule, swept over tolerances
        for t in TAUS:
            d = int(np.sum(rho_hat <= t))
            V = Theta[:, :d]
            acc[f"dim_tau{t}"] += d
            acc[f"fc_tau{t}"] += sup_true_defect(prob, V, beta0) > DELTA
            acc[f"det_tau{t}"] += d >= dstar

        # ---- eigengap heuristic
        dg, Vg, _ = eigengap(prob, fit, max_dim=prob.P - 1)
        acc["dim_gap"] += dg
        acc["fc_gap"] += sup_true_defect(prob, Vg, beta0) > DELTA
        acc["det_gap"] += dg >= dstar

        # ---- Wald rank test
        dr = rank_test_dimension(prob, fit, ALPHA)
        acc["dim_rank"] += dr
        acc["fc_rank"] += sup_true_defect(prob, Theta[:, :dr], beta0) > DELTA
        acc["det_rank"] += dr >= dstar

        # ---- Weyl perturbation certificate
        wb = weyl_certificate(prob, fit, ALPHA)
        dw = int(np.sum(wb <= DELTA))
        acc["dim_weyl"] += dw
        acc["fc_weyl"] += sup_true_defect(prob, Theta[:, :dw], beta0) > DELTA
        acc["det_weyl"] += dw >= dstar

        # ---- split significance test (select on half, test on the other half)
        half = N // 2
        fA = fit_ols(prob, X[:half], Y[:half])
        fB = fit_ols(prob, X[half:], Y[half:])
        declared, th, _ = split_significance(prob, fA, fB, ALPHA)
        acc["fc_split"] += declared and sup_true_defect(prob, th[:, None], beta0) > DELTA
        acc["det_split"] += int(declared) >= min(dstar, 1)

        # ---- SymCert (simultaneous)
        dc, Vc = certified_dimension(prob, fit, DELTA, ALPHA)
        acc["dim_sym"] += dc
        acc["fc_sym"] += sup_true_defect(prob, Vc, beta0) > DELTA
        acc["det_sym"] += dc >= dstar
        if dc == dstar and dstar > 0:
            acc["angle_ok"] += principal_angle(Vc, Strue) < 10.0
        acc["u_first"] += certify_subspace(prob, Theta[:, :1], fit, ALPHA).upper

        # ---- SymCert (split + pointwise bound)
        _, ThA = prob.spectrum(fA.beta)
        dsp = 0
        for d in range(1, prob.P + 1):
            if certify_subspace(prob, ThA[:, :d], fB, ALPHA).upper <= DELTA:
                dsp = d
            else:
                break
        acc["dim_symp"] += dsp
        acc["fc_symp"] += sup_true_defect(prob, ThA[:, :dsp], beta0) > DELTA
        acc["det_symp"] += dsp >= dstar

        acc["refute"] += refute_all(prob, fit, ALPHA)

    out = {k: v / TRIALS for k, v in acc.items()}
    out.update(system=name, N=N, sigma_rel=srel, d_true=dstar,
               true_min_defect=float(prob.spectrum(beta0)[0][0]), P=prob.P)
    return out


def main():
    cells = [(nm, dg, sc, N, s, 1000 + 7919 * i)
             for i, ((nm, dg, sc), N, s) in enumerate(product(SYSTEMS, NS, SIGMAS))]
    w0 = time.time()
    rows = run_cells(one_cell, cells, desc="exp1")
    save("exp1_calibration", {"rows": rows, "taus": TAUS, "trials": TRIALS,
                              "Ns": NS, "sigmas": SIGMAS})
    print(f"exp1 wall {time.time()-w0:.0f}s")


if __name__ == "__main__":
    main()
