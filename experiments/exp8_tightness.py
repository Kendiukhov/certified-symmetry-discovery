"""Experiment 8 -- is the guarantee tight, and what does a finite-sample guarantee cost?

A certificate that never certifies anything would also never certify anything
false.  Five measurements.

**(a) Coverage.**  How often the coefficient region and the certificate
interval contain the truth, as a function of the nominal level, on systems with
and without symmetry.

**(b) Width.**  The sharp certificate (exact image of the ratio over the
region), the separate bound (numerator and denominator bounded independently)
and a calibration-targeting parametric bootstrap (the ``1 - alpha`` quantile of
``rho(theta; beta*)`` with ``beta* ~ N(beta_hat, Sigma_hat)``), on the fixed
rotation generator of the broken-Hopf family.  Reported as the bound itself and
as its excess over the plug-in defect.

**(c) Interval coverage near a symmetry.**  The bootstrap targets nominal
coverage only where the defect is a smooth function of the coefficients; at
and near a true symmetry it is not (the defect is a norm).  We measure the
coverage of the bootstrap percentile interval ``[q_alpha, q_{1-alpha}]``
(nominal ``1 - 2 alpha``) and of the certificate interval as the true defect
approaches zero.

**(d) The tolerance boundary.**  A system whose true rotation defect is just
above the tolerance, ``(1 + tau) delta``, over a range of sample sizes, for the
generator fixed in advance and for the generator selected from the data as the
one with the smallest plug-in defect.  Any procedure with a valid upper bound
certifies such a system with probability at most ``alpha``; we record how
often each procedure does.

**(e) A generator selected from the data.**  On systems without symmetry, the
generator with the smallest plug-in defect is selected on the same data and its
bound is computed as if it had been fixed in advance.  Selection biases its
plug-in defect downward (the winner's curse), which a bootstrap centred at the
plug-in estimate does not see.  We record how often each upper bound contains
the true defect of the selected generator, at small sample sizes and high
noise, where the selection effect is largest.
"""

from __future__ import annotations

import json
import sys
import time
from itertools import product

import numpy as np
from scipy.optimize import brentq

from common import DELTA, RESULTS, rms_field, run_cells, save, sup_true_defect
from common import trials
from hsd import DefectProblem, fit_ols, get_system, simulate_design
from hsd.certify import certify_direction, certify_direction_bootstrap
from hsd.estimation import region_covers
from hsd.systems import broken_hopf

TRIALS = trials(80)
WIDTH_TRIALS = trials(200)
BOUNDARY_TRIALS = trials(300)
SELECTED_TRIALS = trials(300)
ALPHAS = [0.5, 0.3, 0.2, 0.1, 0.05, 0.01]
SIGMA_REL = 0.1


def _rot(prob):
    r = np.zeros(prob.P)
    r[1], r[2] = 1.0, -1.0
    return r / np.linalg.norm(r)


def _eps_for_defect(prob, target):
    """Symmetry-breaking size whose rotation defect equals ``target``."""
    if target == 0:
        return 0.0
    rot = _rot(prob)

    def f(e):
        return prob.rho(rot, broken_hopf(float(e)).beta(prob)) - target
    return brentq(f, 1e-10, 5.0, xtol=1e-14)


def coverage_cell(args):
    """(a) region and interval coverage at each nominal level."""
    name, degF, scale, N, srel, seed = args
    rng = np.random.default_rng(seed)
    sys_ = get_system(name)
    prob = DefectProblem.build(sys_.n, degF, "affine", "box", scale=scale)
    beta0 = sys_.beta(prob)
    sigma = srel * rms_field(sys_, np.random.default_rng(0), scale)
    hits = {a: 0 for a in ALPHAS}
    reg = {a: 0 for a in ALPHAS}
    for _ in range(TRIALS):
        X, Y = simulate_design(sys_, rng, N=N, sigma=sigma, sampler="box", scale=scale)
        fit = fit_ols(prob, X, Y)
        _, Theta = prob.spectrum(fit.beta)
        truths = [prob.rho(Theta[:, k], beta0) for k in range(prob.P)]
        for a in ALPHAS:
            reg[a] += region_covers(fit, beta0, a)
            ok = True
            for k in range(prob.P):
                c = certify_direction(prob, Theta[:, k], fit, a)
                if not (c.lower - 1e-12 <= truths[k] <= c.upper + 1e-12):
                    ok = False
                    break
            hits[a] += ok
    return dict(system=name, N=N, sigma_rel=srel,
                coverage={str(a): hits[a] / TRIALS for a in ALPHAS},
                region_coverage={str(a): reg[a] / TRIALS for a in ALPHAS})


def width_cell(args):
    """(b) bound and excess over the plug-in, three constructions."""
    target, N, alpha, seed = args
    rng = np.random.default_rng(seed)
    prob = DefectProblem.build(2, 3, "affine", "box", scale=1.0)
    rot = _rot(prob)
    sys_ = broken_hopf(_eps_for_defect(prob, target))
    sigma = SIGMA_REL * rms_field(sys_, np.random.default_rng(0), 1.0)
    rec = {k: [] for k in ("sharp", "separate", "boot", "plug")}
    for _ in range(WIDTH_TRIALS):
        X, Y = simulate_design(sys_, rng, N=N, sigma=sigma, sampler="box", scale=1.0)
        fit = fit_ols(prob, X, Y)
        cs = certify_direction(prob, rot, fit, alpha)
        rec["sharp"].append(cs.upper)
        rec["plug"].append(cs.plug_in)
        rec["separate"].append(certify_direction(prob, rot, fit, alpha, ratio="separate").upper)
        rec["boot"].append(certify_direction_bootstrap(prob, rot, fit, alpha, rng=rng).upper)
    r = {k: np.asarray(v) for k, v in rec.items()}
    out = dict(true_defect=target, N=N, alpha=alpha)
    for k in ("sharp", "separate", "boot"):
        out[f"upper_{k}"] = float(np.median(r[k]))
        out[f"excess_{k}"] = float(np.median(r[k] - r["plug"]))
    out["plug"] = float(np.median(r["plug"]))
    return out


def small_defect_cell(args):
    """(c) two-sided interval coverage as the true defect approaches zero."""
    target, N, alpha, seed = args
    rng = np.random.default_rng(seed)
    prob = DefectProblem.build(2, 3, "affine", "box", scale=1.0)
    rot = _rot(prob)
    sys_ = broken_hopf(_eps_for_defect(prob, target))
    true = prob.rho(rot, sys_.beta(prob))
    sigma = SIGMA_REL * rms_field(sys_, np.random.default_rng(0), 1.0)
    cov_cert = cov_boot = 0
    for _ in range(WIDTH_TRIALS):
        X, Y = simulate_design(sys_, rng, N=N, sigma=sigma, sampler="box", scale=1.0)
        fit = fit_ols(prob, X, Y)
        c = certify_direction(prob, rot, fit, alpha)
        b = certify_direction_bootstrap(prob, rot, fit, alpha, rng=rng)
        cov_cert += c.lower - 1e-12 <= true <= c.upper + 1e-12
        cov_boot += b.lower <= true <= b.upper
    return dict(true_defect=float(true), N=N, alpha=alpha,
                coverage_cert=cov_cert / WIDTH_TRIALS,
                coverage_boot=cov_boot / WIDTH_TRIALS,
                nominal_cert=1 - alpha, nominal_boot=1 - 2 * alpha,
                count_boot=int(cov_boot), trials=WIDTH_TRIALS)


def boundary_cell(args):
    """(d) false certification with the truth just past the tolerance."""
    tau, N, alpha, seed = args
    rng = np.random.default_rng(seed)
    prob = DefectProblem.build(2, 3, "affine", "box", scale=1.0)
    rot = _rot(prob)
    eps = _eps_for_defect(prob, DELTA * (1.0 + tau))
    sys_ = broken_hopf(eps)
    beta0 = sys_.beta(prob)
    sigma = SIGMA_REL * rms_field(sys_, np.random.default_rng(0), 1.0)
    keys = ("fix_sharp", "fix_sep", "fix_boot", "sel_sharp", "sel_sep", "sel_boot")
    fc = {k: 0 for k in keys}
    med = {k: [] for k in ("fix_sharp", "fix_boot")}
    T = BOUNDARY_TRIALS
    for _ in range(T):
        X, Y = simulate_design(sys_, rng, N=N, sigma=sigma, sampler="box", scale=1.0)
        fit = fit_ols(prob, X, Y)
        _, Theta = prob.spectrum(fit.beta)
        sel = Theta[:, 0] / np.linalg.norm(Theta[:, 0])
        sel_bad = prob.rho(sel, beta0) > DELTA
        for tag, th, bad in (("fix", rot, True), ("sel", sel, sel_bad)):
            us = certify_direction(prob, th, fit, alpha).upper
            ue = certify_direction(prob, th, fit, alpha, ratio="separate").upper
            ub = certify_direction_bootstrap(prob, th, fit, alpha, rng=rng).upper
            fc[f"{tag}_sharp"] += bad and us <= DELTA
            fc[f"{tag}_sep"] += bad and ue <= DELTA
            fc[f"{tag}_boot"] += bad and ub <= DELTA
            if tag == "fix":
                med["fix_sharp"].append(us)
                med["fix_boot"].append(ub)
    out = dict(tau=tau, N=N, alpha=alpha, eps=float(eps), trials=T,
               true_defect=float(prob.rho(rot, beta0)),
               true_min_defect=float(prob.spectrum(beta0)[0][0]),
               median_upper_sharp=float(np.median(med["fix_sharp"])),
               median_upper_boot=float(np.median(med["fix_boot"])))
    out.update({f"fc_{k}": fc[k] / T for k in keys})
    out.update({f"count_{k}": int(fc[k]) for k in keys})
    return out


def selected_cell(args):
    """(e) upper-bound coverage for the minimal-plug-in-defect generator."""
    name, degF, N, srel, alpha, seed = args
    rng = np.random.default_rng(seed)
    sys_ = get_system(name)
    prob = DefectProblem.build(sys_.n, degF, "affine", "box", scale=1.0)
    beta0 = sys_.beta(prob)
    sigma = srel * rms_field(sys_, np.random.default_rng(0), 1.0)
    T = SELECTED_TRIALS
    cov = {"sharp": 0, "boot": 0}
    gap = []
    for _ in range(T):
        X, Y = simulate_design(sys_, rng, N=N, sigma=sigma, sampler="box", scale=1.0)
        fit = fit_ols(prob, X, Y)
        _, Theta = prob.spectrum(fit.beta)
        th = Theta[:, 0] / np.linalg.norm(Theta[:, 0])
        true = prob.rho(th, beta0)
        c = certify_direction(prob, th, fit, alpha)
        b = certify_direction_bootstrap(prob, th, fit, alpha, rng=rng)
        cov["sharp"] += true <= c.upper + 1e-12
        cov["boot"] += true <= b.upper
        gap.append(true - c.plug_in)
    return dict(system=name, degF=degF, N=N, sigma_rel=srel, alpha=alpha, trials=T,
                true_min_defect=float(prob.spectrum(beta0)[0][0]),
                coverage_upper_sharp=cov["sharp"] / T, coverage_upper_boot=cov["boot"] / T,
                count_boot=int(cov["boot"]), mean_true_minus_plug=float(np.mean(gap)),
                nominal=1 - alpha)


STAGES = ("coverage", "width", "small", "boundary", "selected")


def main(part: str = "all"):
    """``part`` selects one stage so the stages can be run separately; the
    partial results are merged by the ``combine`` stage."""
    w0 = time.time()
    part_dir = RESULTS / "partial"
    part_dir.mkdir(exist_ok=True)
    cells = {
        "coverage": (coverage_cell, [(nm, dg, sc, N, s, 8000 + 977 * i)
                                      for i, ((nm, dg, sc), N, s) in enumerate(product(
                                          [("hopf", 3, 1.0), ("vanderpol", 3, 1.0),
                                           ("rigid_sym", 2, 1.0)], [400], [0.03, 0.1]))]),
        "width": (width_cell, [(t, N, a, 8300 + 431 * i)
                               for i, (t, N, a) in enumerate(product(
                                   [0.0, DELTA * 1.02, DELTA * 2.0],
                                   [400, 1600, 6400, 25600], [0.05, 0.2]))]),
        "small": (small_defect_cell, [(t, N, a, 8400 + 523 * i)
                                      for i, (t, N, a) in enumerate(product(
                                          [0.0, 0.002, 0.005, 0.01, 0.02, 0.05],
                                          [400, 3200], [0.05, 0.2]))]),
        "boundary": (boundary_cell, [(t, N, a, 8500 + 613 * i)
                                     for i, (t, N, a) in enumerate(product(
                                         [0.02, 0.1], [800, 3200, 12800, 51200],
                                         [0.5, 0.2, 0.05]))]),
        "selected": (selected_cell, [(nm, dg, N, s, a, 8600 + 389 * i)
                                     for i, ((nm, dg), N, s, a) in enumerate(product(
                                         [("vanderpol", 3), ("duffing", 3),
                                          ("lotka_volterra", 2), ("selkov", 3)],
                                         [50, 100, 200, 400], [0.2, 0.5],
                                         [0.2, 0.05]))]),
    }
    if part == "combine" or part == "all":
        if part == "all":
            for st in STAGES:
                fn, cl = cells[st]
                (part_dir / f"exp8_{st}.json").write_text(
                    json.dumps(run_cells(fn, cl, desc=f"exp8{st}")))
        payload = {st: json.loads((part_dir / f"exp8_{st}.json").read_text())
                   for st in STAGES}
        payload.update(trials=TRIALS, width_trials=WIDTH_TRIALS,
                       boundary_trials=BOUNDARY_TRIALS, selected_trials=SELECTED_TRIALS,
                       alphas=ALPHAS,
                       sigma_rel=SIGMA_REL)
        save("exp8_tightness", payload)
    else:
        fn, cl = cells[part]
        (part_dir / f"exp8_{part}.json").write_text(
            json.dumps(run_cells(fn, cl, desc=f"exp8{part}")))
    print(f"exp8 ({part}) wall {time.time()-w0:.0f}s")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "all")
