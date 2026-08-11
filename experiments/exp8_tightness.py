"""Experiment 8 -- is the guarantee tight, or merely vacuous?

A certificate that never certifies anything would also never certify anything
false.  Two measurements settle the question.

**(a) Interval coverage.**  The certificate returns an interval for the true
defect.  We record how often the interval actually contains the truth, as a
function of the nominal level, on systems with and without symmetry.  A
procedure that is exactly calibrated tracks the diagonal; ours is conservative
by a measurable and reported amount.

**(b) The boundary.**  False certification is only possible for generators whose
true defect sits just above the tolerance, and only when the sample size puts
the certificate width right at the tolerance.  We construct exactly that
worst case -- a one-parameter family whose defect is tuned to
``(1 + tau) delta`` and a sample size at which the median certified bound
equals ``delta`` -- and measure the false-certification rate there.
"""

from __future__ import annotations

import json
import sys
import time
from itertools import product

import numpy as np
from scipy.optimize import brentq

from common import DELTA, RESULTS, rms_field, run_cells, save, sup_true_defect
from hsd import DefectProblem, fit_ols, get_system, simulate_design
from hsd.certify import (certified_dimension, certify_direction,
                         certify_direction_bootstrap)
from hsd.systems import broken_hopf

TRIALS = 80
BOUNDARY_TRIALS = 300
ALPHAS = [0.5, 0.3, 0.2, 0.1, 0.05, 0.01]


def _rot(prob):
    r = np.zeros(prob.P)
    r[1], r[2] = 1.0, -1.0
    return r / np.linalg.norm(r)


def coverage_cell(args):
    """(a) empirical coverage of the certificate interval at each nominal level."""
    name, degF, scale, N, srel, seed = args
    rng = np.random.default_rng(seed)
    sys_ = get_system(name)
    prob = DefectProblem.build(sys_.n, degF, "affine", "box", scale=scale)
    beta0 = sys_.beta(prob)
    sigma = srel * rms_field(sys_, np.random.default_rng(0), scale)
    hits = {a: 0 for a in ALPHAS}
    for _ in range(TRIALS):
        X, Y = simulate_design(sys_, rng, N=N, sigma=sigma, sampler="box", scale=scale)
        fit = fit_ols(prob, X, Y)
        _, Theta = prob.spectrum(fit.beta)
        truths = [prob.rho(Theta[:, k], beta0) for k in range(prob.P)]
        for a in ALPHAS:
            ok = True
            for k in range(prob.P):
                c = certify_direction(prob, Theta[:, k], fit, a)
                if not (c.lower - 1e-12 <= truths[k] <= c.upper + 1e-12):
                    ok = False
                    break
            hits[a] += ok
    return dict(system=name, N=N, sigma_rel=srel,
                coverage={str(a): hits[a] / TRIALS for a in ALPHAS})


def _eps_for_defect(prob, target):
    """Symmetry-breaking size whose rotation defect equals ``target``."""
    rot = _rot(prob)

    def f(e):
        return prob.rho(rot, broken_hopf(float(e)).beta(prob)) - target
    return brentq(f, 1e-8, 5.0, xtol=1e-12)


def _N_for_width(prob, sys_, sigma, target_width, alpha):
    """Sample size at which the median certified bound is about ``target_width``.

    The bound scales as ``N^{-1/2}``, so one calibration run fixes the constant.
    """
    rng = np.random.default_rng(0)
    rot = _rot(prob)
    N0 = 4000
    ws = []
    for _ in range(15):
        X, Y = simulate_design(sys_, rng, N=N0, sigma=sigma, sampler="box", scale=1.0)
        ws.append(certify_direction(prob, rot, fit_ols(prob, X, Y), alpha).upper)
    w0 = float(np.median(ws))
    return int(np.clip(round(N0 * (w0 / target_width) ** 2), 50, 400_000))


def boundary_cell(args):
    """(b) false certification with the truth pushed just past the tolerance."""
    tau, alpha, srel, seed = args
    rng = np.random.default_rng(seed)
    prob = DefectProblem.build(2, 3, "affine", "box", scale=1.0)
    rot = _rot(prob)
    eps = _eps_for_defect(prob, DELTA * (1.0 + tau))
    sys_ = broken_hopf(eps)
    beta0 = sys_.beta(prob)
    sigma = srel * rms_field(sys_, np.random.default_rng(0), 1.0)
    N = _N_for_width(prob, sys_, sigma, DELTA, alpha)
    fc_dir = fc_proc = fc_boot = 0
    widths, widths_boot = [], []
    T = BOUNDARY_TRIALS
    for _ in range(T):
        X, Y = simulate_design(sys_, rng, N=N, sigma=sigma, sampler="box", scale=1.0)
        fit = fit_ols(prob, X, Y)
        c = certify_direction(prob, rot, fit, alpha)
        cb = certify_direction_bootstrap(prob, rot, fit, alpha, rng=rng)
        fc_dir += c.upper <= DELTA
        fc_boot += cb.upper <= DELTA
        widths.append(c.upper)
        widths_boot.append(cb.upper)
        dc, Vc = certified_dimension(prob, fit, DELTA, alpha)
        fc_proc += (dc > 0) and (sup_true_defect(prob, Vc, beta0) > DELTA)
    return dict(tau=tau, alpha=alpha, sigma_rel=srel, eps=float(eps), N=N,
                true_defect=float(prob.rho(rot, beta0)),
                fc_direction=fc_dir / T, fc_procedure=fc_proc / T,
                fc_bootstrap=fc_boot / T,
                width_exact=float(np.median(widths)),
                width_bootstrap=float(np.median(widths_boot)),
                ratio_to_alpha=(fc_dir / T) / alpha)


def main(part: str = "all"):
    """``part`` selects a stage so the two halves can be run separately; the
    partial results are merged by the ``combine`` stage."""
    w0 = time.time()
    part_dir = RESULTS / "partial"
    part_dir.mkdir(exist_ok=True)
    if part == "combine":
        cov = json.loads((part_dir / "exp8_coverage.json").read_text())
        bnd = json.loads((part_dir / "exp8_boundary.json").read_text())
        save("exp8_tightness", {"coverage": cov, "boundary": bnd,
                                "trials": TRIALS,
                                "boundary_trials": BOUNDARY_TRIALS,
                                "alphas": ALPHAS})
        return
    ccells = [(nm, dg, sc, N, s, 8000 + 977 * i)
              for i, ((nm, dg, sc), N, s) in enumerate(product(
                  [("hopf", 3, 1.0), ("vanderpol", 3, 1.0), ("rigid_sym", 2, 1.0)],
                  [400], [0.03, 0.1]))]
    bcells = [(t, a, 0.1, 8500 + 613 * i)
              for i, (t, a) in enumerate(product([0.02, 0.1, 0.5],
                                                 [0.5, 0.2, 0.05]))]
    if part in ("all", "coverage"):
        cov = run_cells(coverage_cell, ccells, desc="exp8a")
        (part_dir / "exp8_coverage.json").write_text(json.dumps(cov))
    if part in ("all", "boundary"):
        bnd = run_cells(boundary_cell, bcells, desc="exp8b")
        (part_dir / "exp8_boundary.json").write_text(json.dumps(bnd))
    if part == "all":
        save("exp8_tightness", {"coverage": cov, "boundary": bnd,
                                "trials": TRIALS,
                                "boundary_trials": BOUNDARY_TRIALS,
                                "alphas": ALPHAS})
    print(f"exp8 ({part}) wall {time.time()-w0:.0f}s")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "all")
