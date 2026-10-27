"""Experiment 14 -- choosing the model class from the data.

The truth is the Hopf normal form plus ``eps P_k(x) d/dx``, where ``P_k`` is the
Legendre polynomial of degree ``k`` (``k = 5`` or ``7``).  ``P_k`` is
orthogonal to every polynomial of lower degree under the uniform target
measure, so a degree-three fit sees (up to sampling error) the exactly
rotation-symmetric Hopf field, while the true rotation defect grows linearly
in ``eps``.  The family ``{3, k}`` of polynomial degrees contains the truth in
its largest member.

For each ``(k, eps, N)`` we compare five procedures at ``delta = alpha = 0.05``:

``small``        certify in degree 3 (misspecified for ``eps > 0``);
``gate``         lack-of-fit F-test of degree 3 against degree ``k`` at level
                 0.05 on the same data; certify in degree 3 if it does not
                 reject, otherwise in degree ``k``;
``split``        the same choice made on one half, certification on the other;
``largest``      certify in degree ``k`` (valid);
``simultaneous`` certificates in both classes on one event, with the omitted
                 part of degree 3 bounded from the degree-``k`` region
                 (valid; ``alpha/2`` per class).

We record the false-certification rate (judged against the true degree-``k``
system), the certified dimension, the empirical lack-of-fit rejection rate,
its theoretical value from the noncentral F distribution, and the true defect,
to first order in the omitted part, produced by the omitted part of maximal
bracket gain that the test detects with probability 0.5 (an attained value, not
an upper bound; ``hsd.selection.lof_invisible_defect``).
"""

from __future__ import annotations

import time
from fractions import Fraction
from itertools import product

import numpy as np

from common import ALPHA, DELTA, rms_field, run_cells, save, sup_true_defect
from common import trials
from hsd import DefectProblem, fit_ols, get_system, simulate_design
from hsd.certify import certified_dimension
from hsd.selection import (gate_select, lof_invisible_defect, lof_power,
                           simultaneous_over_classes)
from hsd.systems import System

TRIALS = trials(300)
DEGREES = [5, 7]
NS = [800, 3200, 12800, 51200]
DEFECT_MULTIPLES = [0.0, 0.5, 1.0, 1.25, 1.5, 2.0, 3.0]
SIGMA_REL = 0.1

# Legendre coefficients (exact rationals), highest degree first not needed.
_LEGENDRE = {
    5: {5: Fraction(63, 8), 3: Fraction(-70, 8), 1: Fraction(15, 8)},
    7: {7: Fraction(429, 16), 5: Fraction(-693, 16), 3: Fraction(315, 16),
        1: Fraction(-35, 16)},
}


def hidden_break(k: int, eps: float) -> System:
    e = Fraction(eps).limit_denominator(10 ** 12)
    terms = dict(get_system("hopf").terms)
    for p, c in _LEGENDRE[k].items():
        terms[(0, (p, 0))] = terms.get((0, (p, 0)), Fraction(0)) + e * c
    terms = {kk: v for kk, v in terms.items() if v != 0}
    return System(f"hopf_P{k}_{float(eps):g}", 2, k, terms, "hidden break")


def _rot(prob):
    r = np.zeros(prob.P)
    r[1], r[2] = 1.0, -1.0
    return r / np.linalg.norm(r)


def eps_for_defect(k: int, target: float) -> float:
    """``eps`` whose true rotation defect equals ``target`` (linear in eps for
    small values; solved exactly by bisection)."""
    if target == 0:
        return 0.0
    pk = DefectProblem.build(2, k, "affine", "box", scale=1.0)
    rot = _rot(pk)
    lo, hi = 0.0, 1.0
    while pk.rho(rot, hidden_break(k, hi).beta(pk)) < target:
        hi *= 2
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if pk.rho(rot, hidden_break(k, mid).beta(pk)) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def one_cell(args):
    k, mult, N, seed = args
    rng = np.random.default_rng(seed)
    eps = eps_for_defect(k, mult * DELTA)
    sys_ = hidden_break(k, eps)
    p3 = DefectProblem.build(2, 3, "affine", "box", scale=1.0)
    pk = DefectProblem.build(2, k, "affine", "box", scale=1.0)
    beta_k = sys_.beta(pk)
    sigma = SIGMA_REL * rms_field(get_system("hopf"), np.random.default_rng(0), 1.0)
    procs = ("small", "gate", "split", "largest", "simultaneous")
    fc = {p: 0 for p in procs}
    dim = {p: 0 for p in procs}
    lof = 0
    for _ in range(TRIALS):
        X, Y = simulate_design(sys_, rng, N=N, sigma=sigma, sampler="box", scale=1.0)
        f3, fk = fit_ols(p3, X, Y), fit_ols(pk, X, Y)
        d3, V3 = certified_dimension(p3, f3, DELTA, ALPHA)
        dk, Vk = certified_dimension(pk, fk, DELTA, ALPHA)
        g = gate_select([p3, pk], X, Y, ALPHA)
        lof += g == 1
        dg, Vg = (d3, V3) if g == 0 else (dk, Vk)
        h = N // 2
        ga = gate_select([p3, pk], X[:h], Y[:h], ALPHA)
        pp = (p3, pk)[ga]
        dsp, Vsp = certified_dimension(pp, fit_ols(pp, X[h:], Y[h:]), DELTA, ALPHA)
        ds, Vs, _ = simultaneous_over_classes([p3, pk], X, Y, DELTA, ALPHA)
        for p, (d, V) in zip(procs, ((d3, V3), (dg, Vg), (dsp, Vsp), (dk, Vk), (ds, Vs))):
            dim[p] += d
            fc[p] += d > 0 and sup_true_defect(pk, V, beta_k) > DELTA
    n_s, n_k = len(p3.beta_idx), len(pk.beta_idx)
    df1, df2 = 2 * (n_k - n_s), 2 * (N - n_k)
    # Noncentrality of the population design: ||h||^2_{L2(nu)} = eps^2 / (2k+1).
    lam = N * eps ** 2 / (2 * k + 1) / sigma ** 2
    out = dict(k=k, defect_multiple=mult, true_defect=float(pk.rho(_rot(pk), beta_k)),
               true_min_defect=float(pk.spectrum(beta_k)[0][0]), eps=float(eps), N=N,
               sigma=float(sigma), trials=TRIALS, lof_reject=lof / TRIALS,
               lof_power_theory=lof_power(lam, df1, df2, ALPHA),
               invisible_defect_05=lof_invisible_defect(
                   p3, pk, _rot(p3), get_system("hopf").beta(p3), sigma, N, 0.5))
    for p in procs:
        out[f"fc_{p}"] = fc[p] / TRIALS
        out[f"count_{p}"] = int(fc[p])
        out[f"dim_{p}"] = dim[p] / TRIALS
    return out


def main():
    cells = [(k, m, N, 14000 + 307 * i)
             for i, (k, m, N) in enumerate(product(DEGREES, DEFECT_MULTIPLES, NS))]
    w0 = time.time()
    rows = run_cells(one_cell, cells, desc="exp14")
    save("exp14_selection", {"rows": rows, "trials": TRIALS, "degrees": DEGREES,
                             "Ns": NS, "defect_multiples": DEFECT_MULTIPLES,
                             "sigma_rel": SIGMA_REL})
    print(f"exp14 wall {time.time()-w0:.0f}s")


if __name__ == "__main__":
    main()
