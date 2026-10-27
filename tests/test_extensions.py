"""Tests for the certified numerics, the sharp ratio certificate, the output
metric, the dependence-robust covariances, model-class selection, dictionary
models and the closure diagnostic."""

from __future__ import annotations

import numpy as np
import pytest

from hsd import DefectProblem, exact_symmetry_algebra, fit_ols, get_system, simulate_design
from hsd.certify import certified_dimension, certify_direction
from hsd.estimation import region_covers
from hsd.trs import (max_norm_over_ball, min_norm_over_ball, ratio_inf_over_ball,
                     ratio_sup_over_ball, trs_min_lower_bound)


def _ball(rng, q, R, n):
    U = rng.normal(size=(n, q))
    U /= np.linalg.norm(U, axis=1, keepdims=True)
    return np.vstack([U * (R * rng.uniform(size=(n, 1)) ** (1.0 / q)), U * R])


# ----------------------------------------------------- certified numerics
@pytest.mark.parametrize("seed", range(10))
def test_ratio_extrema_dominate_samples_and_are_tight(seed):
    rng = np.random.default_rng(100 + seed)
    k, q = int(rng.integers(2, 8)), int(rng.integers(2, 7))
    R = float(rng.uniform(0.2, 2.5))
    aN, BN = rng.normal(size=k), rng.normal(size=(k, q))
    aD, BD = 3 * rng.normal(size=k), 0.4 * rng.normal(size=(k, q))
    U = _ball(rng, q, R, 50_000)
    r = np.linalg.norm(aN + U @ BN.T, axis=1) / np.linalg.norm(aD + U @ BD.T, axis=1)
    sup = ratio_sup_over_ball(aN, BN, aD, BD, R)
    inf = ratio_inf_over_ball(aN, BN, aD, BD, R)
    assert sup >= r.max() * (1 - 1e-12)
    assert inf <= r.min() * (1 + 1e-12) + 1e-15
    # Tightness: within a small margin of the best sampled value refined
    # along the boundary by projected gradient ascent.
    u = U[np.argmax(r)]
    f = lambda u: np.linalg.norm(aN + BN @ u) / np.linalg.norm(aD + BD @ u)
    for _ in range(3000):
        g = np.zeros(q)
        for j in range(q):
            e = np.zeros(q); e[j] = 1e-7
            g[j] = (f(u + e) - f(u - e)) / 2e-7
        u = u + 0.01 * g
        if np.linalg.norm(u) > R:
            u *= R / np.linalg.norm(u)
    assert f(u) <= sup * (1 + 1e-9)
    assert sup <= f(u) * (1 + 1e-4)


def test_trs_lower_bound_is_valid_for_indefinite_quadratics():
    rng = np.random.default_rng(3)
    for _ in range(40):
        q = int(rng.integers(2, 6))
        A = rng.normal(size=(q, q)); H = (A + A.T) / 2
        g, k, R = rng.normal(size=q), float(rng.normal()), float(rng.uniform(0.3, 2))
        if rng.uniform() < 0.3:                  # hard case: g orthogonal to bottom eigvec
            w, V = np.linalg.eigh(H); g = g - V[:, 0] * (V[:, 0] @ g)
        U = _ball(rng, q, R, 40_000)
        vals = np.einsum("ni,ij,nj->n", U, H, U) + 2 * U @ g + k
        lb = trs_min_lower_bound(H, g, k, R)
        assert lb <= vals.min() + 1e-9
        assert lb >= vals.min() - 0.05 * (1 + abs(vals.min()))   # sampled min is above the true min


def test_norm_bounds_are_on_the_safe_side():
    rng = np.random.default_rng(4)
    for _ in range(30):
        k, q = int(rng.integers(1, 6)), int(rng.integers(1, 6))
        a, B, R = rng.normal(size=k), rng.normal(size=(k, q)), float(rng.uniform(0.1, 3))
        U = _ball(rng, q, R, 30_000)
        v = np.linalg.norm(a + U @ B.T, axis=1)
        assert max_norm_over_ball(a, B, R) >= v.max()
        assert min_norm_over_ball(a, B, R) <= v.min()


# --------------------------------------------------------- sharp certificate
def test_sharp_certificate_is_inside_the_separate_one():
    rng = np.random.default_rng(5)
    prob = DefectProblem.build(2, 3, "affine", "box")
    sys_ = get_system("vanderpol")
    for _ in range(10):
        X, Y = simulate_design(sys_, rng, N=300, sigma=0.3)
        fit = fit_ols(prob, X, Y)
        th = rng.normal(size=prob.P)
        s = certify_direction(prob, th, fit, 0.05, ratio="sharp")
        e = certify_direction(prob, th, fit, 0.05, ratio="separate")
        assert e.lower - 1e-12 <= s.lower <= s.plug_in <= s.upper <= e.upper + 1e-12


# --------------------------------------------------------------- metric
def _scaled_system_beta(sys_, prob_z, s):
    """Coefficients of F~(z) = S^{-1} F(S z) for S = diag(s)."""
    Fz = np.zeros((sys_.n, prob_z.alg.m))
    for (i, a), c in sys_.terms.items():
        Fz[i, prob_z.alg.index[a]] = float(c) * np.prod(np.asarray(s) ** np.array(a)) / s[i]
    return prob_z.beta_of_field(Fz)


def test_defect_is_invariant_under_unit_changes_with_transported_metric():
    sys_ = get_system("lotka_volterra")
    s = np.array([10.0, 0.5])                        # change of units x = S z
    prob_x = DefectProblem.build(2, 2, "affine", "box", scale=1.0)
    prob_z = DefectProblem.build(2, 2, "affine", "box", scale=1.0 / s,
                                 metric=np.diag(s ** 2))
    prob_z_euc = DefectProblem.build(2, 2, "affine", "box", scale=1.0 / s)
    bx, bz = sys_.beta(prob_x), _scaled_system_beta(sys_, prob_z, s)
    rng = np.random.default_rng(6)
    Sinv = np.diag(1.0 / s)
    for _ in range(20):
        Mx, b = rng.normal(size=(2, 2)), rng.normal(size=2)
        Mz, bzv = Sinv @ Mx @ np.diag(s), Sinv @ b          # xi~(z) = S^-1 xi(S z)
        th_x = np.concatenate([Mx.ravel(), b])
        th_z = np.concatenate([Mz.ravel(), bzv])
        assert prob_z.rho(th_z, bz) == pytest.approx(prob_x.rho(th_x, bx), rel=1e-9)
    # with the Euclidean metric the defect changes
    diffs = [abs(prob_z_euc.rho(np.concatenate([(Sinv @ M @ np.diag(s)).ravel(), Sinv @ b]), bz)
                 - prob_x.rho(np.concatenate([M.ravel(), b]), bx))
             for M, b in [(rng.normal(size=(2, 2)), rng.normal(size=2)) for _ in range(5)]]
    assert max(diffs) > 1e-3


# ------------------------------------------------------------ covariances
def test_sandwich_covariances_are_psd_and_cluster_rank_is_respected():
    rng = np.random.default_rng(7)
    prob = DefectProblem.build(2, 3, "affine", "box")
    sys_ = get_system("hopf")
    X, Y = simulate_design(sys_, rng, N=2000, sigma=0.1)
    groups = np.repeat(np.arange(100), 20)
    for cov in ("hc0", "cluster", "hac"):
        f = fit_ols(prob, X, Y, cov=cov, groups=groups)
        assert np.linalg.eigvalsh(f.cov).min() > -1e-12
    few = np.repeat(np.arange(8), 250)              # 8 clusters < 20 coefficients
    f = fit_ols(prob, X, Y, cov="cluster", groups=few)
    assert f.null_basis is not None and f.null_basis.shape[1] >= 20 - 8
    th = np.zeros(prob.P); th[1], th[2] = 1, -1
    assert not np.isfinite(certify_direction(prob, th, f, 0.05).upper)


def test_region_coverage_at_nominal_level():
    rng = np.random.default_rng(8)
    prob = DefectProblem.build(2, 3, "affine", "box")
    sys_ = get_system("duffing")
    b0 = sys_.beta(prob)
    hits = sum(region_covers(fit_ols(prob, *simulate_design(sys_, rng, N=200, sigma=0.3)), b0, 0.2)
               for _ in range(400))
    assert abs(hits / 400 - 0.8) < 0.06


# ---------------------------------------------------------------- selection
def test_omitted_part_bound_covers_the_true_omitted_part():
    from hsd.selection import _l2_sqrt, _omitted_map, omitted_part_bound
    from hsd.systems import hidden_break_hopf
    rng = np.random.default_rng(9)
    p3, p5 = (DefectProblem.build(2, d, "affine", "box") for d in (3, 5))
    sys_ = hidden_break_hopf(0.05)
    misses = 0
    for _ in range(100):
        X, Y = simulate_design(sys_, rng, N=500, sigma=0.1)
        f5 = fit_ols(p5, X, Y)
        true = np.linalg.norm(_l2_sqrt(p5) @ _omitted_map(p3, p5, X) @ sys_.beta(p5))
        misses += omitted_part_bound(p3, p5, X, f5, 0.1) < true
    assert misses <= 10


def test_lof_power_and_invisible_gain():
    from hsd.selection import invisible_gain, lof_noncentrality_for_power, lof_power
    lam = lof_noncentrality_for_power(0.5, 22, 3000)
    assert lof_power(lam, 22, 3000) == pytest.approx(0.5, abs=1e-6)
    p3, p5 = (DefectProblem.build(2, d, "affine", "box") for d in (3, 5))
    rot = np.zeros(p3.P); rot[1], rot[2] = 1, -1
    G = invisible_gain(p3, p5, rot)
    # Direct check on the hidden P5 term, which is orthogonal to degree three.
    from hsd.systems import hidden_break_hopf
    h = hidden_break_hopf(1.0).beta(p5) - hidden_break_hopf(0.0).beta(p5)
    from hsd.selection import _l2_sqrt
    ratio = np.linalg.norm(p5.L_theta(rot / np.linalg.norm(rot)) @ h) / np.linalg.norm(_l2_sqrt(p5) @ h)
    assert ratio <= G * (1 + 1e-9)


# --------------------------------------------------------------- dictionaries
def test_polynomial_dictionary_reproduces_the_exact_problem():
    from hsd.dictionary import Dictionary, DictionaryProblem, gauss_legendre_rule, product_rule
    from hsd.dict_systems import _affine_generators
    prob = DefectProblem.build(2, 3, "affine", "box", scale=1.0)
    exps = [prob.alg.basis[a] for a in prob.beta_idx]
    funcs, grads = [], []
    for (px, py) in exps:
        funcs.append(lambda X, px=px, py=py: X[:, 0] ** px * X[:, 1] ** py)

        def g(X, px=px, py=py):
            G = np.zeros_like(X)
            if px:
                G[:, 0] = px * X[:, 0] ** (px - 1) * X[:, 1] ** py
            if py:
                G[:, 1] = py * X[:, 0] ** px * X[:, 1] ** (py - 1)
            return G
        grads.append(g)
    d = Dictionary(funcs, grads, [str(e) for e in exps])
    X, W = product_rule(gauss_legendre_rule(8, -1, 1), gauss_legendre_rule(8, -1, 1))
    dp = DictionaryProblem.build(2, d, _affine_generators(), X, W)
    rng = np.random.default_rng(10)
    for _ in range(10):
        th, b = rng.normal(size=prob.P), rng.normal(size=prob.Q)
        assert dp.rho(th, b) == pytest.approx(prob.rho(th, b), rel=1e-10)


def _dictionary_case(family, eps, refine=1):
    """Problem and coefficients at a reported setting; ``refine=2`` doubles the rule."""
    from hsd.dict_systems import pendulum_beta, pendulum_problem, radial_beta, radial_problem
    if family == "radial":
        prob = radial_problem(40 * refine, 24 * refine)
        return prob, radial_beta(prob, eps)
    prob = pendulum_problem(32 * refine)
    return prob, pendulum_beta(prob, eps)


_ROT = np.array([0.0, 1.0, -1.0, 0.0, 0.0, 0.0])          # y d/dx - x d/dy
_GEN = np.random.default_rng(12).normal(size=6)          # a fixed generic generator
_REPORTED = [("radial", e) for e in (0.0, 0.03, 0.06, 0.12)] + [("pendulum", 0.1)]


@pytest.mark.parametrize("family,eps", _REPORTED)
def test_dictionary_quadrature_refinement(family, eps):
    # Doubling the rule (radial: radial Gauss--Legendre nodes 40 -> 80 and
    # angular grid 24 -> 48; pendulum: 32x32 -> 64x64 product rule) changes
    # the defect spectrum and the defects of fixed generators by less than a
    # relative 1e-10.
    prob, beta = _dictionary_case(family, eps)
    fine, beta_f = _dictionary_case(family, eps, refine=2)
    r, rf = prob.spectrum(beta)[0], fine.spectrum(beta_f)[0]
    assert np.isfinite(r).all()
    if family == "radial" and eps == 0.0:
        # exact rotation symmetry: the smallest eigenvalue is a numerical zero
        # (square root of rounding error) under both rules; compare the rest
        assert prob.rho(_ROT, beta) < 1e-12 and fine.rho(_ROT, beta_f) < 1e-12
        assert max(r[0], rf[0]) < 1e-7
        r, rf = r[1:], rf[1:]
    np.testing.assert_allclose(r, rf, rtol=1e-10, atol=1e-12)
    for th in (_ROT, _GEN):
        assert prob.rho(th, beta) == pytest.approx(fine.rho(th, beta_f), rel=1e-10, abs=1e-12)


def _analytic_field(family, eps, X):
    """True field and its Jacobian, written out without the dictionary."""
    J = np.zeros((X.shape[0], 2, 2))
    if family == "radial":
        x, y = X[:, 0], X[:, 1]
        r = np.sqrt(x * x + y * y)
        s = 1.0 + r * r
        rr = np.maximum(r, 1e-300)
        F = np.stack([x - x * r - y / s + eps * x * x, y - y * r + x / s], axis=1)
        J[:, 0, 0] = 1 - r - x * x / rr + 2 * x * y / s ** 2 + 2 * eps * x
        J[:, 0, 1] = -x * y / rr - 1 / s + 2 * y * y / s ** 2
        J[:, 1, 0] = -x * y / rr + 1 / s - 2 * x * x / s ** 2
        J[:, 1, 1] = 1 - r - y * y / rr - 2 * x * y / s ** 2
        return F, J
    th, om = X[:, 0], X[:, 1]
    F = np.stack([om, -np.sin(th) - eps * om], axis=1)
    J[:, 0, 1], J[:, 1, 0], J[:, 1, 1] = 1.0, -np.cos(th), -eps
    return F, J


def _mc_defect(theta, X, F, J):
    """Monte Carlo defect of the affine generator ``theta`` and its delta-method
    standard error (covariance of numerator and denominator included)."""
    M, b = theta[:4].reshape(2, 2), theta[4:]
    t1 = np.einsum("kij,kj->ki", J, X @ M.T + b)                 # DF xi
    t2 = F @ M.T                                                 # D xi F
    num = np.sum((t1 - t2) ** 2, axis=1)
    den = np.sum(t1 ** 2, axis=1) + np.sum(t2 ** 2, axis=1)
    ratio = num.mean() / den.mean()
    se_ratio = np.std(num - ratio * den, ddof=1) / (np.sqrt(num.size) * den.mean())
    rho = np.sqrt(ratio)
    return rho, se_ratio / (2 * rho)


@pytest.mark.parametrize("family,eps,seed", [("radial", 0.06, 13), ("radial", 0.12, 14),
                                             ("pendulum", 0.1, 15)])
def test_dictionary_defects_match_monte_carlo(family, eps, seed):
    # 10^6 points drawn from the target measure (area measure on the unit
    # disc, uniform on the pendulum box); the bracket is evaluated from the
    # analytic field, not from the dictionary.
    from hsd.dict_systems import box_sample, disc_sample
    prob, beta = _dictionary_case(family, eps)
    sampler = disc_sample if family == "radial" else box_sample
    X = sampler(np.random.default_rng(seed), 1_000_000)
    F, J = _analytic_field(family, eps, X)
    assert np.abs(prob.rhs(beta, X) - F).max() < 1e-12    # the class contains the truth
    lowest = prob.spectrum(beta)[1][:, 0]
    for th in (_ROT, _GEN, lowest):
        rho_mc, se = _mc_defect(th, X, F, J)
        assert abs(rho_mc - prob.rho(th, beta)) <= 3 * se


# ------------------------------------------------------------- closure / D
def test_closure_diagnostic():
    from hsd.closure import closure_report
    prob = DefectProblem.build(3, 3, "affine", "box")
    S = exact_symmetry_algebra(get_system("sphere_flow3"), prob)
    assert closure_report(prob, S)["max_closure"] < 1e-12
    assert closure_report(prob, S[:, :2])["max_closure"] > 0.5


def test_undefined_defect_directions_are_never_certified():
    from hsd.systems import System
    from fractions import Fraction
    sys_ = System("const", 2, 1, {(0, (0, 0)): Fraction(1)}, "constant field")
    prob = DefectProblem.build(2, 1, "affine", "box")
    rho, Theta = prob.spectrum(sys_.beta(prob))
    assert np.isinf(rho[-1])
    rng = np.random.default_rng(11)
    X, Y = simulate_design(sys_, rng, N=400, sigma=0.01)
    fit = fit_ols(prob, X, Y)
    d, V = certified_dimension(prob, fit, 0.05)
    tr = np.zeros(prob.P); tr[4] = 1.0               # d/dx: undefined (0/0) defect
    assert not certify_direction(prob, tr, fit, 0.05).upper <= 0.05
