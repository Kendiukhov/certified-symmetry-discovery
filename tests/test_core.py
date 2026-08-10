"""Correctness tests for the polynomial algebra, the defect and the certificates.

Every claim in the paper rests on these being right, so the tests check the
symbolic algebra against finite differences, the exact moments against
Monte-Carlo integration, the trust-region extrema against brute force, and the
certificate coverage against its nominal level.
"""

from __future__ import annotations

import numpy as np
import pytest

from hsd import (DefectProblem, bracket, exact_symmetry_algebra, fit_ols,
                 get_system, jac_times, moment_matrix, poly_algebra, simulate_design)
from hsd.certify import certify_direction, certify_subspace, refute_all
from hsd.trs import max_norm_over_ball, min_norm_over_ball


# ---------------------------------------------------------------- polynomials
def test_features_and_eval():
    alg = poly_algebra(2, 3)
    x = np.array([[2.0, 3.0]])
    Phi = alg.features(x)[0]
    for i, a in enumerate(alg.basis):
        assert Phi[i] == pytest.approx(2.0 ** a[0] * 3.0 ** a[1])


def test_bracket_matches_finite_differences():
    rng = np.random.default_rng(0)
    n, alg = 3, poly_algebra(3, 4)
    sub = alg.sub_index(2)
    xi = np.zeros((n, alg.m))
    F = np.zeros((n, alg.m))
    xi[:, sub] = rng.normal(size=(n, len(sub)))
    F[:, sub] = rng.normal(size=(n, len(sub)))
    br = bracket(alg, xi, F)
    x = rng.normal(size=(5, n))
    h = 1e-6

    def Dv(V, x):
        out = np.zeros((x.shape[0], n, n))
        for j in range(n):
            e = np.zeros(n)
            e[j] = h
            out[:, :, j] = (alg.eval(V, x + e) - alg.eval(V, x - e)) / (2 * h)
        return out

    lhs = alg.eval(br, x)
    rhs = np.einsum("kij,kj->ki", Dv(F, x), alg.eval(xi, x)) - \
        np.einsum("kij,kj->ki", Dv(xi, x), alg.eval(F, x))
    assert np.allclose(lhs, rhs, atol=1e-5)


def test_jac_times_matches_direct():
    rng = np.random.default_rng(1)
    n, alg = 2, poly_algebra(2, 4)
    V = np.zeros((n, alg.m)); W = np.zeros((n, alg.m))
    sub = alg.sub_index(2)
    V[:, sub] = rng.normal(size=(n, len(sub)))
    W[:, sub] = rng.normal(size=(n, len(sub)))
    out = jac_times(alg, V, W)
    x = rng.normal(size=(7, n)); h = 1e-6
    J = np.zeros((7, n, n))
    for j in range(n):
        e = np.zeros(n); e[j] = h
        J[:, :, j] = (alg.eval(V, x + e) - alg.eval(V, x - e)) / (2 * h)
    assert np.allclose(alg.eval(out, x), np.einsum("kij,kj->ki", J, alg.eval(W, x)), atol=1e-5)


@pytest.mark.parametrize("measure,kw", [("box", {}), ("gauss", {}), ("ball", {}),
                                        ("annulus", {"inner": 0.4})])
def test_moment_matrix_matches_monte_carlo(measure, kw):
    rng = np.random.default_rng(2)
    n, alg = 2, poly_algebra(2, 3)
    g = moment_matrix(alg, measure, scale=1.3, **kw)
    N, chunk = 4_000_000, 250_000     # chunked so peak memory stays small
    gmc = np.zeros((alg.m, alg.m))
    for _ in range(N // chunk):
        if measure == "box":
            X = rng.uniform(-1.3, 1.3, size=(chunk, n))
        elif measure == "gauss":
            X = rng.normal(0, 1.3, size=(chunk, n))
        else:
            V = rng.normal(size=(chunk, n))
            V /= np.linalg.norm(V, axis=1, keepdims=True)
            r0 = kw.get("inner", 0.0)
            u = rng.uniform(size=chunk)
            X = V * ((r0 ** n + u * (1.3 ** n - r0 ** n)) ** (1.0 / n))[:, None]
        Phi = alg.features(X)
        gmc += Phi.T @ Phi
    gmc /= N
    # Compare on the Cauchy-Schwarz scale: entries of a Gram matrix of monomials
    # up to degree 3 span several orders of magnitude, and the Monte-Carlo error
    # of the high-degree entries is proportional to their own size.
    s = np.sqrt(np.outer(np.diag(g), np.diag(g)))
    assert np.max(np.abs(g - gmc) / s) < 0.02


# --------------------------------------------------------------------- defect
def test_defect_matches_monte_carlo_integrals():
    rng = np.random.default_rng(3)
    prob = DefectProblem.build(2, 3, "affine", "box", scale=1.0)
    sys_ = get_system("hopf")
    beta = sys_.beta(prob)
    theta = rng.normal(size=prob.P)
    theta /= np.linalg.norm(theta)
    alg = prob.alg
    xi = np.einsum("p,pij->ij", theta, prob.gen_basis)
    F = prob.field_of_beta(beta)
    br = bracket(alg, xi, F)
    t1 = jac_times(alg, F, xi); t2 = jac_times(alg, xi, F)
    num = den = 0.0
    n_mc, chunk = 2_000_000, 250_000
    for _ in range(n_mc // chunk):
        X = rng.uniform(-1, 1, size=(chunk, 2))
        num += np.sum(alg.eval(br, X) ** 2)
        den += np.sum(alg.eval(t1, X) ** 2) + np.sum(alg.eval(t2, X) ** 2)
    assert prob.rho(theta, beta) == pytest.approx(np.sqrt(num / den), rel=3e-3)


def test_defect_invariances():
    rng = np.random.default_rng(4)
    prob = DefectProblem.build(2, 3, "affine", "box")
    beta = get_system("hopf").beta(prob)
    theta = rng.normal(size=prob.P)
    r = prob.rho(theta, beta)
    assert prob.rho(3.7 * theta, beta) == pytest.approx(r)      # scaling of xi
    assert prob.rho(theta, -2.5 * beta) == pytest.approx(r)     # scaling of F


def test_true_symmetry_has_zero_defect():
    prob = DefectProblem.build(2, 3, "affine", "box")
    sys_ = get_system("hopf")
    basis = exact_symmetry_algebra(sys_, prob)
    assert basis.shape[1] >= 1
    beta = sys_.beta(prob)
    for k in range(basis.shape[1]):
        assert prob.rho(basis[:, k], beta) < 1e-12


def test_exact_symmetry_algebra_matches_centraliser():
    """For x' = A x with linear generators, the symmetry algebra is the
    centraliser of A, whose dimension we can compute independently."""
    for name in ("lin_rot2", "lin_diag2", "lin_jordan2", "lin_generic3", "lin_scalar3"):
        sys_ = get_system(name)
        n = sys_.n
        prob = DefectProblem.build(n, 1, "linear", "box")
        A = np.zeros((n, n))
        for (i, a), c in sys_.terms.items():
            A[i, np.argmax(a)] = float(c)
        # dim of the commutant = dim null of the Kronecker commutator map
        Kron = np.kron(np.eye(n), A) - np.kron(A.T, np.eye(n))
        expect = n * n - np.linalg.matrix_rank(Kron, tol=1e-10)
        got = exact_symmetry_algebra(sys_, prob).shape[1]
        assert got == expect, (name, got, expect)


def test_lorenz_has_no_continuous_affine_symmetry():
    prob = DefectProblem.build(3, 2, "affine", "box")
    assert exact_symmetry_algebra(get_system("lorenz"), prob).shape[1] == 0


def test_sphere_flow_has_so3():
    prob = DefectProblem.build(3, 3, "affine", "box")
    assert exact_symmetry_algebra(get_system("sphere_flow3"), prob).shape[1] == 3


# ------------------------------------------------------------ trust regions
@pytest.mark.parametrize("seed", range(12))
def test_trs_extrema_beat_brute_force(seed):
    rng = np.random.default_rng(seed)
    k, q = rng.integers(2, 7), rng.integers(2, 7)
    a = rng.normal(size=k) * rng.choice([0.0, 1.0, 5.0])
    B = rng.normal(size=(k, q))
    if seed % 3 == 0:                       # rank-deficient case
        B[:, 0] = 0.0
    R = float(rng.uniform(0.1, 3.0))
    hi = max_norm_over_ball(a, B, R)
    lo = min_norm_over_ball(a, B, R)
    U = rng.normal(size=(200_000, q))
    U /= np.linalg.norm(U, axis=1, keepdims=True)
    U *= rng.uniform(size=(200_000, 1)) ** (1.0 / q) * R
    vals = np.linalg.norm(a + U @ B.T, axis=1)
    assert hi >= vals.max() - 1e-8
    assert lo <= vals.min() + 1e-8
    assert hi <= np.linalg.norm(a) + R * np.linalg.norm(B, 2) + 1e-8


# -------------------------------------------------------------- certificates
def test_certificate_brackets_the_truth_and_covers():
    """Empirical coverage of the simultaneous certificate at its nominal level."""
    rng = np.random.default_rng(7)
    prob = DefectProblem.build(2, 3, "affine", "box", scale=1.0)
    sys_ = get_system("hopf")
    beta0 = sys_.beta(prob)
    alpha, trials, miss = 0.1, 300, 0
    for t in range(trials):
        X, Y = simulate_design(sys_, rng, N=400, sigma=0.3, sampler="box", scale=1.0)
        fit = fit_ols(prob, X, Y)
        _, Theta = prob.spectrum(fit.beta)
        bad = False
        for k in range(prob.P):
            c = certify_direction(prob, Theta[:, k], fit, alpha, "simultaneous")
            true = prob.rho(Theta[:, k], beta0)
            if not (c.lower - 1e-9 <= true <= c.upper + 1e-9):
                bad = True
        miss += bad
    assert miss <= trials * (alpha + 0.05)


def test_subspace_certificate_dominates_directions():
    rng = np.random.default_rng(8)
    prob = DefectProblem.build(2, 3, "affine", "box")
    sys_ = get_system("hopf")
    X, Y = simulate_design(sys_, rng, N=800, sigma=0.2)
    fit = fit_ols(prob, X, Y)
    _, Theta = prob.spectrum(fit.beta)
    V = Theta[:, :2]
    cs = certify_subspace(prob, V, fit, 0.05)
    beta0 = sys_.beta(prob)
    for _ in range(200):
        c = rng.normal(size=2)
        th = V @ (c / np.linalg.norm(c))
        assert prob.rho(th, beta0) <= cs.upper + 1e-9


def test_refute_all_is_a_valid_lower_bound():
    rng = np.random.default_rng(9)
    prob = DefectProblem.build(2, 3, "affine", "box")
    sys_ = get_system("vanderpol")
    beta0 = sys_.beta(prob)
    truemin = prob.spectrum(beta0)[0][0]
    for _ in range(50):
        X, Y = simulate_design(sys_, rng, N=1500, sigma=0.05)
        fit = fit_ols(prob, X, Y)
        assert refute_all(prob, fit, 0.05) <= truemin + 1e-9


def test_pointwise_certificate_covers_on_independent_direction():
    rng = np.random.default_rng(10)
    prob = DefectProblem.build(2, 3, "affine", "box")
    sys_ = get_system("duffing")
    beta0 = sys_.beta(prob)
    theta = rng.normal(size=prob.P); theta /= np.linalg.norm(theta)
    alpha, trials, miss = 0.1, 400, 0
    for _ in range(trials):
        X, Y = simulate_design(sys_, rng, N=300, sigma=0.4)
        fit = fit_ols(prob, X, Y)
        c = certify_direction(prob, theta, fit, alpha, "pointwise")
        true = prob.rho(theta, beta0)
        miss += not (c.lower - 1e-9 <= true <= c.upper + 1e-9)
    assert miss <= trials * alpha


def test_satterthwaite_matches_monte_carlo():
    from hsd.baselines import satterthwaite_pvalue
    rng = np.random.default_rng(11)
    lam = np.array([3.0, 1.0, 0.5, 0.1])
    V = np.diag(lam)
    draws = (rng.normal(size=(400_000, 4)) ** 2) @ lam
    for qq in (0.5, 0.8, 0.95):
        s = float(np.quantile(draws, qq))
        assert satterthwaite_pvalue(s, V) == pytest.approx(1 - qq, abs=0.03)
