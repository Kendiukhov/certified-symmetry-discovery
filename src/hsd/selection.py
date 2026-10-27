"""Choosing the model class from the data, and what that does to the guarantee.

The certificate is conditional on the fitted class containing the truth.  In
practice the class (here: the polynomial degree) is often chosen from the same
data.  This module implements the procedures compared in the paper.

``gate``
    Enlarge the class until a lack-of-fit F-test against the largest class no
    longer rejects, then certify in the chosen class on the same data.  A
    heuristic: failure to reject is not evidence of correct specification, and
    the chosen class is data-dependent.  No guarantee.

``split``
    Choose the class by the same rule on one half of the data and certify on
    the other half.  This removes the reuse of the data, so the region in the
    chosen class is exact *for the projection of the truth onto that class*;
    it does not make a too-small class contain the truth.  No guarantee unless
    the chosen class is correct.

``largest``
    Certify in the largest class, which is assumed to contain the truth.
    Valid at level ``alpha`` under that assumption.

``simultaneous``
    Certify in every class of the nested family at once.  In a smaller class
    ``d`` the truth is ``F_{beta_d} + h_d``, where ``F_{beta_d}`` is its
    least-squares projection on the design and ``h_d`` the remainder.  If the
    largest class contains the truth, ``h_d`` is a linear image of the
    largest-class coefficients, so a bound ``eta_d`` on ``||h_d||_{L2(nu)}``
    follows from the largest-class confidence region by one trust-region
    maximum; the class-``d`` certificate then pays for ``h_d`` in the ``L2``
    form.  The region of every class covers its own projection of the truth,
    and the largest-class region covers the truth, all on one event of
    probability at least ``1 - alpha`` when the error budget is split as
    ``alpha / K`` per class (Bonferroni over the ``K`` classes).  The reported
    bound for a generator is the smallest over classes, so choosing the class
    after seeing the data costs nothing further.

``lof_invisible_defect`` gives the quantitative link between the power of the
lack-of-fit test and the defect error it can leave undetected.
"""

from __future__ import annotations

import numpy as np
from scipy import optimize, stats

from .certify import Certificate, certify_direction, certify_subspace
from .estimation import _features, fit_ols, lack_of_fit_pvalue
from .polynomials import moment_matrix
from .trs import max_norm_over_ball

__all__ = ["gate_select", "omitted_part_bound", "certify_with_model_error",
           "certified_dimension_with_model_error", "simultaneous_over_classes",
           "lof_noncentrality_for_power", "lof_power", "invisible_gain",
           "lof_invisible_defect"]


# ------------------------------------------------------------------ gate
def gate_select(probs, X, Y, alpha_lof: float = 0.05) -> int:
    """Index of the smallest class in ``probs`` (ordered small to large) whose
    lack-of-fit test against the largest class does not reject."""
    big = probs[-1]
    for i, p in enumerate(probs[:-1]):
        if lack_of_fit_pvalue(p, big, X, Y) >= alpha_lof:
            return i
    return len(probs) - 1


# ------------------------------------------------ omitted part of a class
def _embed(small, big) -> np.ndarray:
    """Matrix mapping small-class coefficients into the big class."""
    E = np.zeros((big.Q, small.Q))
    ms, mb = len(small.beta_idx), len(big.beta_idx)
    pos = {int(a): j for j, a in enumerate(big.beta_idx)}
    for i in range(small.n):
        for j, a in enumerate(small.beta_idx):
            E[i * mb + pos[int(a)], i * ms + j] = 1.0
    return E


def _omitted_map(small, big, X) -> np.ndarray:
    """Linear map from big-class coefficients ``beta`` to the coefficients (in
    the big class) of ``h = F_beta - Pi_small F_beta``, where ``Pi_small`` is
    the least-squares projection on the small class over the design ``X``."""
    Ps, Pb = _features(small, X), _features(big, X)
    proj = np.linalg.lstsq(Ps, Pb, rcond=None)[0]          # (ms, mb)
    Pi = np.kron(np.eye(small.n), proj)                    # small coef from big coef
    return np.eye(big.Q) - _embed(small, big) @ Pi


def _l2_sqrt(big) -> np.ndarray:
    g = moment_matrix(big.alg, big.measure, big.scale, sub=big.beta_idx)
    W = np.eye(big.n) if big.metric is None else big.metric
    G = np.kron(W, g)
    w, U = np.linalg.eigh((G + G.T) / 2)
    return (U * np.sqrt(np.clip(w, 0.0, None))) @ U.T


def omitted_part_bound(small, big, X, fit_big, alpha: float) -> float:
    """Upper confidence bound on ``||h_small||_{L2(nu)}`` from the big-class
    region (valid if the big class contains the truth)."""
    A = _l2_sqrt(big) @ _omitted_map(small, big, X)
    R = fit_big.ellipsoid_radius(alpha)
    return max_norm_over_ball(A @ fit_big.beta, A @ fit_big.cov_sqrt, R)


def certify_with_model_error(prob, theta, fit, alpha, eta2, big) -> Certificate:
    return certify_direction(prob, theta, fit, alpha, eta_l2=(eta2, big))


def certified_dimension_with_model_error(prob, fit, delta, alpha, eta2, big):
    """Dimension of the certified plug-in subspace when the class may omit a
    part ``h`` with ``||h||_{L2(nu)} <= eta2`` lying in the class ``big``.

    For ``d = 1`` the direction certificate with the ``L2`` model-error form is
    used.  For ``d >= 2`` the subspace bound is widened by
    ``(sum_p ||B_p||^2)^{1/2} eta2`` in the numerator and the analogous
    amount in the denominator, with ``B_p`` the bracket map of the ``p``-th
    orthonormal basis generator (a valid, slightly loose bound on
    ``sup_theta ||B_theta||``).
    """
    from .certify import _model_error_gains
    _, Theta = prob.spectrum(fit.beta)
    best_d, best_V = 0, np.zeros((prob.P, 0))
    for d in range(1, prob.P + 1):
        V = Theta[:, :d]
        if d == 1:
            up = certify_direction(prob, V[:, 0], fit, alpha, eta_l2=(eta2, big)).upper
        else:
            Vq, _ = np.linalg.qr(V)
            gains = np.array([_model_error_gains(big, Vq[:, k]) for k in range(d)])
            gL = float(np.sqrt(np.sum(gains[:, 0] ** 2)))
            gM = float(np.sqrt(np.sum(gains[:, 1] ** 2 + gains[:, 2] ** 2)))
            num, den = _subspace_parts(prob, Vq, fit, alpha)
            den = den - gM * eta2
            up = np.inf if den <= 0 else (num + gL * eta2) / den
        if up <= delta:
            best_d, best_V = d, V
        else:
            break
    return best_d, best_V


def _subspace_parts(prob, Vq, fit, alpha):
    R = fit.ellipsoid_radius(alpha)
    S = fit.cov_sqrt
    TA = np.einsum("kpq,pd->kdq", prob.Tw, Vq).reshape(-1, prob.Q)
    TM = np.concatenate([np.einsum("kpq,pd->kdq", prob.T1w, Vq).reshape(-1, prob.Q),
                         np.einsum("kpq,pd->kdq", prob.T2w, Vq).reshape(-1, prob.Q)], axis=0)
    radA = max_norm_over_ball(np.zeros(TA.shape[0]), TA @ S, R)
    radM = max_norm_over_ball(np.zeros(TM.shape[0]), TM @ S, R)
    smax = float(np.linalg.svd(prob.A(fit.beta) @ Vq, compute_uv=False)[0])
    smin = float(np.linalg.svd(prob.M(fit.beta) @ Vq, compute_uv=False)[-1])
    return smax + radA, max(smin - radM, 0.0)


def simultaneous_over_classes(probs, X, Y, delta: float, alpha: float = 0.05,
                              cov: str = "ols"):
    """Certified plug-in subspace, simultaneously valid over a nested family.

    Returns ``(d, V, k)``: the largest certified dimension over the classes,
    its subspace and the index of the class that achieved it.  Each class gets
    ``alpha / K``; the largest class's region also bounds the omitted parts.
    """
    K = len(probs)
    a = alpha / K
    big = probs[-1]
    fits = [fit_ols(p, X, Y, cov=cov) for p in probs]
    best = (0, np.zeros((big.P, 0)), K - 1)
    from .certify import certified_dimension
    for k, (p, f) in enumerate(zip(probs, fits)):
        if k == K - 1:
            d, V = certified_dimension(p, f, delta, a)
        else:
            eta = omitted_part_bound(p, big, X, fits[-1], a)
            d, V = certified_dimension_with_model_error(p, f, delta, a, eta, big)
        if d > best[0]:
            best = (d, V, k)
    return best


# ----------------------------------------- what lack-of-fit cannot see
def lof_power(lam: float, df1: int, df2: int, alpha_lof: float = 0.05) -> float:
    """Power of the lack-of-fit F-test at noncentrality ``lam``."""
    crit = stats.f.ppf(1 - alpha_lof, df1, df2)
    if lam <= 0:          # central case (scipy's noncentral F is unreliable at 0)
        return float(stats.f.sf(crit, df1, df2))
    return float(stats.ncf.sf(crit, df1, df2, lam))


def lof_noncentrality_for_power(gamma: float, df1: int, df2: int,
                                alpha_lof: float = 0.05) -> float:
    """Noncentrality at which the lack-of-fit test has power exactly ``gamma``."""
    if gamma <= alpha_lof:
        return 0.0
    f = lambda lam: lof_power(lam, df1, df2, alpha_lof) - gamma
    hi = 1.0
    while f(hi) < 0:
        hi *= 2.0
    return float(optimize.brentq(f, 0.0, hi, xtol=1e-10))


def invisible_gain(small, big, theta, X=None) -> float:
    """``G(xi) = max ||[xi, h]||_{L2(nu)} / ||h||_{L2(mu)}`` over the fields
    ``h`` of the big class that are orthogonal to the small class in
    ``L2(mu)``; ``mu`` is the empirical measure of ``X`` (or ``nu`` if ``X`` is
    None)."""
    if X is None:
        g = moment_matrix(big.alg, big.measure, big.scale, sub=big.beta_idx)
    else:
        Pb = _features(big, X)
        g = Pb.T @ Pb / X.shape[0]
    W = np.eye(big.n) if big.metric is None else big.metric
    G = np.kron(W, g)
    E = _embed(small, big)
    # Orthogonal complement of the small class in the G geometry.
    Gs = E.T @ G @ E
    Pperp = np.eye(big.Q) - E @ np.linalg.solve(Gs, E.T @ G)
    w, U = np.linalg.eigh((G + G.T) / 2)
    Gh = (U * np.sqrt(np.clip(w, 0, None))) @ U.T
    Gih = (U / np.sqrt(np.clip(w, 1e-300, None))) @ U.T
    # Basis of the complement, G-orthonormal.
    Bc = Pperp @ Gih
    uu, ss, _ = np.linalg.svd(Gh @ Bc)
    keep = ss > 1e-10 * ss.max()
    Hb = Gih @ uu[:, keep]                 # G-orthonormal basis of the complement
    L = big.L_theta(np.asarray(theta, float) / np.linalg.norm(theta))
    return float(np.linalg.svd(L @ Hb, compute_uv=False)[0])


def lof_invisible_defect(small, big, theta, beta_small, sigma: float, N: int,
                         gamma: float = 0.5, alpha_lof: float = 0.05, X=None) -> float:
    """True defect of ``xi_theta``, to first order in the omitted part, when
    ``xi_theta`` is an exact symmetry of the fitted class's field
    ``F_{beta_small}`` and the truth is ``F_{beta_small} + h*``.  Here ``h*``
    is the omitted part of maximal bracket gain ``||[xi, h]||_{nu,W}``, scaled
    so that the lack-of-fit test detects it with probability ``gamma``.  This
    is the value one omitted part attains, not an upper bound: ``h*``
    maximises the bracket term, not the defect, and the exact defect at
    ``F_{beta_small} +/- h*`` can be slightly larger.

    The test of ``small`` against ``big`` has ``df1 = n (m_big - m_small)`` and
    ``df2 = n (N - m_big)`` and noncentrality ``N ||h||^2_{L2(mu)} / sigma^2``;
    the omitted part is invisible at power ``gamma`` when
    ``||h||_{L2(mu)} <= sigma (lambda_gamma / N)^{1/2}``, and it can then
    contribute up to ``G(xi) sigma (lambda_gamma / N)^{1/2}`` to the bracket.
    The denominator is evaluated at ``F_{beta_small}`` (first order in
    ``h``).
    """
    n = small.n
    ms, mb = len(small.beta_idx), len(big.beta_idx)
    df1, df2 = n * (mb - ms), n * (N - mb)
    lam = lof_noncentrality_for_power(gamma, df1, df2, alpha_lof)
    G = invisible_gain(small, big, theta, X)
    den = float(np.linalg.norm(small.M_theta(np.asarray(theta, float)
                                             / np.linalg.norm(theta)) @ beta_small))
    return float(G * sigma * np.sqrt(lam / N) / den)
