"""SymCert: confidence certificates for the relative equivariance defect.

Given a fitted vector field ``beta_hat`` with a confidence region for ``beta``,
we return bounds on the defect of the **true** system,

    rho(theta; beta_true) = || Lt(theta) beta_true || / || Mt(theta) beta_true ||,

both of whose ingredients are norms of linear images of ``beta``.  Two regimes:

``simultaneous``
    Propagate the ``1 - alpha`` confidence ellipsoid for ``beta`` through the
    two norms with exact trust-region extrema.  The resulting upper bound
    ``U(theta)`` is valid **for every generator simultaneously**, because the
    single event ``{beta_true in ellipsoid}`` implies all of them at once.  No
    multiplicity correction, no sample splitting.

``pointwise``
    For a generator chosen independently of the fit (e.g. selected on a held-out
    split), use Gaussian concentration of ``|| Lt(theta) (beta_hat - beta) ||``.
    This is tighter -- its width scales with the *effective* dimension
    ``tr(V)/||V||`` rather than with the full parameter count.

A generator is **certified at tolerance delta** iff its upper bound is at most
``delta``.  Because the bound covers the truth with probability ``1 - alpha``,
the probability of certifying any generator whose true defect exceeds ``delta``
is at most ``alpha`` -- a family-wise false-certification guarantee.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from .trs import max_norm_over_ball, min_norm_over_ball

__all__ = ["Certificate", "certify_direction", "certify_direction_bootstrap",
           "certify_subspace", "refute_all", "certified_dimension"]


@dataclass
class Certificate:
    upper: float
    lower: float
    plug_in: float
    mode: str
    alpha: float


def _sigma_upper(fit, alpha: float) -> float:
    """Upper confidence factor for the noise scale (multiplies ``cov``)."""
    if not np.isfinite(fit.dof):
        return 1.0
    return float(fit.dof / stats.chi2.ppf(alpha, fit.dof))


def certify_direction(prob, theta: np.ndarray, fit, alpha: float = 0.05,
                      mode: str = "simultaneous",
                      eta: tuple[float, float] = (0.0, 0.0),
                      eta_l2: tuple[float, object] | None = None) -> Certificate:
    """Two-sided confidence bounds on ``rho(theta; beta_true)``.

    Model error can be admitted in either of two ways.

    ``eta = (eta0, eta1)`` -- the truth is ``F = F_beta + h`` with
    ``sup |h| <= eta0`` and ``sup ||Dh||_F <= eta1`` on the target domain, ``h``
    otherwise arbitrary and smooth.  The bracket then moves by at most
    ``eta1 ||xi||_{L2} + eta0 ||D xi||_{L2}``.  Valid for any smooth truth, but
    conservative because it spends supremum norms.

    ``eta_l2 = (eta2, prob_big)`` -- the truth lies in the larger polynomial
    class described by ``prob_big`` and the omitted part has
    ``||h||_{L2(nu)} <= eta2``.  The bracket then moves by at most
    ``||B_theta||_op eta2`` with ``B_theta`` the exact (computable) map from
    ``h`` to ``[xi_theta, h]`` in the two ``L2(nu)`` geometries.  This is much
    tighter and is the recommended form whenever the truth can be bracketed by
    a richer polynomial class.

    ``eta = (0, 0)`` with ``eta_l2 = None`` assumes the model class contains
    the truth.
    """
    theta = np.asarray(theta, dtype=float)
    theta = theta / np.linalg.norm(theta)
    L = prob.L_theta(theta)
    M1, M2 = prob.M1_theta(theta), prob.M2_theta(theta)
    a_L, a_1, a_2 = L @ fit.beta, M1 @ fit.beta, M2 @ fit.beta
    plug = float(np.linalg.norm(a_L)
                 / max(np.hypot(np.linalg.norm(a_1), np.linalg.norm(a_2)), 1e-300))

    if fit.null_basis is not None and fit.null_basis.size:
        # The design leaves some coefficient directions completely undetermined.
        if np.linalg.norm(L @ fit.null_basis) > 1e-10 * max(np.linalg.norm(L), 1e-300):
            return Certificate(np.inf, 0.0, plug, mode, alpha)

    if mode == "simultaneous":
        R = fit.ellipsoid_radius(alpha)
        S = fit.cov_sqrt
        hi_num = max_norm_over_ball(a_L, L @ S, R)
        lo_num = min_norm_over_ball(a_L, L @ S, R)
        hi_1 = max_norm_over_ball(a_1, M1 @ S, R)
        lo_1 = min_norm_over_ball(a_1, M1 @ S, R)
        hi_2 = max_norm_over_ball(a_2, M2 @ S, R)
        lo_2 = min_norm_over_ball(a_2, M2 @ S, R)
    elif mode == "pointwise":
        a_num, a_den, a_sig = 0.98 * alpha, 0.005 * alpha, 0.01 * alpha
        cov = fit.cov * _sigma_upper(fit, a_sig)
        rL = _gauss_norm_bound(L, cov, a_num)
        r1 = _gauss_norm_bound(M1, cov, a_den)
        r2 = _gauss_norm_bound(M2, cov, a_den)
        hi_num, lo_num = np.linalg.norm(a_L) + rL, max(np.linalg.norm(a_L) - rL, 0.0)
        hi_1, lo_1 = np.linalg.norm(a_1) + r1, max(np.linalg.norm(a_1) - r1, 0.0)
        hi_2, lo_2 = np.linalg.norm(a_2) + r2, max(np.linalg.norm(a_2) - r2, 0.0)
    else:
        raise ValueError(mode)

    if eta != (0.0, 0.0):
        nxi, ndxi = prob.generator_norms(theta)
        s_num, s_1, s_2 = eta[1] * nxi + eta[0] * ndxi, eta[1] * nxi, eta[0] * ndxi
        hi_num, lo_num = hi_num + s_num, max(lo_num - s_num, 0.0)
        lo_1, hi_1 = max(lo_1 - s_1, 0.0), hi_1 + s_1
        lo_2, hi_2 = max(lo_2 - s_2, 0.0), hi_2 + s_2

    if eta_l2 is not None and eta_l2[0] > 0:
        eta2, big = eta_l2
        cL, c1, c2 = _model_error_gains(big, theta)
        hi_num, lo_num = hi_num + cL * eta2, max(lo_num - cL * eta2, 0.0)
        lo_1, hi_1 = max(lo_1 - c1 * eta2, 0.0), hi_1 + c1 * eta2
        lo_2, hi_2 = max(lo_2 - c2 * eta2, 0.0), hi_2 + c2 * eta2

    lo_den = float(np.hypot(lo_1, lo_2))
    hi_den = float(np.hypot(hi_1, hi_2))
    upper = np.inf if lo_den <= 0 else hi_num / lo_den
    lower = 0.0 if hi_den <= 0 else lo_num / hi_den
    return Certificate(float(upper), float(lower), plug, mode, alpha)


def _model_error_gains(big, theta):
    """Operator norms of ``h -> [xi_theta, h]``, ``DF_h xi`` and ``D xi F_h``
    from ``L2(nu)`` on the model class of ``big`` to ``L2(nu)`` on fields."""
    from .polynomials import moment_matrix
    g = moment_matrix(big.alg, big.measure, big.scale, sub=big.beta_idx)
    G = np.kron(np.eye(big.n), g)
    w, U = np.linalg.eigh((G + G.T) / 2)
    Gih = (U / np.sqrt(np.clip(w, 1e-300, None))) @ U.T
    out = []
    for M in (big.L_theta(theta), big.M1_theta(theta), big.M2_theta(theta)):
        out.append(float(np.linalg.svd(M @ Gih, compute_uv=False)[0]) if M.size else 0.0)
    return tuple(out)


def _gauss_norm_bound(L: np.ndarray, cov: np.ndarray, alpha: float) -> float:
    """Valid ``1 - alpha`` upper bound for ``||L z||``, ``z ~ N(0, cov)``.

    Borell--TIS / Gaussian concentration: for a Gaussian vector ``W = L z`` with
    covariance ``V``, ``P(||W|| >= sqrt(tr V) + sqrt(2 ||V||_op log(1/alpha)))
    <= alpha`` because ``E||W|| <= sqrt(tr V)`` and ``w -> ||w||`` is
    1-Lipschitz.
    """
    if L.shape[0] == 0:
        return 0.0
    w, U = np.linalg.eigh((cov + cov.T) / 2)
    B = L @ (U * np.sqrt(np.clip(w, 0.0, None)))       # V = B B^T
    G = B.T @ B                                        # same nonzero spectrum
    tr = float(np.trace(G))
    op = float(max(np.linalg.eigvalsh((G + G.T) / 2).max(), 0.0))
    return float(np.sqrt(max(tr, 0.0)) + np.sqrt(2.0 * op * np.log(1.0 / alpha)))


def certify_subspace(prob, V: np.ndarray, fit, alpha: float = 0.05) -> Certificate:
    """Upper bound on ``sup_{theta in span(V), ||theta||=1} rho(theta; beta_true)``.

    Valid simultaneously over all subspaces, so ``V`` may be data-dependent.
    Uses ``sigma_max(A P) <= sigma_max(A_hat P) + ||Delta A P||_F`` with the
    Frobenius term maximised exactly over the confidence ellipsoid.
    """
    V = np.asarray(V, dtype=float)
    if V.ndim == 1:
        V = V[:, None]
    Vq, _ = np.linalg.qr(V)
    d = Vq.shape[1]
    if d == 1:
        return certify_direction(prob, Vq[:, 0], fit, alpha, mode="simultaneous")

    if fit.null_basis is not None and fit.null_basis.size:
        TAn = np.einsum("kpq,pd->kdq", prob.Tw, Vq).reshape(-1, prob.Q) @ fit.null_basis
        if np.linalg.norm(TAn) > 1e-10 * max(np.linalg.norm(prob.Tw), 1e-300):
            return Certificate(np.inf, 0.0, np.inf, "subspace", alpha)

    R = fit.ellipsoid_radius(alpha)
    S = fit.cov_sqrt
    # A(beta) V  and  M(beta) V  are linear in beta; flatten to vectors so that
    # the Frobenius norm is a Euclidean norm of a linear image of beta.
    TA = np.einsum("kpq,pd->kdq", prob.Tw, Vq).reshape(-1, prob.Q)
    TM = np.concatenate([np.einsum("kpq,pd->kdq", prob.T1w, Vq).reshape(-1, prob.Q),
                         np.einsum("kpq,pd->kdq", prob.T2w, Vq).reshape(-1, prob.Q)], axis=0)
    radA = max_norm_over_ball(np.zeros(TA.shape[0]), TA @ S, R)
    radM = max_norm_over_ball(np.zeros(TM.shape[0]), TM @ S, R)

    Ahat = prob.A(fit.beta) @ Vq
    Mhat = prob.M(fit.beta) @ Vq
    smax = float(np.linalg.svd(Ahat, compute_uv=False)[0])
    smin = float(np.linalg.svd(Mhat, compute_uv=False)[-1])
    hi_num = smax + radA
    lo_den = max(smin - radM, 0.0)
    plug = float(smax / max(smin, 1e-300))
    upper = np.inf if lo_den <= 0 else hi_num / lo_den
    return Certificate(float(upper), 0.0, plug, "subspace", alpha)


def refute_all(prob, fit, alpha: float = 0.05) -> float:
    """Lower confidence bound on ``min_{theta != 0} rho(theta; beta_true)``.

    If this exceeds ``delta`` we may assert, with confidence ``1 - alpha``, that
    the system has **no** ``delta``-approximate symmetry in the candidate class.
    """
    if fit.null_basis is not None and fit.null_basis.size:
        return 0.0
    R = fit.ellipsoid_radius(alpha)
    S = fit.cov_sqrt
    TA = prob.Tw.reshape(-1, prob.Q)
    TM = np.concatenate([prob.T1w.reshape(-1, prob.Q), prob.T2w.reshape(-1, prob.Q)], axis=0)
    radA = max_norm_over_ball(np.zeros(TA.shape[0]), TA @ S, R)
    radM = max_norm_over_ball(np.zeros(TM.shape[0]), TM @ S, R)
    Ahat, Mhat = prob.A(fit.beta), prob.M(fit.beta)
    smin_A = float(np.linalg.svd(Ahat, compute_uv=False)[-1])
    smax_M = float(np.linalg.svd(Mhat, compute_uv=False)[0])
    return float(max(smin_A - radA, 0.0) / (smax_M + radM))


def certified_dimension(prob, fit, delta: float, alpha: float = 0.05,
                        max_dim: int | None = None) -> tuple[int, np.ndarray]:
    """Largest ``d`` for which a ``d``-dimensional candidate subalgebra can be
    certified at tolerance ``delta``.

    The candidate nested subspaces are spanned by the generalised eigenvectors
    of ``(C(beta_hat), D(beta_hat))`` with the smallest plug-in defects.  These
    are data-dependent, which is exactly why the *simultaneous* certificate is
    needed: its validity does not depend on how the subspace was chosen.
    """
    _, Theta = prob.spectrum(fit.beta)
    P = prob.P if max_dim is None else min(max_dim, prob.P)
    best_d, best_V = 0, np.zeros((prob.P, 0))
    for d in range(1, P + 1):
        cert = certify_subspace(prob, Theta[:, :d], fit, alpha)
        if cert.upper <= delta:
            best_d, best_V = d, Theta[:, :d]
        else:
            break
    return best_d, best_V


def certify_direction_bootstrap(prob, theta: np.ndarray, fit, alpha: float = 0.05,
                                n_draw: int = 400, rng=None) -> Certificate:
    """Calibration-targeting alternative to the exact certificate.

    Draws ``beta* ~ N(beta_hat, Sigma_hat)`` and returns the ``1 - alpha``
    quantile of ``rho(theta; beta*)``.  This targets nominal coverage rather
    than guaranteeing it: it is first-order correct where ``rho`` is smooth, but
    it has no finite-sample validity, and none at all at ``rho = 0`` where the
    functional is not differentiable.  We include it only to measure how much of
    the exact certificate's width is the price of validity.
    """
    rng = np.random.default_rng(0) if rng is None else rng
    theta = np.asarray(theta, float)
    theta = theta / np.linalg.norm(theta)
    L, M1, M2 = prob.L_theta(theta), prob.M1_theta(theta), prob.M2_theta(theta)
    S = fit.cov_sqrt
    Z = rng.normal(size=(fit.beta.size, n_draw))
    B = fit.beta[:, None] + S @ Z
    num = np.linalg.norm(L @ B, axis=0)
    den = np.sqrt(np.linalg.norm(M1 @ B, axis=0) ** 2
                  + (np.linalg.norm(M2 @ B, axis=0) ** 2 if M2.size else 0.0))
    r = num / np.maximum(den, 1e-300)
    plug = float(np.linalg.norm(L @ fit.beta)
                 / max(np.hypot(np.linalg.norm(M1 @ fit.beta),
                                np.linalg.norm(M2 @ fit.beta) if M2.size else 0.0), 1e-300))
    return Certificate(float(np.quantile(r, 1 - alpha)),
                       float(np.quantile(r, alpha)), plug, "bootstrap", alpha)
