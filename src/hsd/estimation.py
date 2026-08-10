"""Estimating the vector field and the uncertainty of its coefficients.

We use the standard "measured derivative" design of data-driven model discovery
(as in SINDy): pairs ``(x_i, y_i)`` with ``y_i = F(x_i) + eps_i``.  The model is
linear in its coefficients, ``F_beta(x) = sum_a beta_a phi_a(x)`` over monomials,
so ordinary least squares gives an estimate ``beta_hat`` together with an
**exact** finite-sample confidence ellipsoid under Gaussian noise, and an
asymptotically valid one (sandwich form) otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

__all__ = ["Fit", "fit_ols", "simulate_design"]


@dataclass
class Fit:
    """Least-squares fit of a linear-in-parameters vector field.

    ``beta`` is flattened component-major, matching
    :meth:`DefectProblem.beta_of_field`.  ``cov`` is the estimated covariance of
    ``beta_hat``; ``dof`` is the residual degrees of freedom (``inf`` when the
    covariance is a sandwich estimate and only asymptotic validity is claimed).
    """

    beta: np.ndarray
    cov: np.ndarray
    dof: float
    sigma2: float
    n_samples: int
    exact_gaussian: bool = True
    cov_sqrt_: np.ndarray | None = None
    null_basis: np.ndarray | None = None   # unidentified directions of beta
    design_cond: float = 1.0

    @property
    def cov_sqrt(self) -> np.ndarray:
        if self.cov_sqrt_ is None:
            w, V = np.linalg.eigh((self.cov + self.cov.T) / 2)
            w = np.clip(w, 0.0, None)
            self.cov_sqrt_ = (V * np.sqrt(w)) @ V.T
        return self.cov_sqrt_

    def ellipsoid_radius(self, alpha: float) -> float:
        """``sqrt(q)`` such that ``||cov^{-1/2}(beta - beta_hat)|| <= sqrt(q)``
        is a ``1 - alpha`` confidence region for ``beta``.

        Exact (an ``F`` quantile) in the homoskedastic Gaussian model with an
        estimated noise level; a chi-square quantile in the asymptotic case.
        """
        Q = self.beta.size
        if self.exact_gaussian and np.isfinite(self.dof):
            return float(np.sqrt(Q * stats.f.ppf(1 - alpha, Q, self.dof)))
        return float(np.sqrt(stats.chi2.ppf(1 - alpha, Q)))


def fit_ols(prob, X: np.ndarray, Y: np.ndarray, robust: bool = False) -> Fit:
    """OLS fit of ``F`` on monomial features, with a confidence ellipsoid.

    Parameters
    ----------
    prob : :class:`~hsd.defect.DefectProblem` (supplies the monomial basis).
    X : ``(N, n)`` states.
    Y : ``(N, n)`` noisy right-hand sides.
    robust : if True use a heteroskedasticity-consistent (sandwich) covariance
        and drop the exact-Gaussian claim.
    """
    alg = prob.alg
    Phi = alg.features(X)[:, prob.beta_idx]           # (N, m)
    N, m = Phi.shape
    n = prob.n
    G = Phi.T @ Phi
    w, U = np.linalg.eigh((G + G.T) / 2)
    tol = max(N, 1.0) * np.finfo(float).eps * max(w.max(), 1e-300) * m
    ok = w > tol
    Ginv = (U[:, ok] / w[ok]) @ U[:, ok].T
    rank = int(ok.sum())
    null = None
    if rank < m:
        # Directions of beta the design cannot see at all: the certificate must
        # treat them as completely unknown rather than as known to be zero.
        null = np.kron(np.eye(n), U[:, ~ok])
    B = Ginv @ (Phi.T @ Y)                            # (m, n) coefficients
    resid = Y - Phi @ B                               # (N, n)
    dof = n * (N - rank)
    sigma2 = float((resid ** 2).sum() / max(dof, 1))
    beta = B.T.ravel()                                # component-major
    cond = float(w.max() / max(w[ok].min(), 1e-300)) if rank else np.inf
    if not robust:
        cov = np.kron(np.eye(n), sigma2 * Ginv)
        # component-major flattening: beta[i*m + a]  ->  block i is sigma2*Ginv
        return Fit(beta, cov, float(dof), sigma2, N, exact_gaussian=True,
                   null_basis=null, design_cond=cond)
    # Sandwich: Cov(vec) = (I ⊗ Ginv) Omega (I ⊗ Ginv), Omega built from
    # per-sample outer products of (resid ⊗ phi).
    Z = np.einsum("ni,na->nia", resid, Phi).reshape(N, n * m)
    Omega = Z.T @ Z
    Kmat = np.kron(np.eye(n), Ginv)
    cov = Kmat @ Omega @ Kmat
    return Fit(beta, cov, np.inf, sigma2, N, exact_gaussian=False,
               null_basis=null, design_cond=cond)


def lack_of_fit_pvalue(prob_small, prob_large, X: np.ndarray, Y: np.ndarray) -> float:
    """``F``-test of the smaller model class inside the larger one.

    A small p-value says the fitted class cannot represent the data-generating
    field, which invalidates any certificate computed inside that class.  Used
    as a mandatory pre-condition of certification.
    """
    n = prob_small.n
    Ps = prob_small.alg.features(X)[:, prob_small.beta_idx]
    Pl = prob_large.alg.features(X)[:, prob_large.beta_idx]
    N = X.shape[0]
    rs = Y - Ps @ np.linalg.lstsq(Ps, Y, rcond=None)[0]
    rl = Y - Pl @ np.linalg.lstsq(Pl, Y, rcond=None)[0]
    ms, ml = Ps.shape[1], Pl.shape[1]
    df1, df2 = n * (ml - ms), n * (N - ml)
    if df1 <= 0 or df2 <= 0:
        return 1.0
    num = ((rs ** 2).sum() - (rl ** 2).sum()) / df1
    den = (rl ** 2).sum() / df2
    if den <= 0:
        return 0.0
    return float(stats.f.sf(max(num / den, 0.0), df1, df2))


def simulate_design(sys_, rng, N: int, sigma: float, sampler="box", scale=1.0,
                    inner=0.0, noise="gauss", df: float = 4.0,
                    hetero: float = 0.0):
    """Draw a design ``X ~ mu`` and noisy right-hand sides ``Y``.

    ``sampler`` controls the *data* distribution ``mu`` and is deliberately
    allowed to differ from the *target* domain ``nu`` used to define the defect;
    that mismatch is the mechanism behind support-induced spurious symmetry.
    """
    n = sys_.n
    if sampler == "box":
        X = rng.uniform(-scale, scale, size=(N, n))
    elif sampler == "gauss":
        X = rng.normal(0.0, scale, size=(N, n))
    elif sampler in ("ball", "annulus"):
        V = rng.normal(size=(N, n))
        V /= np.linalg.norm(V, axis=1, keepdims=True)
        u = rng.uniform(size=N)
        r = (inner ** n + u * (scale ** n - inner ** n)) ** (1.0 / n)
        X = V * r[:, None]
    elif sampler == "halfbox":
        X = rng.uniform(-scale, scale, size=(N, n))
        X[:, 0] = np.abs(X[:, 0])
    elif sampler == "shell":                      # thin shell near |x| = scale
        V = rng.normal(size=(N, n))
        V /= np.linalg.norm(V, axis=1, keepdims=True)
        X = V * (scale * (1.0 + 0.02 * rng.normal(size=N)))[:, None]
    else:
        raise ValueError(sampler)
    Ytrue = sys_.rhs(X)
    if noise == "gauss":
        E = rng.normal(size=(N, n))
    elif noise == "t":
        E = rng.standard_t(df, size=(N, n)) / np.sqrt(df / (df - 2.0))
    elif noise == "laplace":
        E = rng.laplace(size=(N, n)) / np.sqrt(2.0)
    else:
        raise ValueError(noise)
    s = sigma * (1.0 + hetero * np.linalg.norm(X, axis=1, keepdims=True))
    return X, Ytrue + s * E


def fit_ols_scalar(prob, X: np.ndarray, y: np.ndarray, robust: bool = False) -> Fit:
    """OLS fit of a scalar basis-function model (used by :mod:`hsd.invariance`)."""
    Phi = prob.alg.features(X)[:, prob.beta_idx]
    N, m = Phi.shape
    G = Phi.T @ Phi
    w, U = np.linalg.eigh((G + G.T) / 2)
    tol = max(N, 1.0) * np.finfo(float).eps * max(w.max(), 1e-300) * m
    ok = w > tol
    Ginv = (U[:, ok] / w[ok]) @ U[:, ok].T
    rank = int(ok.sum())
    null = U[:, ~ok] if rank < m else None
    beta = Ginv @ (Phi.T @ y)
    resid = y - Phi @ beta
    dof = N - rank
    sigma2 = float((resid ** 2).sum() / max(dof, 1))
    cond = float(w.max() / max(w[ok].min(), 1e-300)) if rank else np.inf
    if not robust:
        return Fit(beta, sigma2 * Ginv, float(dof), sigma2, N, True, None, null, cond)
    Z = resid[:, None] * Phi
    cov = Ginv @ (Z.T @ Z) @ Ginv
    return Fit(beta, cov, np.inf, sigma2, N, False, None, null, cond)
