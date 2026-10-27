"""Estimating the vector field and the uncertainty of its coefficients.

We use the standard "measured derivative" design of data-driven model discovery
(as in SINDy): pairs ``(x_i, y_i)`` with ``y_i = F(x_i) + eps_i``.  The model is
linear in its coefficients, ``F_beta(x) = sum_a beta_a phi_a(x)`` over monomials,
so ordinary least squares gives an estimate ``beta_hat`` together with an
**exact** finite-sample confidence ellipsoid under Gaussian noise, and
asymptotically valid ones otherwise:

``cov="ols"``      homoskedastic Gaussian errors; exact ``F`` ellipsoid.
``cov="hc0"``      White's heteroskedasticity-consistent sandwich; independent
                   rows, asymptotic in the number of rows.
``cov="cluster"``  cluster-robust sandwich over groups (e.g. trajectories),
                   in the jackknife (CR3) form, with the Hotelling-type ``F``
                   reference for the radius;
                   arbitrary dependence within a group, independence across
                   groups, asymptotic in the number of groups ``G``.  Its rank
                   is at most ``G``; directions it cannot see are treated as
                   unidentified, so ``G`` must exceed the number of
                   coefficients for the certificate to be finite.
``cov="hac"``      Newey--West (Bartlett) sandwich applied within each group
                   (time-ordered rows), with Andrews' AR(1) plug-in bandwidth
                   unless ``lags`` is given; asymptotic in the length of the
                   series.

Whatever the choice, the certificates of :mod:`hsd.certify` inherit exactly
the coverage of the resulting region and nothing more.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

__all__ = ["Fit", "fit_ols", "simulate_design", "region_covers",
           "lack_of_fit_pvalue"]


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
    n_clusters: int | None = None          # set for the cluster-robust region

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
        estimated noise level; for the cluster-robust estimate with ``G``
        clusters the small-sample reference ``Q (G-1)/(G-Q) F_{Q, G-Q}`` (the
        Hotelling form, which the chi-square approaches as ``G`` grows); a
        chi-square quantile in the other asymptotic cases.
        """
        Q = self.beta.size
        if self.exact_gaussian and np.isfinite(self.dof):
            return float(np.sqrt(Q * stats.f.ppf(1 - alpha, Q, self.dof)))
        G = self.n_clusters
        if G is not None and G > Q:
            return float(np.sqrt(Q * (G - 1) / (G - Q) * stats.f.ppf(1 - alpha, Q, G - Q)))
        return float(np.sqrt(stats.chi2.ppf(1 - alpha, Q)))


def fit_ols(prob, X: np.ndarray, Y: np.ndarray, robust: bool = False,
            cov: str | None = None, groups: np.ndarray | None = None,
            lags: int | None = None) -> Fit:
    """OLS fit of ``F`` on the model features, with a confidence ellipsoid.

    Parameters
    ----------
    prob : :class:`~hsd.defect.DefectProblem` or any object with ``features``.
    X : ``(N, n)`` states.
    Y : ``(N, n)`` noisy right-hand sides.
    robust : shorthand for ``cov="hc0"`` (kept for backward compatibility).
    cov : ``"ols"``, ``"hc0"``, ``"cluster"`` or ``"hac"`` (module docstring).
    groups : ``(N,)`` group labels for ``"cluster"`` and ``"hac"``; rows of a
        group must be in time order for ``"hac"``.
    lags : Bartlett truncation lag for ``"hac"`` (default: Andrews' rule).
    """
    cov = cov or ("hc0" if robust else "ols")
    Phi = _features(prob, X)                          # (N, m)
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
    if cov == "ols":
        V = np.kron(np.eye(n), sigma2 * Ginv)
        # component-major flattening: beta[i*m + a]  ->  block i is sigma2*Ginv
        return Fit(beta, V, float(dof), sigma2, N, exact_gaussian=True,
                   null_basis=null, design_cond=cond)
    # Sandwich forms: Cov(vec) = (I ⊗ Ginv) Omega (I ⊗ Ginv), Omega built from
    # the per-row scores z_t = resid_t ⊗ phi_t.
    Z = np.einsum("ni,na->nia", resid, Phi).reshape(N, n * m)
    if cov == "hc0":
        Omega = Z.T @ Z
    elif cov == "cluster":
        if groups is None:
            raise ValueError("cluster-robust covariance needs group labels")
        # Jackknife (CR3) form: each cluster's residuals are inflated by
        # (I - H_gg)^{-1}, which removes the downward bias that high-leverage
        # clusters cause in the plain sandwich; recommended when clusters are
        # few relative to the number of coefficients.
        labels = np.unique(groups)
        g = labels.size
        S = np.zeros((g, n * m))
        for j, lab in enumerate(labels):
            idx = groups == lab
            Pg = Phi[idx]
            Hg = Pg @ Ginv @ Pg.T
            rg = np.linalg.lstsq(np.eye(Hg.shape[0]) - Hg, resid[idx], rcond=None)[0]
            S[j] = np.einsum("ti,ta->ia", rg, Pg).reshape(-1)
        Omega = (S.T @ S) * ((g - 1) / max(g, 1))
    elif cov == "hac":
        if groups is None:
            groups = np.zeros(N, dtype=int)
        Omega = np.zeros((n * m, n * m))
        L = lags if lags is not None else _andrews_lag(resid, groups)
        for gval in np.unique(groups):
            Zg = Z[groups == gval]
            Omega += Zg.T @ Zg
            for l in range(1, min(L, Zg.shape[0] - 1) + 1):
                wl = 1.0 - l / (L + 1.0)
                Gl = Zg[l:].T @ Zg[:-l]
                Omega += wl * (Gl + Gl.T)
    else:
        raise ValueError(cov)
    Kmat = np.kron(np.eye(n), Ginv)
    V = Kmat @ Omega @ Kmat
    V = (V + V.T) / 2
    # A sandwich estimate can be rank deficient (e.g. fewer clusters than
    # coefficients); directions outside its range are unidentified, never
    # "known exactly".
    wv, Uv = np.linalg.eigh(V)
    small = wv <= 1e-12 * max(wv.max(), 1e-300)
    if small.any():
        # The range of V excludes the design's null directions as well, so its
        # null space alone is the complete set of unidentified directions.
        null = Uv[:, small]
    return Fit(beta, V, np.inf, sigma2, N, exact_gaussian=False,
               null_basis=null, design_cond=cond,
               n_clusters=int(np.unique(groups).size) if cov == "cluster" else None)


def _features(prob, X: np.ndarray) -> np.ndarray:
    if hasattr(prob, "features"):
        return prob.features(X)
    return prob.alg.features(X)[:, prob.beta_idx]


def _andrews_lag(resid: np.ndarray, groups: np.ndarray) -> int:
    """Andrews (1991) AR(1) plug-in bandwidth for the Bartlett kernel, from the
    pooled lag-one autocorrelation of the residuals within groups."""
    num = den = 0.0
    T = []
    for gval in np.unique(groups):
        r = resid[groups == gval]
        T.append(r.shape[0])
        if r.shape[0] > 1:
            num += float(np.sum(r[1:] * r[:-1]))
            den += float(np.sum(r[:-1] ** 2))
    rho = float(np.clip(num / max(den, 1e-300), -0.97, 0.97))
    a1 = 4.0 * rho ** 2 / ((1.0 - rho) ** 2 * (1.0 + rho) ** 2)
    Tbar = float(np.mean(T))
    return int(max(1, np.ceil(1.1447 * (a1 * Tbar) ** (1.0 / 3.0))))


def region_covers(fit: Fit, beta_true: np.ndarray, alpha: float) -> bool:
    """Whether the ``1 - alpha`` coefficient region of ``fit`` contains
    ``beta_true`` -- the primitive event on which every certificate rests."""
    d = fit.beta - beta_true
    if fit.null_basis is not None and fit.null_basis.size:
        # Unidentified directions are unconstrained: project them out.
        Qn, _ = np.linalg.qr(fit.null_basis)
        d = d - Qn @ (Qn.T @ d)
    w, U = np.linalg.eigh((fit.cov + fit.cov.T) / 2)
    z = U.T @ d
    pos = w > 1e-12 * max(w.max(), 1e-300)
    if np.any(np.abs(z[~pos]) > 1e-9 * max(np.linalg.norm(d), 1e-300)):
        return False
    stat = float(np.sum(z[pos] ** 2 / w[pos]))
    return stat <= fit.ellipsoid_radius(alpha) ** 2


def lack_of_fit_pvalue(prob_small, prob_large, X: np.ndarray, Y: np.ndarray) -> float:
    """``F``-test of the smaller model class inside the larger one.

    A small p-value says the fitted class cannot represent the data-generating
    field, which invalidates any certificate computed inside that class.  Used
    as a mandatory pre-condition of certification.
    """
    n = prob_small.n
    Ps = _features(prob_small, X)
    Pl = _features(prob_large, X)
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
