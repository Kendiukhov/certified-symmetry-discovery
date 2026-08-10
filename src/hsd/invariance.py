"""Symmetries of a *learned function*, not of a dynamical system.

Otto et al. (2025) treat both cases with the same operator.  For a scalar model
``f: R^n -> R`` and the trivial action on the output, the Lie derivative is
``L_xi f = grad f . xi``, so ``xi`` generates an invariance exactly when the
gradient of ``f`` is everywhere orthogonal to the generator.  The scale-free
defect is the root-mean-square cosine of the angle between them,

    rho(theta)^2 = || grad f . xi ||^2_{L2(nu)}
                   / || |grad f| |xi| ||^2_{L2(nu)}   in [0, 1],

which is again a ratio of squared norms of linear maps of the model
coefficients, so the certificates of :mod:`hsd.certify` apply verbatim to a
basis-function regression model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.linalg as sla

from .polynomials import PolyAlgebra, gram_sqrt, moment_matrix, poly_algebra

__all__ = ["InvarianceProblem"]


@dataclass
class InvarianceProblem:
    """Exact tensors for invariance of a scalar polynomial model.

    Presents the same interface as :class:`~hsd.defect.DefectProblem` so that
    every certificate and baseline works unchanged.
    """

    n: int
    deg_F: int
    deg_xi: int
    alg: PolyAlgebra
    gen_basis: np.ndarray
    gen_names: list[str]
    beta_idx: np.ndarray
    res_idx: np.ndarray
    Tnum: np.ndarray
    Tden: np.ndarray
    Tw: np.ndarray
    T1w: np.ndarray
    T2w: np.ndarray
    measure: str
    scale: float

    @staticmethod
    def build(n: int, deg_F: int, generators: str = "linear",
              measure: str = "box", scale: float = 1.0, inner: float = 0.0):
        from .defect import affine_generator_basis, linear_generator_basis
        deg_xi = 1
        deg_res = deg_F + deg_xi - 1
        alg = poly_algebra(n, max(deg_F, deg_res))
        gen, names = (linear_generator_basis(n, alg) if generators == "linear"
                      else affine_generator_basis(n, alg))
        beta_idx = alg.sub_index(deg_F)
        res_idx = alg.sub_index(deg_res)
        P, Q = len(gen), len(beta_idx)

        # numerator: grad f . xi   (one polynomial)
        Tnum = np.zeros((len(res_idx), P, Q))
        # denominator blocks: (d_k f) * xi_j for all (k, j)
        Tden = np.zeros((n * n * len(res_idx), P, Q))
        for q, a in enumerate(beta_idx):
            f = np.zeros(alg.m)
            f[a] = 1.0
            grads = [_diff(alg, f, k) for k in range(n)]
            for p in range(P):
                xi = gen[p]
                acc = np.zeros(alg.m)
                for k in range(n):
                    prod = _mul(alg, grads[k], xi[k])
                    acc += prod
                    for j in range(n):
                        blk = (k * n + j) * len(res_idx)
                        Tden[blk:blk + len(res_idx), p, q] = _mul(alg, grads[k], xi[j])[res_idx]
                Tnum[:, p, q] = acc[res_idx]

        g = moment_matrix(alg, measure=measure, scale=scale, inner=inner, sub=res_idx)
        gh = gram_sqrt(g)
        Tw = np.einsum("ij,jpq->ipq", gh, Tnum)
        T1w = np.einsum("ij,jpq->ipq", np.kron(np.eye(n * n), gh), Tden)
        T2w = np.zeros((0, P, Q))
        return InvarianceProblem(n, deg_F, deg_xi, alg, gen, names, beta_idx,
                                 res_idx, Tnum, Tden, Tw, T1w, T2w, measure, scale)

    # ---- the DefectProblem interface used by certify.py / baselines.py ----
    @property
    def P(self) -> int:
        return self.Tw.shape[1]

    @property
    def Q(self) -> int:
        return self.Tw.shape[2]

    def A(self, beta):
        return self.Tw @ beta

    def M(self, beta):
        return self.T1w @ beta

    def L_theta(self, theta):
        return np.einsum("kpq,p->kq", self.Tw, theta)

    def M_theta(self, theta):
        return np.einsum("kpq,p->kq", self.T1w, theta)

    def M1_theta(self, theta):
        return self.M_theta(theta)

    def M2_theta(self, theta):
        return np.zeros((0, self.Q))

    def C(self, beta):
        A = self.A(beta)
        return A.T @ A

    def D(self, beta):
        M = self.M(beta)
        return M.T @ M

    def rho(self, theta, beta):
        num = np.linalg.norm(self.A(beta) @ theta)
        den = np.linalg.norm(self.M(beta) @ theta)
        return float(num / den) if den > 0 else np.inf

    def spectrum(self, beta, jitter: float = 1e-12):
        C, D = self.C(beta), self.D(beta)
        D = D + jitter * np.trace(D) / D.shape[0] * np.eye(D.shape[0])
        w, V = sla.eigh((C + C.T) / 2, (D + D.T) / 2)
        w = np.clip(w, 0.0, None)
        o = np.argsort(w)
        return np.sqrt(w[o]), V[:, o]

    def beta_of_function(self, coefs: np.ndarray) -> np.ndarray:
        return coefs[self.beta_idx]

    def empirical(self, X: np.ndarray):
        """Same tensors with ``L2`` norms replaced by empirical averages."""
        Phi = self.alg.features(X)[:, self.res_idx]
        gh = gram_sqrt((Phi.T @ Phi) / X.shape[0])
        n = self.n
        return InvarianceProblem(
            self.n, self.deg_F, self.deg_xi, self.alg, self.gen_basis,
            self.gen_names, self.beta_idx, self.res_idx, self.Tnum, self.Tden,
            np.einsum("ij,jpq->ipq", gh, self.Tnum),
            np.einsum("ij,jpq->ipq", np.kron(np.eye(n * n), gh), self.Tden),
            self.T2w, "empirical", self.scale)


def _diff(alg, p, k):
    out = np.zeros(alg.m)
    nz = np.flatnonzero(p)
    if nz.size:
        np.add.at(out, alg.didx[nz, k], alg.dcoef[nz, k] * p[nz])
    return out


def _mul(alg, p, q):
    out = np.zeros(alg.m)
    nzp, nzq = np.flatnonzero(p), np.flatnonzero(q)
    for i in nzp:
        idx = alg.mul[i, nzq]
        np.add.at(out, idx, p[i] * q[nzq])
    return out
