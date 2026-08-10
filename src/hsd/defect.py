"""The relative equivariance defect and its exact bilinear structure.

For a polynomial vector field ``F_beta`` and a candidate generator
``xi_theta = sum_p theta_p zeta_p`` we measure how far ``xi_theta`` is from
generating a symmetry of ``x' = F(x)`` by the **relative equivariance defect**

    rho(theta; beta)^2 = || [xi_theta, F_beta] ||^2_{L2(nu)}
                         / ( || DF_beta xi_theta ||^2_{L2(nu)}
                             + || D xi_theta F_beta ||^2_{L2(nu)} ).

``rho = 0`` exactly when ``xi_theta`` generates a symmetry on the support of the
target measure ``nu``.  The functional is invariant under rescaling of
``xi_theta``, rescaling of ``F``, and any invertible change of basis of the
candidate space -- so a tolerance stated in terms of ``rho`` is meaningful
across systems, unlike a raw singular value of the Lie-derivative operator.

Both the numerator and the denominator are *squared norms of linear maps of the
model coefficients* ``beta``:

    numerator   = || Lt(theta) beta ||^2 ,
    denominator = || Mt(theta) beta ||^2 ,

which is what makes exact confidence propagation possible (see ``certify.py``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.linalg as sla

from .polynomials import (PolyAlgebra, gram_sqrt, jac_times, moment_matrix,
                          poly_algebra)

__all__ = ["DefectProblem", "affine_generator_basis", "linear_generator_basis"]


def linear_generator_basis(n: int, alg: PolyAlgebra) -> tuple[np.ndarray, list[str]]:
    """Basis of linear vector fields ``xi(x) = S x`` (the Lie algebra gl(n))."""
    basis, names = [], []
    for i in range(n):
        for j in range(n):
            Z = np.zeros((n, alg.m))
            e = [0] * n
            e[j] = 1
            Z[i, alg.index[tuple(e)]] = 1.0
            basis.append(Z)
            names.append(f"x{j}*d/dx{i}")
    return np.array(basis), names


def affine_generator_basis(n: int, alg: PolyAlgebra) -> tuple[np.ndarray, list[str]]:
    """Basis of affine vector fields ``xi(x) = S x + b`` (the Lie algebra of the
    affine group ``GL(n) x R^n``)."""
    basis, names = linear_generator_basis(n, alg)
    basis = list(basis)
    for i in range(n):
        Z = np.zeros((n, alg.m))
        Z[i, alg.index[tuple([0] * n)]] = 1.0
        basis.append(Z)
        names.append(f"d/dx{i}")
    return np.array(basis), names


@dataclass
class DefectProblem:
    """Pre-computed exact tensors for one (state dimension, degree, candidate
    class, target measure) configuration.

    Attributes
    ----------
    T : ``(K, P, Q)`` bracket tensor, ``coef([xi_theta, F_beta]) = T . theta . beta``.
    T1, T2 : the two individual terms ``DF xi`` and ``D xi F``.
    Tw, T1w, T2w : the same tensors after multiplication by the square root of
        the residual Gram matrix, so that Euclidean norms of the contracted
        vectors equal ``L2(nu)`` norms of the corresponding vector fields.
    """

    n: int
    deg_F: int
    deg_xi: int
    alg: PolyAlgebra
    gen_basis: np.ndarray
    gen_names: list[str]
    beta_idx: np.ndarray          # monomial indices used by F
    res_idx: np.ndarray           # monomial indices spanned by the residual
    T: np.ndarray
    T1: np.ndarray
    T2: np.ndarray
    Tw: np.ndarray
    T1w: np.ndarray
    T2w: np.ndarray
    measure: str
    scale: float
    _Gxi: np.ndarray | None = None
    _Gdxi: np.ndarray | None = None

    # ---------------------------------------------------------------- build
    @staticmethod
    def build(n: int, deg_F: int, generators: str = "affine",
              measure: str = "box", scale: float = 1.0, inner: float = 0.0,
              deg_xi: int | None = None) -> "DefectProblem":
        deg_xi = {"linear": 1, "affine": 1, "quadratic": 2}[generators] if deg_xi is None else deg_xi
        deg_res = deg_F + deg_xi - 1
        alg = poly_algebra(n, max(deg_F, deg_xi, deg_res))
        if generators == "linear":
            gen, names = linear_generator_basis(n, alg)
        elif generators == "affine":
            gen, names = affine_generator_basis(n, alg)
        elif generators == "quadratic":
            gen, names = _quadratic_generator_basis(n, alg)
        else:
            raise ValueError(generators)

        beta_idx = alg.sub_index(deg_F)
        res_idx = alg.sub_index(deg_res)
        P, Q, K = len(gen), n * len(beta_idx), n * len(res_idx)

        T = np.zeros((K, P, Q))
        T1 = np.zeros((K, P, Q))
        T2 = np.zeros((K, P, Q))
        for p in range(P):
            xi = gen[p]
            for comp in range(n):
                for qa, a in enumerate(beta_idx):
                    q = comp * len(beta_idx) + qa
                    F = np.zeros((n, alg.m))
                    F[comp, a] = 1.0
                    t1 = jac_times(alg, F, xi)      # DF xi
                    t2 = jac_times(alg, xi, F)      # D xi F
                    T1[:, p, q] = t1[:, res_idx].ravel()
                    T2[:, p, q] = t2[:, res_idx].ravel()
        T = T1 - T2

        g = moment_matrix(alg, measure=measure, scale=scale, inner=inner, sub=res_idx)
        gh = gram_sqrt(g)
        W = np.kron(np.eye(n), gh)
        Tw = np.einsum("ij,jpq->ipq", W, T)
        T1w = np.einsum("ij,jpq->ipq", W, T1)
        T2w = np.einsum("ij,jpq->ipq", W, T2)
        return DefectProblem(n, deg_F, deg_xi, alg, gen, names, beta_idx, res_idx,
                             T, T1, T2, Tw, T1w, T2w, measure, scale)

    def with_gram(self, g: np.ndarray, tag: str = "empirical") -> "DefectProblem":
        """Same tensors, re-weighted by a different residual Gram matrix.

        Used to build the *data-measure* version of the defect: replacing the
        exact target-domain moments by the empirical moments of the observed
        states reproduces what a practitioner computes when the sample is taken
        to stand in for the domain of interest.
        """
        W = np.kron(np.eye(self.n), gram_sqrt(g))
        return DefectProblem(self.n, self.deg_F, self.deg_xi, self.alg,
                             self.gen_basis, self.gen_names, self.beta_idx,
                             self.res_idx, self.T, self.T1, self.T2,
                             np.einsum("ij,jpq->ipq", W, self.T),
                             np.einsum("ij,jpq->ipq", W, self.T1),
                             np.einsum("ij,jpq->ipq", W, self.T2),
                             tag, self.scale)

    def empirical(self, X: np.ndarray) -> "DefectProblem":
        """Version of the problem whose ``L2`` norms are empirical averages over
        the observed states ``X``."""
        Phi = self.alg.features(X)[:, self.res_idx]
        return self.with_gram((Phi.T @ Phi) / X.shape[0])

    # ------------------------------------------------------------- geometry
    @property
    def P(self) -> int:
        return self.T.shape[1]

    @property
    def Q(self) -> int:
        return self.T.shape[2]

    def beta_of_field(self, F: np.ndarray) -> np.ndarray:
        """Flatten a vector field's coefficients into the ``beta`` vector."""
        return F[:, self.beta_idx].ravel()

    def field_of_beta(self, beta: np.ndarray) -> np.ndarray:
        F = np.zeros((self.n, self.alg.m))
        F[:, self.beta_idx] = beta.reshape(self.n, -1)
        return F

    # --------------------------------------------------------------- defect
    def A(self, beta: np.ndarray) -> np.ndarray:
        """``(K, P)`` matrix with columns ``L2(nu)``-weighted brackets ``[zeta_p, F_beta]``."""
        return self.Tw @ beta

    def M(self, beta: np.ndarray) -> np.ndarray:
        """``(2K, P)`` stacked normalisation matrix."""
        return np.vstack([self.T1w @ beta, self.T2w @ beta])

    def L_theta(self, theta: np.ndarray) -> np.ndarray:
        """``(K, Q)`` map ``beta -> `` weighted bracket coefficients."""
        return np.einsum("kpq,p->kq", self.Tw, theta)

    def M_theta(self, theta: np.ndarray) -> np.ndarray:
        """``(2K, Q)`` map ``beta -> `` weighted normalisation coefficients."""
        return np.vstack([np.einsum("kpq,p->kq", self.T1w, theta),
                          np.einsum("kpq,p->kq", self.T2w, theta)])

    def M1_theta(self, theta: np.ndarray) -> np.ndarray:
        """``beta -> DF_beta xi_theta`` (weighted)."""
        return np.einsum("kpq,p->kq", self.T1w, theta)

    def M2_theta(self, theta: np.ndarray) -> np.ndarray:
        """``beta -> D xi_theta F_beta`` (weighted)."""
        return np.einsum("kpq,p->kq", self.T2w, theta)

    def generator_norms(self, theta: np.ndarray) -> tuple[float, float]:
        """``(||xi_theta||_{L2(nu)}, ||D xi_theta||_{F, L2(nu)})``.

        These enter the sensitivity analysis for model error: a perturbation
        ``h`` of the vector field changes the bracket by ``Dh xi - D xi h``,
        whose norm is bounded by ``sup|Dh| ||xi|| + sup|h| ||D xi||``.
        """
        if self._Gxi is None:
            self._build_generator_grams()
        t = np.asarray(theta, float)
        return (float(np.sqrt(max(t @ self._Gxi @ t, 0.0))),
                float(np.sqrt(max(t @ self._Gdxi @ t, 0.0))))

    def _build_generator_grams(self) -> None:
        alg = self.alg
        g = moment_matrix(alg, self.measure, self.scale)
        P, n = self.P, self.n
        Gxi = np.zeros((P, P))
        Gdxi = np.zeros((P, P))
        D = [np.stack([_deriv_field(alg, self.gen_basis[p], k) for k in range(n)])
             for p in range(P)]
        for p in range(P):
            for q in range(P):
                Gxi[p, q] = np.einsum("im,mn,in->", self.gen_basis[p], g, self.gen_basis[q])
                Gdxi[p, q] = np.einsum("kim,mn,kin->", D[p], g, D[q])
        self._Gxi, self._Gdxi = Gxi, Gdxi

    def C(self, beta: np.ndarray) -> np.ndarray:
        A = self.A(beta)
        return A.T @ A

    def D(self, beta: np.ndarray) -> np.ndarray:
        M = self.M(beta)
        return M.T @ M

    def rho(self, theta: np.ndarray, beta: np.ndarray) -> float:
        """Relative equivariance defect of ``xi_theta`` for the field ``F_beta``."""
        num = np.linalg.norm(self.A(beta) @ theta)
        den = np.linalg.norm(self.M(beta) @ theta)
        return float(num / den) if den > 0 else np.inf

    def spectrum(self, beta: np.ndarray, jitter: float = 1e-12):
        """Generalised eigenpairs of ``(C, D)``: squared defects and directions.

        Returns ``(rho_sorted, Theta)`` where ``Theta[:, k]`` is the ``k``-th
        candidate generator, ``D``-normalised, with defect ``rho_sorted[k]``.
        """
        C, D = self.C(beta), self.D(beta)
        D = D + jitter * np.trace(D) / D.shape[0] * np.eye(D.shape[0])
        w, V = sla.eigh((C + C.T) / 2, (D + D.T) / 2)
        w = np.clip(w, 0.0, None)
        order = np.argsort(w)
        return np.sqrt(w[order]), V[:, order]


def _quadratic_generator_basis(n: int, alg: PolyAlgebra):
    """All polynomial vector fields of degree <= 2 (used for large-P studies)."""
    basis, names = [], []
    sub = alg.sub_index(2)
    for i in range(n):
        for a in sub:
            Z = np.zeros((n, alg.m))
            Z[i, a] = 1.0
            basis.append(Z)
            names.append(f"x^{alg.basis[a]}*d/dx{i}")
    return np.array(basis), names


def _deriv_field(alg, V: np.ndarray, k: int) -> np.ndarray:
    """Column ``k`` of the Jacobian of a vector field, as a vector field."""
    out = np.zeros_like(V)
    for i in range(alg.n):
        nz = np.flatnonzero(V[i])
        if nz.size:
            np.add.at(out[i], alg.didx[nz, k], alg.dcoef[nz, k] * V[i, nz])
    return out
