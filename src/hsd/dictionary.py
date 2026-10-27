"""Beyond polynomials: vector fields that are linear in a fixed dictionary.

The certificate needs only one structural fact: the defect is a ratio of norms
of linear images of the model coefficients.  That holds for any model of the
form ``F_beta(x) = sum_{i,a} beta_{i,a} phi_a(x) e_i`` with fixed, differentiable
dictionary functions ``phi_a`` -- trigonometric libraries for angles, radial
functions, any SINDy-style library -- and any finite candidate class of
generators.  For ``F = phi_a e_i``,

    DF xi  = (grad phi_a . xi) e_i ,        D xi F = phi_a (D xi) e_i ,

so both terms of the bracket are linear in ``beta`` pointwise, and their
``L2(nu)`` norms are quadratic forms whose matrices are integrals of products
of dictionary functions, gradients and generator components.

Here those integrals are computed by quadrature.  The rules provided are exact
for the libraries used in the paper: a uniform grid on the torus integrates
trigonometric polynomials of degree below the grid size exactly, and
Gauss--Legendre rules integrate polynomials exactly.  For integrands that are
smooth but not polynomial (radial functions on a disc) the Gauss--Legendre rule
in the radius converges exponentially and is used at a size where its error is
at the level of rounding; the test suite checks this.

The resulting :class:`DictionaryProblem` exposes the same interface as
:class:`hsd.defect.DefectProblem`, so every certificate, baseline and
diagnostic applies unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from .defect import DefectProblem, _metric_matrix, _sym_sqrt

__all__ = ["Dictionary", "Generator", "DictionaryProblem", "torus_rule",
           "gauss_legendre_rule", "disc_rule", "product_rule"]


@dataclass
class Dictionary:
    """Scalar dictionary functions with their gradients.

    ``funcs[a](X) -> (N,)`` and ``grads[a](X) -> (N, n)``.
    """
    funcs: Sequence[Callable[[np.ndarray], np.ndarray]]
    grads: Sequence[Callable[[np.ndarray], np.ndarray]]
    names: Sequence[str]

    @property
    def m(self) -> int:
        return len(self.funcs)

    def features(self, X: np.ndarray) -> np.ndarray:
        return np.stack([f(X) for f in self.funcs], axis=1)


@dataclass
class Generator:
    """A candidate generator: its field ``(N, n)`` and Jacobian ``(N, n, n)``."""
    field: Callable[[np.ndarray], np.ndarray]
    jac: Callable[[np.ndarray], np.ndarray]
    name: str


class DictionaryProblem(DefectProblem):
    """Exact bilinear forms of the defect for a dictionary model.

    Built by :meth:`build`; behaves like a :class:`DefectProblem` for every
    certificate and diagnostic.  ``rhs(beta, X)`` evaluates the model field.
    """

    dictionary: Dictionary = None
    generators: Sequence[Generator] = None

    @staticmethod
    def build(n: int, dictionary: Dictionary, generators: Sequence[Generator],
              nodes: np.ndarray, weights: np.ndarray,
              metric: np.ndarray | None = None,
              measure_name: str = "quadrature") -> "DictionaryProblem":
        nodes = np.asarray(nodes, float)
        w = np.asarray(weights, float)
        w = w / w.sum()
        Nq, m, P = nodes.shape[0], dictionary.m, len(generators)
        Phi = dictionary.features(nodes)                               # (Nq, m)
        Gr = np.stack([g(nodes) for g in dictionary.grads], axis=1)    # (Nq, m, n)
        Z = np.stack([g.field(nodes) for g in generators])             # (P, Nq, n)
        DZ = np.stack([g.jac(nodes) for g in generators])              # (P, Nq, n, n)
        dphi = np.einsum("kan,pkn->pka", Gr, Z)                        # (P, Nq, m)
        Q = n * m
        T1 = np.zeros((Nq, n, P, n, m))
        for i in range(n):
            T1[:, i, :, i, :] = np.transpose(dphi, (1, 0, 2))
        T2 = np.einsum("ka,pkji->kjpia", Phi, DZ)                      # (Nq, n, P, n, m)
        Wh = _sym_sqrt(_metric_matrix(metric, n))
        sw = np.sqrt(w)[:, None, None, None, None]
        T1w = np.einsum("jl,klpia->kjpia", Wh, T1 * sw).reshape(Nq * n, P, Q)
        T2w = np.einsum("jl,klpia->kjpia", Wh, T2 * sw).reshape(Nq * n, P, Q)
        # Compress the output index without changing any norm: every bracket
        # lies in the span of the columns of [T1w, T2w].
        K = Nq * n
        if K > 2 * P * Q:
            stack = np.concatenate([T1w.reshape(K, P * Q), T2w.reshape(K, P * Q)], axis=1)
            U, s, _ = np.linalg.svd(stack, full_matrices=False)
            U = U[:, s > 1e-13 * s.max()]
            T1w = np.einsum("kr,kpq->rpq", U, T1w)
            T2w = np.einsum("kr,kpq->rpq", U, T2w)
        Tw = T1w - T2w
        prob = DictionaryProblem(n, -1, -1, None, None,
                                 [g.name for g in generators], None, None,
                                 Tw, T1w, T2w, Tw, T1w, T2w, measure_name, 1.0,
                                 None if metric is None else _metric_matrix(metric, n))
        prob.dictionary = dictionary
        prob.generators = generators
        return prob

    def features(self, X: np.ndarray) -> np.ndarray:
        return self.dictionary.features(X)

    def rhs(self, beta: np.ndarray, X: np.ndarray) -> np.ndarray:
        """Model field ``F_beta`` at the states ``X`` (``(N, n)``)."""
        B = np.asarray(beta, float).reshape(self.n, self.dictionary.m)
        return self.features(X) @ B.T

    def beta_from_terms(self, terms: dict) -> np.ndarray:
        """Coefficient vector from ``{(component, dictionary name): value}``."""
        beta = np.zeros(self.Q)
        idx = {nm: a for a, nm in enumerate(self.dictionary.names)}
        for (i, nm), v in terms.items():
            beta[i * self.dictionary.m + idx[nm]] = float(v)
        return beta


# --------------------------------------------------------------- quadrature
def torus_rule(M: int) -> tuple[np.ndarray, np.ndarray]:
    """Uniform grid on ``[0, 2 pi)``: exact for trigonometric polynomials of
    degree below ``M``."""
    x = 2 * np.pi * np.arange(M) / M
    return x, np.full(M, 1.0 / M)


def gauss_legendre_rule(M: int, a: float, b: float) -> tuple[np.ndarray, np.ndarray]:
    """Gauss--Legendre on ``[a, b]`` for the uniform probability measure:
    exact for polynomials of degree at most ``2M - 1``."""
    t, w = np.polynomial.legendre.leggauss(M)
    return 0.5 * (b - a) * t + 0.5 * (b + a), w / 2.0


def product_rule(*rules) -> tuple[np.ndarray, np.ndarray]:
    """Tensor product of one-dimensional rules."""
    grids = np.meshgrid(*[r[0] for r in rules], indexing="ij")
    wgrid = np.meshgrid(*[r[1] for r in rules], indexing="ij")
    X = np.stack([g.ravel() for g in grids], axis=1)
    W = np.prod(np.stack([g.ravel() for g in wgrid], axis=1), axis=1)
    return X, W / W.sum()


def disc_rule(Mr: int, Mphi: int, r0: float = 0.0, r1: float = 1.0):
    """Uniform measure on the annulus ``r0 <= |x| <= r1`` in the plane:
    Gauss--Legendre in the radius with the area weight ``r``, uniform grid in
    the angle."""
    r, wr = gauss_legendre_rule(Mr, r0, r1)
    wr = wr * r
    phi, wphi = torus_rule(Mphi)
    R, PH = np.meshgrid(r, phi, indexing="ij")
    WR, WP = np.meshgrid(wr, wphi, indexing="ij")
    X = np.stack([(R * np.cos(PH)).ravel(), (R * np.sin(PH)).ravel()], axis=1)
    W = (WR * WP).ravel()
    return X, W / W.sum()
