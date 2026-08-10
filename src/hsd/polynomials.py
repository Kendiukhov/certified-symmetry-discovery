"""Exact polynomial vector-field algebra on R^n.

A scalar polynomial of degree <= D in n variables is stored as a coefficient
vector over the graded-lexicographic monomial basis returned by
:func:`monomial_basis`.  A polynomial vector field is an ``(n, m)`` array whose
row ``i`` holds the coefficients of its ``i``-th component.

Everything in this module is exact linear algebra on coefficient arrays: no
sampling and no quadrature.  Integrals against the target measure enter only
through the monomial moment matrix built in :func:`moment_matrix`.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import product

import numpy as np

__all__ = [
    "monomial_basis",
    "PolyAlgebra",
    "bracket",
    "jac_times",
    "moment_matrix",
    "gram_sqrt",
]


def monomial_basis(n: int, deg: int) -> list[tuple[int, ...]]:
    """All exponent tuples ``a`` with ``|a| <= deg``, ordered by total degree.

    The ordering is deterministic (total degree, then lexicographic) so that
    coefficient vectors can be compared across calls.
    """
    out: list[tuple[int, ...]] = []
    for total in range(deg + 1):
        for a in product(range(total + 1), repeat=n):
            if sum(a) == total:
                out.append(tuple(a))
    out.sort(key=lambda a: (sum(a),) + tuple(-x for x in a))
    return out


@dataclass(frozen=True)
class PolyAlgebra:
    """Multiplication / differentiation tables for polynomials of degree <= D.

    Attributes
    ----------
    n : number of variables.
    deg : maximum total degree carried by the algebra.
    basis : list of exponent tuples.
    index : mapping exponent tuple -> position in ``basis``.
    mul : ``mul[i, j]`` is the index of ``basis[i] + basis[j]`` or ``-1`` when
        the product exceeds ``deg`` (which never happens in our use because we
        always instantiate the algebra at a degree large enough to be closed).
    dcoef, didx : ``d/dx_k basis[i] = dcoef[i, k] * basis[didx[i, k]]``; the
        coefficient is ``0`` when the derivative vanishes.
    """

    n: int
    deg: int
    basis: tuple[tuple[int, ...], ...]
    index: dict[tuple[int, ...], int]
    mul: np.ndarray
    dcoef: np.ndarray
    didx: np.ndarray

    @property
    def m(self) -> int:
        return len(self.basis)

    def degrees(self) -> np.ndarray:
        return np.array([sum(a) for a in self.basis], dtype=int)

    def sub_index(self, deg: int) -> np.ndarray:
        """Indices of the monomials of total degree <= ``deg``."""
        return np.flatnonzero(self.degrees() <= deg)

    def eval(self, coefs: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Evaluate a vector field ``coefs`` of shape ``(n, m)`` at points ``x``.

        ``x`` has shape ``(N, n)``; the result has shape ``(N, n)``.
        """
        return self.features(x) @ coefs.T

    def features(self, x: np.ndarray) -> np.ndarray:
        """Monomial feature matrix ``Phi`` of shape ``(N, m)``."""
        x = np.atleast_2d(x)
        exps = np.array(self.basis, dtype=int)  # (m, n)
        # (N, m): prod_k x[:, k] ** exps[j, k]
        return np.prod(x[:, None, :] ** exps[None, :, :], axis=2)


@lru_cache(maxsize=None)
def poly_algebra(n: int, deg: int) -> PolyAlgebra:
    basis = monomial_basis(n, deg)
    index = {a: i for i, a in enumerate(basis)}
    m = len(basis)
    mul = -np.ones((m, m), dtype=int)
    for i, a in enumerate(basis):
        for j, b in enumerate(basis):
            c = tuple(ai + bi for ai, bi in zip(a, b))
            if sum(c) <= deg:
                mul[i, j] = index[c]
    dcoef = np.zeros((m, n))
    didx = np.zeros((m, n), dtype=int)
    for i, a in enumerate(basis):
        for k in range(n):
            if a[k] > 0:
                b = list(a)
                b[k] -= 1
                dcoef[i, k] = a[k]
                didx[i, k] = index[tuple(b)]
    return PolyAlgebra(n, deg, tuple(basis), index, mul, dcoef, didx)


def _mul_poly(alg: PolyAlgebra, p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Product of two scalar polynomials given as coefficient vectors."""
    out = np.zeros(alg.m)
    nzp = np.flatnonzero(p)
    nzq = np.flatnonzero(q)
    for i in nzp:
        idx = alg.mul[i, nzq]
        if np.any(idx < 0):
            raise ValueError("product exceeds the algebra degree")
        np.add.at(out, idx, p[i] * q[nzq])
    return out


def _diff_poly(alg: PolyAlgebra, p: np.ndarray, k: int) -> np.ndarray:
    out = np.zeros(alg.m)
    nz = np.flatnonzero(p)
    if nz.size:
        np.add.at(out, alg.didx[nz, k], alg.dcoef[nz, k] * p[nz])
    return out


def jac_times(alg: PolyAlgebra, V: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Return the vector field ``x -> DV(x) W(x)``.

    ``V`` and ``W`` are ``(n, m)`` coefficient arrays over ``alg``.
    """
    n = alg.n
    out = np.zeros((n, alg.m))
    for i in range(n):
        for j in range(n):
            dv = _diff_poly(alg, V[i], j)
            if np.any(dv):
                out[i] += _mul_poly(alg, dv, W[j])
    return out


def bracket(alg: PolyAlgebra, xi: np.ndarray, F: np.ndarray) -> np.ndarray:
    """Lie bracket ``[xi, F] = DF xi - D xi F`` of two polynomial vector fields.

    This is the Lie derivative ``L_xi F`` of Otto et al. (2025, Eq. 16) for a
    dynamical system ``x' = F(x)`` under the flow of ``xi``.
    """
    return jac_times(alg, F, xi) - jac_times(alg, xi, F)


# --------------------------------------------------------------------------
# Moments of the target measure
# --------------------------------------------------------------------------

def _radial_moment(k: np.ndarray, n: int, r0: float, r1: float) -> np.ndarray:
    """``E[r^k]`` for a radius drawn with density proportional to ``r^{n-1}``
    on ``[r0, r1]`` (uniform measure on a ball or annulus)."""
    num = (r1 ** (k + n) - r0 ** (k + n)) / (k + n)
    den = (r1 ** n - r0 ** n) / n
    return num / den


def moment_matrix(alg: PolyAlgebra, measure: str = "box", scale: float = 1.0,
                  inner: float = 0.0, sub: np.ndarray | None = None) -> np.ndarray:
    """Monomial Gram matrix ``g[a, b] = int x^a x^b d nu(x)``.

    ``measure`` is one of

    ``"box"``      uniform on ``[-scale, scale]^n``,
    ``"gauss"``    ``N(0, scale^2 I_n)``,
    ``"ball"``     uniform on the Euclidean ball of radius ``scale``,
    ``"annulus"``  uniform on ``{inner <= |x| <= scale}``.

    All four have closed-form monomial moments, so the Gram matrix carries **no
    Monte-Carlo error**: the geometry of the target domain is exact and only the
    model coefficients are statistically uncertain.  ``sub`` optionally
    restricts to a subset of basis indices.
    """
    idx = np.arange(alg.m) if sub is None else np.asarray(sub)
    exps = np.array([alg.basis[i] for i in idx], dtype=int)
    s = exps[:, None, :] + exps[None, :, :]  # (k, k, n) summed exponents
    n = alg.n
    even = np.all(s % 2 == 0, axis=2)
    if measure == "box":
        # int_{-L}^{L} t^p dt / (2L) = L^p / (p+1) for even p, else 0.
        val = np.where(s % 2 == 0, scale ** s / (s + 1.0), 0.0)
        return np.prod(val, axis=2)
    if measure == "gauss":
        # E[t^p] = scale^p (p-1)!! for even p, else 0.
        def dfact(p):
            out = np.ones_like(p, dtype=float)
            for k in range(3, int(p.max()) + 1, 2):
                out = np.where(p >= k, out * k, out)
            return out
        val = np.where(s % 2 == 0, scale ** s * dfact(s), 0.0)
        return np.prod(val, axis=2)
    if measure in ("ball", "annulus"):
        # Factor into a radial part and a uniform-on-the-sphere angular part:
        #   E[x^a] = E[r^{|a|}] * E[u^a],
        #   E[u^a] = prod_i Gamma((a_i+1)/2) * Gamma(n/2)
        #            / (pi^{n/2} Gamma((n+|a|)/2)).
        from scipy.special import gammaln
        tot = s.sum(axis=2)
        log_ang = (gammaln((s + 1.0) / 2.0).sum(axis=2) + gammaln(n / 2.0)
                   - (n / 2.0) * np.log(np.pi) - gammaln((n + tot) / 2.0))
        r0 = 0.0 if measure == "ball" else float(inner)
        rad = _radial_moment(tot.astype(float), n, r0, float(scale))
        return np.where(even, np.exp(log_ang) * rad, 0.0)
    raise ValueError(f"unknown measure {measure!r}")


def gram_sqrt(g: np.ndarray, tol: float = 1e-14) -> np.ndarray:
    """Symmetric PSD square root of a monomial Gram matrix."""
    w, V = np.linalg.eigh((g + g.T) / 2.0)
    w = np.clip(w, 0.0, None)
    small = w < tol * max(w.max(), 1.0)
    w[small] = 0.0
    return (V * np.sqrt(w)) @ V.T
