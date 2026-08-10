"""Benchmark polynomial dynamical systems with exactly known symmetry algebras.

Every system is specified by *rational* coefficients, so the true Lie algebra of
symmetries inside the candidate class can be obtained as the **exact rational
nullspace** of the linear map ``theta -> coef([xi_theta, F])`` (see
:func:`exact_symmetry_algebra`).  No thresholding, no floating-point tolerance:
the ground truth used to score every experiment is exact.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction

import numpy as np
import sympy as sp

from .defect import DefectProblem

__all__ = ["System", "SYSTEMS", "get_system", "exact_symmetry_algebra"]


@dataclass
class System:
    """A polynomial vector field ``x' = F(x)`` on ``R^n``.

    ``terms`` maps ``(component, exponent tuple) -> Fraction`` coefficient.
    """

    name: str
    n: int
    deg: int
    terms: dict[tuple[int, tuple[int, ...]], Fraction]
    description: str = ""
    true_dim: int | None = None       # dim of the symmetry algebra (filled in lazily)
    tags: tuple[str, ...] = field(default_factory=tuple)

    def field(self, alg) -> np.ndarray:
        F = np.zeros((self.n, alg.m))
        for (i, a), c in self.terms.items():
            F[i, alg.index[a]] = float(c)
        return F

    def beta(self, prob: DefectProblem) -> np.ndarray:
        return prob.beta_of_field(self.field(prob.alg))

    def rhs(self, x: np.ndarray) -> np.ndarray:
        """Evaluate ``F`` at points ``x`` of shape ``(N, n)``."""
        x = np.atleast_2d(x)
        out = np.zeros_like(x)
        for (i, a), c in self.terms.items():
            out[:, i] += float(c) * np.prod(x ** np.array(a), axis=1)
        return out


def _t(*pairs):
    return {(i, tuple(a)): Fraction(c) for i, a, c in pairs}


# --------------------------------------------------------------------------
# Library
# --------------------------------------------------------------------------
def _linear_system(name, A, description, tags=()):
    n = A.shape[0]
    terms = {}
    for i in range(n):
        for j in range(n):
            if A[i, j] != 0:
                e = [0] * n
                e[j] = 1
                terms[(i, tuple(e))] = Fraction(A[i, j]).limit_denominator(10 ** 6)
    return System(name, n, 1, terms, description, tags=tags)


_SYS: dict[str, System] = {}


def _add(s: System) -> System:
    _SYS[s.name] = s
    return s


# ---- linear systems (ground-truth symmetry algebra = centraliser of A) ----
_add(_linear_system(
    "lin_rot2", np.array([[0.0, -1.0], [1.0, 0.0]]),
    "Planar harmonic oscillator x' = -y, y' = x (rotationally symmetric).",
    tags=("linear", "symmetric")))
_add(_linear_system(
    "lin_diag2", np.array([[-1.0, 0.0], [0.0, -2.0]]),
    "Planar node with distinct eigenvalues -1, -2.",
    tags=("linear",)))
_add(_linear_system(
    "lin_jordan2", np.array([[-1.0, 1.0], [0.0, -1.0]]),
    "Planar defective (Jordan) node with a repeated eigenvalue.",
    tags=("linear",)))
_add(_linear_system(
    "lin_generic3", np.array([[-1.0, 2.0, 0.0], [0.0, -3.0, 1.0], [1.0, 0.0, -2.0]]),
    "Generic 3-d linear system with distinct eigenvalues.",
    tags=("linear",)))
_add(_linear_system(
    "lin_scalar3", np.diag([-1.0, -1.0, -1.0]),
    "Isotropic 3-d contraction x' = -x (maximal symmetry: all of gl(3)).",
    tags=("linear", "symmetric")))

# ---- nonlinear planar systems ----
_add(System(
    "hopf", 2, 3,
    _t((0, (1, 0), 1), (0, (0, 1), -1), (0, (3, 0), -1), (0, (1, 2), -1),
       (1, (1, 0), 1), (1, (0, 1), 1), (1, (2, 1), -1), (1, (0, 3), -1)),
    "Supercritical Hopf normal form (unit limit cycle); exact SO(2) symmetry.",
    tags=("nonlinear", "symmetric")))
_add(System(
    "vanderpol", 2, 3,
    _t((0, (0, 1), 1), (1, (1, 0), -1), (1, (0, 1), 1), (1, (2, 1), -1)),
    "Van der Pol oscillator (mu = 1); no continuous symmetry in the affine class.",
    tags=("nonlinear",)))
_add(System(
    "duffing", 2, 3,
    _t((0, (0, 1), 1), (1, (1, 0), 1), (1, (3, 0), -1), (1, (0, 1), Fraction(-1, 5))),
    "Damped Duffing oscillator.",
    tags=("nonlinear",)))
_add(System(
    "lotka_volterra", 2, 2,
    _t((0, (1, 0), 1), (0, (1, 1), -1), (1, (0, 1), -1), (1, (1, 1), 1)),
    "Lotka-Volterra predator-prey model.",
    tags=("nonlinear",)))
_add(System(
    "selkov", 2, 3,
    _t((0, (0, 0), Fraction(1, 10)), (0, (1, 0), -1), (0, (0, 1), Fraction(1, 2)),
       (0, (2, 1), 1), (1, (0, 1), Fraction(-1, 2)), (1, (2, 1), -1)),
    "Selkov glycolysis model (affine-inhomogeneous, no continuous symmetry).",
    tags=("nonlinear",)))

# ---- three-dimensional systems ----
_add(System(
    "lorenz", 3, 2,
    _t((0, (1, 0, 0), -10), (0, (0, 1, 0), 10),
       (1, (1, 0, 0), 28), (1, (0, 1, 0), -1), (1, (1, 0, 1), -1),
       (2, (1, 1, 0), 1), (2, (0, 0, 1), Fraction(-8, 3))),
    "Lorenz system (only a discrete Z2 symmetry; no continuous one).",
    tags=("nonlinear", "chaotic")))


def _euler_rigid_body(name, I1, I2, I3, description, tags):
    """Free rigid body in body coordinates: w1' = (I2-I3)/I1 w2 w3, etc."""
    c = [Fraction(I2 - I3, I1), Fraction(I3 - I1, I2), Fraction(I1 - I2, I3)]
    terms = {}
    if c[0] != 0:
        terms[(0, (0, 1, 1))] = c[0]
    if c[1] != 0:
        terms[(1, (1, 0, 1))] = c[1]
    if c[2] != 0:
        terms[(2, (1, 1, 0))] = c[2]
    return System(name, 3, 2, terms, description, tags=tags)


_add(_euler_rigid_body("rigid_sym", 1, 1, 2,
                       "Euler equations of a symmetric top I1 = I2 (an extra S^1 symmetry).",
                       ("nonlinear", "symmetric")))
_add(_euler_rigid_body("rigid_asym", 1, 2, 3,
                       "Euler equations of an asymmetric rigid body.",
                       ("nonlinear",)))

_add(System(
    "sphere_flow3", 3, 3,
    _t((0, (1, 0, 0), 1), (0, (3, 0, 0), -1), (0, (1, 2, 0), -1), (0, (1, 0, 2), -1),
       (1, (0, 1, 0), 1), (1, (2, 1, 0), -1), (1, (0, 3, 0), -1), (1, (0, 1, 2), -1),
       (2, (0, 0, 1), 1), (2, (2, 0, 1), -1), (2, (0, 2, 1), -1), (2, (0, 0, 3), -1)),
    "Radial flow onto the unit sphere x' = x(1-|x|^2); full SO(3) symmetry.",
    tags=("nonlinear", "symmetric")))

SYSTEMS = _SYS


def get_system(name: str) -> System:
    return SYSTEMS[name]


def broken_hopf(eps) -> System:
    """Hopf normal form plus a symmetry-breaking term of size ``eps``.

    At ``eps = 0`` the rotation generator is an exact symmetry; the relative
    defect of the rotation grows continuously from zero with ``eps``.  This
    one-parameter family is what makes a *resolution* question well posed: how
    small a symmetry violation can a procedure certify away, and how large a
    violation can it detect?
    """
    e = Fraction(eps).limit_denominator(10 ** 9)
    terms = dict(get_system("hopf").terms)
    terms[(0, (2, 0))] = terms.get((0, (2, 0)), Fraction(0)) + e
    return System(f"hopf_eps{float(eps):g}", 2, 3, terms,
                  f"Hopf normal form with an x^2 symmetry-breaking term of size {float(eps):g}.",
                  tags=("nonlinear", "family"))


def truncate(sys_: System, deg: int) -> System:
    """Drop all terms of total degree above ``deg`` (model-class truncation)."""
    terms = {k: v for k, v in sys_.terms.items() if sum(k[1]) <= deg}
    return System(f"{sys_.name}_trunc{deg}", sys_.n, min(sys_.deg, deg), terms,
                  f"{sys_.description} (truncated to degree {deg})", tags=sys_.tags)


# --------------------------------------------------------------------------
# Exact ground truth
# --------------------------------------------------------------------------
def exact_symmetry_algebra(sys_: System, prob: DefectProblem) -> np.ndarray:
    """Exact basis of ``{theta : [xi_theta, F] = 0}`` via rational linear algebra.

    Returns a ``(P, d)`` array whose columns span the true symmetry algebra of
    ``sys_`` inside the candidate class of ``prob`` (``d = 0`` gives shape
    ``(P, 0)``).  Computed with :mod:`sympy` over the rationals, so the answer
    is exact and does not depend on any tolerance.
    """
    alg = prob.alg
    n, P = prob.n, prob.P
    # Column p of the matrix is coef([zeta_p, F]) in the residual basis.
    Fq = sp.zeros(n, alg.m)
    for (i, a), c in sys_.terms.items():
        Fq[i, alg.index[a]] = sp.Rational(c.numerator, c.denominator)
    rows = []
    for p in range(P):
        col = _sym_bracket(alg, prob.gen_basis[p], Fq, prob.res_idx)
        rows.append(col)
    Mat = sp.Matrix.hstack(*rows)          # K x P exact matrix
    ns = Mat.nullspace()
    if not ns:
        return np.zeros((P, 0))
    B = np.array([[float(v) for v in vec] for vec in ns]).T
    # orthonormalise for numerical convenience
    Qm, _ = np.linalg.qr(B)
    return Qm[:, : B.shape[1]]


def _sym_bracket(alg, xi_np, Fq, res_idx):
    """Exact ``coef([xi, F])`` where ``xi`` is a float 0/1 basis field."""
    n = alg.n
    xi = sp.zeros(n, alg.m)
    for i in range(n):
        for j in np.flatnonzero(xi_np[i]):
            xi[i, int(j)] = sp.Rational(int(round(xi_np[i, j])), 1)
    out = _sym_jac_times(alg, Fq, xi) - _sym_jac_times(alg, xi, Fq)
    return sp.Matrix([out[i, int(a)] for i in range(n) for a in res_idx])


def _sym_jac_times(alg, V, W):
    n = alg.m
    nn = alg.n
    out = sp.zeros(nn, n)
    for i in range(nn):
        for j in range(nn):
            # d V_i / d x_j
            for idx in range(n):
                if V[i, idx] == 0:
                    continue
                c, k = alg.dcoef[idx, j], alg.didx[idx, j]
                if c == 0:
                    continue
                for idx2 in range(n):
                    if W[j, idx2] == 0:
                        continue
                    kk = alg.mul[int(k), idx2]
                    if kk < 0:
                        raise ValueError("degree overflow in exact bracket")
                    out[i, int(kk)] += sp.Integer(int(c)) * V[i, idx] * W[j, idx2]
    return out
