"""Is a certified subspace of approximate generators closed under the bracket?

A certified subspace says that every generator in it has a small defect.  It
does not say that the subspace is (close to) a Lie algebra: for exact
symmetries the bracket of two symmetries is again a symmetry (Jacobi
identity), but the set of ``delta``-approximate generators is neither a vector
space nor closed under the bracket.  This module measures how far a subspace
is from closure.

For generators ``xi_a, xi_b`` in the subspace ``V`` the commutator
``[xi_a, xi_b] = D xi_b xi_a - D xi_a xi_b`` is computed exactly, and its part
outside ``V`` is measured on the same scale as the defect:

    c(a, b) = || P_perp [xi_a, xi_b] ||_{L2(nu)}
              / ( ||D xi_b xi_a||^2_{L2(nu)} + ||D xi_a xi_b||^2_{L2(nu)} )^{1/2},

with ``P_perp`` the ``L2(nu)``-orthogonal projection onto the complement of
``V`` inside the candidate class.  ``c = 0`` for every pair exactly when ``V``
is a Lie subalgebra; ``c`` lies in ``[0, sqrt(2)]``.  When the candidate class
is itself closed under the bracket (affine generators are), each commutator is
also a candidate generator, and its defect can be certified like any other.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np

from .polynomials import jac_times, moment_matrix

__all__ = ["closure_report"]


def _field(prob, theta):
    return np.einsum("p,pij->ij", theta, prob.gen_basis)


def _l2_gram(prob):
    """Gram matrix of vector fields over the whole monomial basis under ``nu``."""
    g = moment_matrix(prob.alg, prob.measure, prob.scale)
    W = np.eye(prob.n) if prob.metric is None else prob.metric
    return np.kron(W, g)


def closure_report(prob, V: np.ndarray, fit=None, alpha: float = 0.05,
                   rel_tol: float = 0.05) -> dict:
    """Closure diagnostics for the subspace spanned by the columns of ``V``.

    Returns ``max_closure`` (the largest ``c(a, b)`` over pairs of an
    ``L2(nu)``-orthonormal basis of ``V``), ``class_residual`` (how far the
    commutators fall outside the candidate class; zero for affine generators)
    and, if ``fit`` is given, ``commutator_upper``: the largest certified upper
    bound on the defect of a commutator whose size relative to its two terms
    exceeds ``rel_tol`` (``n_commutators`` counts them; smaller commutators,
    as in an abelian algebra, are negligible and have no direction to certify).
    """
    from .certify import certify_direction

    V = np.atleast_2d(np.asarray(V, float).T).T
    d = V.shape[1]
    out = {"dim": d, "max_closure": 0.0, "class_residual": 0.0,
           "commutator_upper": 0.0, "n_commutators": 0}
    if d < 2:
        return out
    G = _l2_gram(prob)
    flat = prob.gen_basis.reshape(prob.P, -1)              # (P, n*m)
    Gp = flat @ G @ flat.T                                  # generator Gram
    # Orthonormal basis of V in the L2(nu) geometry of the fields.
    w, U = np.linalg.eigh(V.T @ Gp @ V)
    Vo = V @ (U / np.sqrt(np.clip(w, 1e-300, None)))
    # Projector onto V inside the candidate class, in theta coordinates.
    PV = Vo @ Vo.T @ Gp
    worst = resid_worst = up_worst = 0.0
    n_comm = 0
    for i, j in combinations(range(d), 2):
        xa, xb = _field(prob, Vo[:, i]), _field(prob, Vo[:, j])
        t1 = jac_times(prob.alg, xb, xa)                    # D xi_b xi_a
        t2 = jac_times(prob.alg, xa, xb)                    # D xi_a xi_b
        com = (t1 - t2).ravel()
        # Express the commutator in the candidate basis (exact for affine).
        coef, *_ = np.linalg.lstsq(flat.T, com, rcond=None)
        r = com - flat.T @ coef
        nrm = lambda f: float(np.sqrt(max(f @ G @ f, 0.0)))
        den = float(np.hypot(nrm(t1.ravel()), nrm(t2.ravel())))
        resid_worst = max(resid_worst, nrm(r) / den if den > 0 else 0.0)
        perp = coef - PV @ coef
        num = nrm(flat.T @ perp)
        worst = max(worst, num / den if den > 0 else 0.0)
        # A commutator that is negligible relative to its two terms (as in an
        # abelian algebra) has no meaningful direction to certify.
        if fit is not None and den > 0 and nrm(com) / den > rel_tol:
            n_comm += 1
            up_worst = max(up_worst, certify_direction(prob, coef, fit, alpha).upper)
    out.update(max_closure=worst, class_residual=resid_worst,
               commutator_upper=up_worst, n_commutators=n_comm)
    return out
