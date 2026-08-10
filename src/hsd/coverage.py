"""How incomplete coverage of the state space manufactures symmetry.

If the data occupy only part of the region where the model is meant to hold,
the empirical bracket ``[xi, F]`` can vanish on the data while being large on
the target domain.  Because every bracket is a polynomial in a fixed
finite-dimensional space, this "extrapolation gap" is an exactly computable
number: the largest ratio of the target-domain norm to the data norm over that
polynomial space.
"""

from __future__ import annotations

import numpy as np

from .polynomials import moment_matrix

__all__ = ["extrapolation_factor", "empirical_residual_gram", "coverage_report"]


def empirical_residual_gram(prob, X: np.ndarray) -> np.ndarray:
    """Gram matrix of the residual monomial basis under the empirical measure."""
    Phi = prob.alg.features(X)[:, prob.res_idx]
    return (Phi.T @ Phi) / X.shape[0]


def extrapolation_factor(prob, X: np.ndarray, ridge: float = 0.0) -> dict:
    """Coverage diagnostics comparing the data measure to the target measure.

    Returns a dictionary with

    ``kappa``   ``lambda_max(G_mu^{-1/2} G_nu G_mu^{-1/2})``: the worst-case
                inflation of a squared bracket norm when moving from the data
                to the target domain.  ``inf`` iff some residual polynomial
                vanishes on the data, which is exactly the situation in which
                arbitrarily many spurious symmetries appear.
    ``kappa_rev`` the reverse factor, controlling deflation of the normaliser.
    ``bound``   ``sqrt(kappa * kappa_rev)``: the factor by which the defect on
                the target domain can exceed the defect measured on the data.
    ``rank_deficit`` number of residual directions the data cannot resolve.
    """
    Gmu = empirical_residual_gram(prob, X)
    Gnu = moment_matrix(prob.alg, prob.measure, prob.scale, sub=prob.res_idx)
    k = Gmu.shape[0]
    if ridge > 0:
        Gmu = Gmu + ridge * np.trace(Gmu) / k * np.eye(k)
    wmu = np.linalg.eigvalsh((Gmu + Gmu.T) / 2)
    tol = 1e-10 * max(wmu.max(), 1e-300)
    deficit = int(np.sum(wmu <= tol))
    if deficit > 0:
        return {"kappa": np.inf, "kappa_rev": np.inf, "bound": np.inf,
                "rank_deficit": deficit}
    Gmi = _inv_sqrt(Gmu)
    kappa = float(np.linalg.eigvalsh(Gmi @ Gnu @ Gmi).max())
    Gni = _inv_sqrt(Gnu)
    kappa_rev = float(np.linalg.eigvalsh(Gni @ Gmu @ Gni).max())
    return {"kappa": kappa, "kappa_rev": kappa_rev,
            "bound": float(np.sqrt(kappa * kappa_rev)), "rank_deficit": 0}


def _inv_sqrt(G: np.ndarray) -> np.ndarray:
    w, V = np.linalg.eigh((G + G.T) / 2)
    w = np.clip(w, 1e-300, None)
    return (V / np.sqrt(w)) @ V.T


def coverage_report(prob, X: np.ndarray) -> str:
    d = extrapolation_factor(prob, X)
    if not np.isfinite(d["bound"]):
        return ("coverage is rank deficient: %d residual directions are invisible "
                "in the data, so defects measured on the data say nothing about "
                "the target domain" % d["rank_deficit"])
    return ("defect on the target domain can exceed the defect measured on the "
            "data by up to a factor %.3g" % d["bound"])
