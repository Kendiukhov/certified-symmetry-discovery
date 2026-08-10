"""Baseline symmetry-discovery rules.

Each baseline maps a fitted model to a declared symmetry subspace.  They span
the practice implied by the nullspace framework of Otto et al. (2025) and the
two remedies suggested most often for noisy data: singular-value perturbation
bounds and sample splitting.

* :func:`naive_threshold` -- declare every direction whose plug-in defect falls
  below a fixed tolerance (the de-facto standard).
* :func:`eigengap` -- pick the dimension at the largest relative gap in the
  plug-in defect spectrum.
* :func:`weyl_certificate` -- a valid but coarse certificate obtained from Weyl's
  inequality applied to the Gram matrix, instead of to the defect ratio.
* :func:`split_significance` -- select a direction on one half of the data and
  run a Wald test of ``H0: the direction is an exact symmetry`` on the other
  half, declaring a symmetry when the test fails to reject.
* :func:`rank_test_dimension` -- a Wald-type rank test for the dimension of the
  nullspace, in the spirit of Robin & Smith (2000) and Kleibergen & Paap (2006).
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from .trs import max_norm_over_ball

__all__ = ["naive_threshold", "eigengap", "weyl_certificate", "split_significance",
           "rank_test_dimension", "satterthwaite_pvalue"]


def naive_threshold(prob, fit, tau: float) -> tuple[int, np.ndarray, np.ndarray]:
    """Directions with plug-in defect ``<= tau``."""
    rho, Theta = prob.spectrum(fit.beta)
    d = int(np.sum(rho <= tau))
    return d, Theta[:, :d], rho


def eigengap(prob, fit, max_dim: int | None = None) -> tuple[int, np.ndarray, np.ndarray]:
    """Dimension at the largest relative gap ``rho_{d+1} / rho_d``."""
    rho, Theta = prob.spectrum(fit.beta)
    P = len(rho) if max_dim is None else min(max_dim + 1, len(rho))
    eps = 1e-300
    ratios = [(rho[d] + eps) / (rho[d - 1] + eps) for d in range(1, P)]
    d = int(np.argmax(ratios)) + 1 if ratios else 0
    return d, Theta[:, :d], rho


def weyl_certificate(prob, fit, alpha: float = 0.05) -> np.ndarray:
    """Weyl-inequality upper bounds on the true defects of the plug-in directions.

    ``lambda_k(C(beta)) <= lambda_k(C(beta_hat)) + ||C(beta) - C(beta_hat)||``,
    with the perturbation bounded by
    ``2 ||A_hat||_F ||Delta A||_F + ||Delta A||_F^2`` and ``||Delta A||_F``
    maximised exactly over the confidence ellipsoid.  Dividing by a lower bound
    on ``lambda_min(D)`` turns this into a bound on the defect.  This is the
    natural "singular-value perturbation" answer; it is valid but markedly
    looser than the ratio-wise certificate of :mod:`hsd.certify`, because it
    spends a single global perturbation budget on every direction.
    """
    R = fit.ellipsoid_radius(alpha)
    S = fit.cov_sqrt
    TA = prob.Tw.reshape(-1, prob.Q)
    TM = np.concatenate([prob.T1w.reshape(-1, prob.Q), prob.T2w.reshape(-1, prob.Q)], axis=0)
    dA = max_norm_over_ball(np.zeros(TA.shape[0]), TA @ S, R)
    dM = max_norm_over_ball(np.zeros(TM.shape[0]), TM @ S, R)
    Ahat, Mhat = prob.A(fit.beta), prob.M(fit.beta)
    nA = float(np.linalg.norm(Ahat, 2))
    nM = float(np.linalg.norm(Mhat, 2))
    pertC = 2 * nA * dA + dA ** 2
    pertD = 2 * nM * dM + dM ** 2
    Chat = Ahat.T @ Ahat
    lamC = np.clip(np.linalg.eigvalsh((Chat + Chat.T) / 2), 0.0, None)
    Dhat = Mhat.T @ Mhat
    lamD_min = float(np.clip(np.linalg.eigvalsh((Dhat + Dhat.T) / 2).min() - pertD, 0.0, None))
    if lamD_min <= 0:
        return np.full(prob.P, np.inf)
    return np.sqrt(np.sort(lamC + pertC) / lamD_min)


def _quad_eigs(L: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Eigenvalues of ``L cov L^T`` computed on the (small) parameter side."""
    w, U = np.linalg.eigh((cov + cov.T) / 2)
    S = U * np.sqrt(np.clip(w, 0.0, None))
    B = L @ S
    return np.clip(np.linalg.eigvalsh(B.T @ B), 0.0, None)


def satterthwaite_pvalue(stat: float, V: np.ndarray = None, *,
                         eigs: np.ndarray = None) -> float:
    """``P(sum_i lambda_i chi^2_1 >= stat)`` by the Welch--Satterthwaite match."""
    lam = eigs if eigs is not None else np.clip(np.linalg.eigvalsh((V + V.T) / 2), 0.0, None)
    lam = lam[lam > 1e-14 * max(lam.max(initial=0.0), 1e-300)]
    if lam.size == 0:
        return 0.0 if stat > 0 else 1.0
    E, var = lam.sum(), 2.0 * np.sum(lam ** 2)
    k = 2.0 * E ** 2 / var
    scale = var / (2.0 * E)
    return float(stats.chi2.sf(stat / scale, k))


def split_significance(prob, fit_select, fit_test, alpha: float = 0.05):
    """Select a direction on one split, test ``H0: exact symmetry`` on the other.

    Returns ``(declared, theta, pvalue)``.  ``declared`` is ``True`` when the
    test *fails to reject*, which is how a significance test is used to
    "discover" a symmetry.  The test itself is valid -- it rejects a true
    symmetry with probability at most ``alpha`` -- but accepting ``H0`` carries
    no guarantee whatsoever, which is precisely the honesty gap.
    """
    _, Theta = prob.spectrum(fit_select.beta)
    theta = Theta[:, 0]
    theta = theta / np.linalg.norm(theta)
    L = prob.L_theta(theta)
    r = L @ fit_test.beta
    p = satterthwaite_pvalue(float(r @ r), eigs=_quad_eigs(L, fit_test.cov))
    return (p > alpha), theta, p


def rank_test_dimension(prob, fit, alpha: float = 0.05, max_dim: int | None = None) -> int:
    """Largest ``d`` such that ``H0: dim null >= d`` is not rejected at level ``alpha``.

    The statistic is the squared ``L2(nu)`` norm of the bracket restricted to the
    ``d`` plug-in directions with smallest defect, referred to the
    Welch--Satterthwaite approximation of its null distribution.  Like every
    rank test, it places "a symmetry exists" in the null hypothesis.
    """
    _, Theta = prob.spectrum(fit.beta)
    cov_sqrt = fit.cov_sqrt
    P = prob.P if max_dim is None else min(max_dim, prob.P)
    best = 0
    for d in range(1, P + 1):
        Vd = Theta[:, :d]
        Ld = np.einsum("kpq,pd->kdq", prob.Tw, Vd).reshape(-1, prob.Q)
        r = Ld @ fit.beta
        p = satterthwaite_pvalue(float(r @ r), eigs=_quad_eigs(Ld, cov_sqrt @ cov_sqrt.T))
        if p > alpha:
            best = d
        else:
            break
    return best
