"""Certified extrema over the ball ``||u|| <= R``: norms, and ratios of norms.

Every function here returns a bound that errs in the safe direction: an upper
bound for a maximum and a lower bound for a minimum.  The bounds come from
Lagrangian duality rather than from the point a root finder happens to stop
at, so the accuracy of the one-dimensional root find affects only how tight a
bound is, never whether it is valid.  Strong duality holds for the
trust-region subproblem even when its quadratic is indefinite (the S-lemma),
so the dual bounds are also tight: the gap is set by the root-finding
tolerance and is far below anything visible in the certificates.

Three problems are solved.

``max_norm_over_ball`` / ``min_norm_over_ball``
    ``max`` / ``min`` of ``||a + B u||`` over the ball, the two extrema needed
    by the separate (numerator and denominator bounded independently)
    certificate.

``trs_min_lower_bound``
    A lower bound on ``min u'Hu + 2 g'u + k`` over the ball for symmetric,
    possibly indefinite ``H``: the dual function of the trust-region
    subproblem, maximised over its multiplier.

``ratio_sup_over_ball`` / ``ratio_inf_over_ball``
    The supremum and infimum of ``||aN + BN u|| / ||aD + BD u||`` over the
    ball.  For a fixed level ``t``, whether the ratio can exceed ``sqrt(t)`` is
    the sign of ``max ||aN + BN u||^2 - t ||aD + BD u||^2``, a trust-region
    subproblem with an indefinite quadratic.  Dinkelbach's iteration (each
    step solves that subproblem and moves ``t`` to the ratio at its solution,
    a feasible point) converges to the extremum from inside; the result is
    then certified from outside by one dual bound, with bisection on ``t`` as
    a fall-back.

The multipliers are found by a safeguarded Newton iteration on the secular
equation, in the reciprocal form used by More and Sorensen, which is concave
and nearly linear, so Newton steps started left of the root increase
monotonically to it.  Floating-point rounding is covered by a relative safety
margin (``REL_SAFETY``), several orders of magnitude above the duality gaps
observed in the test suite.
"""

from __future__ import annotations

import numpy as np

__all__ = ["max_norm_over_ball", "min_norm_over_ball", "trs_min_lower_bound",
           "ratio_sup_over_ball", "ratio_inf_over_ball", "REL_SAFETY",
           "RATIO_RTOL", "ZERO_RTOL"]

REL_SAFETY = 1e-9          # inflation of upper bounds / deflation of lower bounds
RATIO_RTOL = 1e-7          # relative tolerance on the ratio extrema
ZERO_RTOL = 1e-12          # a norm below this fraction of its scale counts as zero
_ITERS = 200
_TINY = 1e-300


def _scale(a: np.ndarray, B: np.ndarray, R: float) -> float:
    """Natural size of ``||a + Bu||`` on the ball, the unit of rounding error."""
    return float(np.linalg.norm(a) + R * (np.linalg.norm(B, 2) if B.size else 0.0))


def _secular_root(b: np.ndarray, w2: np.ndarray, R2: float) -> float:
    """Root ``s > 0`` of ``f(s) = sum w2 / (b + s)^2 = R2`` for ``b >= 0``.

    Returns ``0.0`` when ``f(0+) <= R2`` (no root: the left end is optimal).
    Safeguarded Newton on ``psi(s) = f(s)^(-1/2) - R2^(-1/2)``, which is
    increasing and concave, so iterates started left of the root increase
    monotonically to it; bisection takes over if anything goes wrong.
    """
    pos = w2 > 0
    if not pos.any():
        return 0.0
    b, w2 = b[pos], w2[pos]
    if np.all(b > 0) and float(np.sum(w2 / b ** 2)) <= R2:
        return 0.0
    rR = 1.0 / np.sqrt(R2)
    # Start left of the root: f(s) >= R2 for any s below the smallest term's
    # own root, s0 = max(0, sqrt(w2_i / R2) - b_i) for the dominant term.
    s = max(float(np.max(np.sqrt(w2 / R2) - b)), 0.0)
    if s <= 0:
        s = _TINY
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        for _ in range(100):
            d = b + s
            f = float(np.sum(w2 / d ** 2))
            if not np.isfinite(f):
                s = max(s * 1e3, 1e-280)
                continue
            if abs(f - R2) <= 1e-13 * R2:
                return s
            if f < R2:                    # overshoot (rounding): fall back
                break
            f3 = float(np.sum(w2 / d ** 3))
            step = (rR - f ** -0.5) / (f ** -1.5 * f3)
            if not np.isfinite(step) or step <= s * 1e-15:
                return s
            s = s + step
    lo = 0.0
    hi = max(float(np.sqrt(np.sum(w2) / R2)), 1e-300)
    while float(np.sum(w2 / (b + hi) ** 2)) > R2:
        hi *= 2.0
    for _ in range(_ITERS):
        mid = 0.5 * (lo + hi)
        with np.errstate(over="ignore", divide="ignore"):
            if float(np.sum(w2 / (b + mid) ** 2)) > R2:
                lo = mid
            else:
                hi = mid
        if hi - lo <= 1e-15 * hi:
            break
    return hi


def _eig(B: np.ndarray, a: np.ndarray):
    Q = B.T @ B
    lam, V = np.linalg.eigh((Q + Q.T) / 2.0)
    lam = np.clip(lam, 0.0, None)
    return lam, V.T @ (B.T @ a), float(a @ a)


def _safe_sum(num2: np.ndarray, den: np.ndarray) -> float:
    """``sum num2 / den`` with 0/0 read as 0 and x/0 as +inf."""
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        t = np.where(den > 0, num2 / np.where(den > 0, den, 1.0),
                     np.where(num2 == 0, 0.0, np.inf))
    return float(np.sum(t))


# ------------------------------------------------------------ norm extrema
def max_norm_over_ball(a: np.ndarray, B: np.ndarray, R: float) -> float:
    """Certified upper bound on ``max_{||u|| <= R} ||a + B u||``.

    For every ``mu > lambda_max(B'B)``,
    ``||a + Bu||^2 <= a'a + mu R^2 + c'(mu I - B'B)^{-1} c`` on the ball
    (``c = B'a``); the bound is minimised over ``mu`` at the root of its
    derivative, the classical secular equation.
    """
    if R <= 0 or B.size == 0:
        return float(np.linalg.norm(a))
    lam, ct, aa = _eig(B, a)
    lmax = float(lam.max())
    gap = lmax - lam                    # >= 0; mu - lam = gap + s with s = mu - lmax
    s = _secular_root(gap, ct ** 2, R * R)
    if s <= 0:                          # hard case: optimum at the left end
        s = _TINY * max(lmax, 1.0)
    best = aa + (lmax + s) * R * R + _safe_sum(ct ** 2, gap + s)
    return float(np.sqrt(max(best, 0.0)) * (1.0 + REL_SAFETY)
                 + REL_SAFETY * _scale(a, B, R))


def min_norm_over_ball(a: np.ndarray, B: np.ndarray, R: float) -> float:
    """Certified lower bound on ``min_{||u|| <= R} ||a + B u||``.

    For every ``nu >= 0`` with ``B'B + nu I`` positive definite,
    ``||a + Bu||^2 >= a'a - c'(B'B + nu I)^{-1} c - nu R^2`` on the ball.
    When the unconstrained minimiser lies inside the ball the optimal
    multiplier is ``nu = 0`` and the bound is the least-squares residual of
    ``a`` on the range of ``B``.
    """
    if R <= 0 or B.size == 0:
        return float(np.linalg.norm(a))
    lam, ct, aa = _eig(B, a)
    tiny = 1e-12 * max(float(lam.max()), 1e-300)
    pos = lam > tiny
    u_norm = float(np.linalg.norm(ct[pos] / lam[pos]))
    if u_norm <= R:
        coef, *_ = np.linalg.lstsq(B, -a, rcond=None)
        resid = float(np.linalg.norm(a + B @ coef))
        if np.linalg.norm(coef) <= R * (1 + 1e-12):
            return float(max(resid * (1.0 - REL_SAFETY)
                             - REL_SAFETY * _scale(a, B, R), 0.0))
    nu = _secular_root(lam, ct ** 2, R * R)
    if nu <= 0:
        nu = _TINY
    best = aa - _safe_sum(ct ** 2, lam + nu) - nu * R * R
    return float(max(np.sqrt(max(best, 0.0)) * (1.0 - REL_SAFETY)
                     - REL_SAFETY * _scale(a, B, R), 0.0))


# -------------------------------------------- general trust-region problem
def _trs_solve(H: np.ndarray, g: np.ndarray, k: float, R: float):
    """Dual lower bound on ``min_{||u||<=R} u'Hu + 2g'u + k`` and a feasible
    point attaining the minimum up to the root-finding tolerance."""
    lam, V = np.linalg.eigh((H + H.T) / 2.0)
    gt = V.T @ g
    left = max(0.0, -float(lam.min()))
    base = np.maximum(lam + left, 0.0)
    s = _secular_root(base, gt ** 2, R * R)
    if s > 0:
        lb = k - _safe_sum(gt ** 2, base + s) - (left + s) * R * R
        u = -V @ (gt / (base + s))
    elif left == 0.0 and np.all(lam > 0):
        # Interior minimiser of a positive definite quadratic.
        lb = k - _safe_sum(gt ** 2, lam)
        u = -V @ (gt / lam)
    else:
        # Hard case: the minimiser uses the bottom eigenvector to reach the
        # boundary; the dual is evaluated just right of the left end.
        s = _TINY * max(float(np.abs(lam).max()), 1.0)
        lb = k - _safe_sum(gt ** 2, base + s) - (left + s) * R * R
        ut = np.zeros_like(gt)
        nz = base > 0
        ut[nz] = -gt[nz] / base[nz]
        n2 = float(ut @ ut)
        if n2 < R * R:
            ut[int(np.argmin(base))] += np.sqrt(R * R - n2)
        u = V @ ut
    nu = float(np.linalg.norm(u))
    if nu > R:
        u = u * (R / nu)
    return float(lb), u


def trs_min_lower_bound(H: np.ndarray, g: np.ndarray, k: float, R: float) -> float:
    """Certified lower bound on ``min_{||u|| <= R} u'Hu + 2 g'u + k``.

    ``H`` is symmetric and may be indefinite.  The dual function
    ``d(nu) = k - g'(H + nu I)^{-1} g - nu R^2`` is a lower bound for every
    ``nu >= max(0, -lambda_min(H))`` (strictly above when ``g`` has a
    component in the bottom eigenspace) and is concave there; its maximum
    equals the minimum of the subproblem (no duality gap), including the hard
    case, which is approached as a limit.
    """
    return _trs_solve(H, g, k, R)[0]


# ---------------------------------------------------------- ratio extrema
def _parts(aN, BN, aD, BD):
    return (BN.T @ BN, BN.T @ aN, float(aN @ aN),
            BD.T @ BD, BD.T @ aD, float(aD @ aD))


def _ratio2(u, aN, BN, aD, BD) -> float:
    n = aN + BN @ u
    dd = aD + BD @ u
    den = float(dd @ dd)
    return float(n @ n) / den if den > 0 else np.inf


def ratio_sup_over_ball(aN, BN, aD, BD, R: float, t_hi: float | None = None,
                        rtol: float = RATIO_RTOL) -> float:
    """Certified upper bound on ``sup_{||u|| <= R} ||aN + BN u|| / ||aD + BD u||``.

    Returns ``inf`` when the denominator can vanish on the ball.  ``t_hi`` is an
    optional known upper bound on the ratio (for example the separate bound),
    used by the bisection fall-back.
    """
    if min_norm_over_ball(aD, BD, R) <= ZERO_RTOL * _scale(aD, BD, R):
        return float("inf")
    QN, cN, kN, QD, cD, kD = _parts(aN, BN, aD, BD)

    def below(r):
        """Dual certificate that the ratio is < r on the whole ball."""
        t = r * r
        return _trs_solve(t * QD - QN, t * cD - cN, t * kD - kN, R)[0] > 0.0

    # Dinkelbach from inside: every iterate is the ratio at a feasible point.
    t = _ratio2(np.zeros(BN.shape[1]), aN, BN, aD, BD)
    for _ in range(60):
        _, u = _trs_solve(t * QD - QN, t * cD - cN, t * kD - kN, R)
        t_new = _ratio2(u, aN, BN, aD, BD)
        if not t_new > t * (1 + 1e-13):
            break
        t = t_new
    lo = float(np.sqrt(t))
    hi = lo * (1.0 + rtol)
    if below(hi):
        return float(hi * (1.0 + REL_SAFETY))
    # Fall-back: bisection between the attained value and a certified end.
    if t_hi is None or not np.isfinite(t_hi):
        t_hi = max_norm_over_ball(aN, BN, R) / min_norm_over_ball(aD, BD, R)
    hi = max(float(t_hi), hi) * (1.0 + 1e-12)
    for _ in range(60):
        if below(hi):
            break
        hi *= 1.0 + 1e-6
    for _ in range(_ITERS):
        if hi - lo <= rtol * hi:
            break
        mid = 0.5 * (lo + hi)
        if below(mid):
            hi = mid
        else:
            lo = mid
    return float(hi * (1.0 + REL_SAFETY))


def ratio_inf_over_ball(aN, BN, aD, BD, R: float, t_lo: float = 0.0,
                        rtol: float = RATIO_RTOL) -> float:
    """Certified lower bound on ``inf_{||u|| <= R} ||aN + BN u|| / ||aD + BD u||``.

    ``t_lo`` is an optional known lower bound (for example the separate bound),
    used by the bisection fall-back.
    """
    QN, cN, kN, QD, cD, kD = _parts(aN, BN, aD, BD)
    t = _ratio2(np.zeros(BN.shape[1]), aN, BN, aD, BD)
    if not np.isfinite(t):
        return float(max(t_lo, 0.0))

    def above(r):
        """Dual certificate that the ratio is > r on the whole ball."""
        s = r * r
        return _trs_solve(QN - s * QD, cN - s * cD, kN - s * kD, R)[0] > 0.0

    for _ in range(60):
        _, u = _trs_solve(QN - t * QD, cN - t * cD, kN - t * kD, R)
        t_new = _ratio2(u, aN, BN, aD, BD)
        if not t_new < t * (1 - 1e-13):
            break
        t = t_new
    hi = float(np.sqrt(max(t, 0.0)))
    if hi <= 0:
        return 0.0
    lo = hi * (1.0 - rtol)
    if above(lo):
        return float(lo * (1.0 - REL_SAFETY))
    lo = max(float(t_lo) * (1.0 - 1e-12), 0.0)
    if lo > 0 and not above(lo):
        lo = 0.0
    for _ in range(_ITERS):
        if hi - lo <= rtol * max(hi, 1e-300):
            break
        mid = 0.5 * (lo + hi)
        if above(mid):
            lo = mid
        else:
            hi = mid
    return float(lo * (1.0 - REL_SAFETY))
