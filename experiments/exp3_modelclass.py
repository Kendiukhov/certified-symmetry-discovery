"""Experiment 3 -- the model class is the dominant source of false symmetry.

Three studies:

**(a) Truncation.** Fitting a smaller polynomial class than the truth produces a
model with *more* symmetry than the system has.  We measure how often that
turns into a false certification, whether a lack-of-fit test catches it, and
what it costs to certify inside a class large enough to contain the truth.

**(b) Sparsification.** Sequentially thresholded least squares -- the standard
sparse regression of data-driven model discovery -- deletes small terms.  On a
weakly symmetry-breaking system it deletes exactly the term that breaks the
symmetry, and the fitted model becomes exactly symmetric.

**(c) Sensitivity to model error.** For a genuinely non-polynomial system the
truth is outside every polynomial class.  The certificate then needs an
assumed bound ``eta`` on the approximation error; we report the certified
tolerance as a function of that assumption.
"""

from __future__ import annotations

import time

import numpy as np

from common import ALPHA, DELTA, run_cells, rms_field, save, sup_true_defect
from hsd import DefectProblem, exact_symmetry_algebra, fit_ols, get_system, simulate_design
from hsd.certify import certified_dimension, certify_direction
from hsd.estimation import lack_of_fit_pvalue
from hsd.nonpoly import SMOOTH_SYSTEMS, mc_defect
from hsd.defect import _deriv_field as _dcol
from hsd.systems import broken_hopf

TRIALS = 300


# ------------------------------------------------------------------ (a)
def truncation_cell(args):
    name, true_deg, fit_deg, scale, N, srel, seed = args
    rng = np.random.default_rng(seed)
    sys_ = get_system(name)
    prob_fit = DefectProblem.build(sys_.n, fit_deg, "affine", "box", scale=scale)
    prob_true = DefectProblem.build(sys_.n, true_deg, "affine", "box", scale=scale)
    beta0 = sys_.beta(prob_true)
    dstar = exact_symmetry_algebra(sys_, prob_true).shape[1]
    sigma = srel * rms_field(sys_, np.random.default_rng(0), scale)
    acc = dict(fc_naive=0.0, fc_sym=0.0, dim_naive=0.0, dim_sym=0.0,
               lof_reject=0.0, dim_sym_gated=0.0, fc_sym_gated=0.0)
    for _ in range(TRIALS):
        X, Y = simulate_design(sys_, rng, N=N, sigma=sigma, sampler="box", scale=scale)
        fit = fit_ols(prob_fit, X, Y)
        rho_hat, Theta = prob_fit.spectrum(fit.beta)
        d = int(np.sum(rho_hat <= DELTA))
        acc["dim_naive"] += d
        acc["fc_naive"] += _fc(prob_fit, prob_true, Theta[:, :d], beta0)
        dc, Vc = certified_dimension(prob_fit, fit, DELTA, ALPHA)
        acc["dim_sym"] += dc
        acc["fc_sym"] += _fc(prob_fit, prob_true, Vc, beta0)
        p = lack_of_fit_pvalue(prob_fit, prob_true, X, Y)
        reject = p < ALPHA
        acc["lof_reject"] += reject
        acc["dim_sym_gated"] += 0 if reject else dc
        acc["fc_sym_gated"] += 0.0 if reject else _fc(prob_fit, prob_true, Vc, beta0)
    out = {k: v / TRIALS for k, v in acc.items()}
    out.update(system=name, fit_degree=fit_deg, true_degree=true_deg, N=N,
               sigma_rel=srel, d_true=dstar)
    return out


def _fc(prob_fit, prob_true, V, beta0):
    """False certification is always judged against the *true* system, using the
    same generators embedded in the true problem's candidate space."""
    if V.size == 0:
        return 0.0
    return float(sup_true_defect(prob_true, V, beta0) > DELTA)


# ------------------------------------------------------------------ (b)
def stlsq(Phi, Y, thresh, iters=10):
    B = np.linalg.lstsq(Phi, Y, rcond=None)[0]
    for _ in range(iters):
        small = np.abs(B) < thresh
        B = np.where(small, 0.0, B)
        for j in range(Y.shape[1]):
            big = ~small[:, j]
            if big.any():
                B[big, j] = np.linalg.lstsq(Phi[:, big], Y[:, j], rcond=None)[0]
    return B


def sparsity_study():
    rng = np.random.default_rng(31)
    prob = DefectProblem.build(2, 3, "affine", "box", scale=1.0)
    rot = np.zeros(prob.P)
    rot[1], rot[2] = 1.0, -1.0            # the rotation generator
    rows = []
    for eps in [0.0, 0.02, 0.05, 0.1, 0.2, 0.4]:
        sys_ = broken_hopf(eps)
        beta0 = sys_.beta(prob)
        true_rho = prob.rho(rot, beta0)
        sigma = 0.05 * rms_field(sys_, np.random.default_rng(0), 1.0)
        for thresh in [0.0, 0.05, 0.1, 0.25]:
            dim_naive = cert = fc = 0
            for _ in range(TRIALS):
                X, Y = simulate_design(sys_, rng, N=1000, sigma=sigma, scale=1.0)
                Phi = prob.alg.features(X)[:, prob.beta_idx]
                B = stlsq(Phi.copy(), Y.copy(), thresh)
                beta = B.T.ravel()
                rho_hat, Theta = prob.spectrum(beta)
                d = int(np.sum(rho_hat <= DELTA))
                dim_naive += d
                fc += sup_true_defect(prob, Theta[:, :d], beta0) > DELTA
                # honest alternative: certify with the *unpruned* OLS fit
                fit = fit_ols(prob, X, Y)
                dc, Vc = certified_dimension(prob, fit, DELTA, ALPHA)
                cert += dc
            rows.append(dict(eps=eps, threshold=thresh, true_rot_defect=float(true_rho),
                             dim_stlsq=dim_naive / TRIALS, fc_stlsq=fc / TRIALS,
                             dim_symcert=cert / TRIALS))
    return rows


# ------------------------------------------------------------------ (c)
def eta_sensitivity():
    rng = np.random.default_rng(32)
    rows = []
    for key, scale, fit_deg in [("rot_nonpoly", 1.0, 5), ("pendulum", 1.5, 5)]:
        sm = SMOOTH_SYSTEMS[key]
        prob = DefectProblem.build(2, fit_deg, "affine", "box", scale=scale)
        Xr = rng.uniform(-scale, scale, size=(200_000, 2))
        rms = float(np.sqrt(np.mean(np.sum(sm.rhs(Xr) ** 2, axis=1))))
        X = rng.uniform(-scale, scale, size=(20_000, 2))
        Y = sm.rhs(X) + 0.02 * rms * rng.normal(size=X.shape)
        fit = fit_ols(prob, X, Y)
        rho_hat, Theta = prob.spectrum(fit.beta)
        theta = Theta[:, 0]
        true_rho, se = mc_defect(sm, prob, theta, rng)
        # Size of the model error actually incurred, measured on a dense grid:
        # eta0 bounds |h| and eta1 bounds the Frobenius norm of its Jacobian.
        Xg = rng.uniform(-scale, scale, size=(200_000, 2))
        Ffit = prob.field_of_beta(fit.beta)
        h = sm.rhs(Xg) - prob.alg.eval(Ffit, Xg)
        eta0_true = float(np.linalg.norm(h, axis=1).max())
        Jp = np.stack([prob.alg.eval(_dcol(prob.alg, Ffit, k), Xg) for k in range(2)], axis=2)
        dh = sm.jac(Xg) - Jp
        eta1_true = float(np.linalg.norm(dh, axis=(1, 2)).max())
        for eta_mult in [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]:
            c = certify_direction(prob, theta, fit, ALPHA, "simultaneous",
                                  eta=(eta_mult * eta0_true, eta_mult * eta1_true))
            rows.append(dict(system=key, eta_mult=eta_mult,
                             eta0=eta_mult * eta0_true, eta1=eta_mult * eta1_true,
                             eta0_measured=eta0_true, eta1_measured=eta1_true,
                             upper=float(c.upper),
                             plug_in=float(c.plug_in), true_defect=float(true_rho),
                             true_defect_se=float(se), rms=rms))
    return rows


def main():
    w0 = time.time()
    cells = []
    i = 0
    for name, tdeg, scale in [("vanderpol", 3, 1.0), ("duffing", 3, 2.0),
                              ("hopf", 3, 1.0), ("selkov", 3, 2.0)]:
        for fdeg in (1, 2, 3):
            for N in (400, 1600):
                cells.append((name, tdeg, fdeg, scale, N, 0.02, 3000 + 5171 * i))
                i += 1
    rows = run_cells(truncation_cell, cells, desc="exp3a")
    sp = sparsity_study()
    et = eta_sensitivity()
    save("exp3_modelclass", {"truncation": rows, "sparsity": sp,
                             "eta_sensitivity": et, "trials": TRIALS})
    print(f"exp3 wall {time.time()-w0:.0f}s")


if __name__ == "__main__":
    main()
