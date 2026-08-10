# Research plan — Statistically honest symmetry discovery

## Gap
Otto, Zolman, Kutz & Brunton (JMLR 2025, v26/24-1315) reduce Lie-group symmetry
discovery to a nullspace computation: `sym_G(F) = Null(L_F)`, obtained as the
nullspace of the PSD operator `<eta, S xi> = int (L_eta F)^T (L_xi F) dmu`.
Their conclusion explicitly lists as future work: *"It will also be important to
study the perturbative effects of noisy data in algorithms to discover and
promote symmetry, with the goal of understanding the effects of problem
dimension, noise level, and amount of data."*  That is exactly our target.

## Core observations
1. **The burden of proof is backwards.** The standard pipeline declares a
   symmetry when a singular value is *small*. That is a *failure to reject*
   `H0: defect = 0`; it controls the probability of missing a symmetry, not the
   probability of inventing one. Honest discovery needs an **equivalence test**:
   certify `xi` only if we can *reject* `defect >= delta`.
2. **A scale-free defect.** Define the relative equivariance defect
   `rho(xi)^2 = || [xi,F] ||^2 / ( ||DF xi||^2 + ||D xi F||^2 )`, all norms in
   `L^2(nu)`. Invariant to rescaling `xi`, `F`, and to any change of basis of
   the candidate space. `rho = 0` iff exact symmetry. Minimizing `rho` is a
   generalized eigenvalue problem `C theta = lambda D theta`.
3. **Both numerator and denominator are squared norms of *linear* maps of the
   model coefficients `beta`.** Hence a confidence ellipsoid for `beta`
   propagates in closed form (trust-region subproblems) to a **simultaneous
   upper confidence bound `U(theta) >= rho(theta; beta_true)` for all theta at
   once** — family-wise false-certification control with no multiplicity
   correction and no sample splitting.
4. **Limited coverage lands in the right place.** We define the defect on a
   user-chosen target domain `nu` (Gram matrices exact, no MC error); all
   statistical error sits in `Sigma_hat`, which blows up in directions the data
   never excited. The certificate then *refuses to certify* — automatically.
5. **Resolution limit.** A Le Cam two-point argument shows no honest procedure
   can certify tolerances below `~ sigma / sqrt(N)`; our certificate matches
   this rate.

## Deliverables
- `rho`: scale-free defect + generalized eigenproblem formulation.
- SymCert-S: simultaneous certificate (exact finite-sample under Gaussian noise).
- SymCert-P: pointwise/split certificate (tighter, needs sample splitting).
- Lower certificate: refute existence of any delta-symmetry.
- kappa: closed-form extrapolation factor characterising support-induced
  spurious symmetry; `kappa = inf` iff a residual polynomial vanishes on supp(mu).
- Baselines: fixed threshold, eigengap, Weyl/matrix-Bernstein, split significance
  test, rank test (Kleibergen-Paap style).
- Experiments E1-E7 (calibration, power, coverage, misspecification, ML model,
  split-vs-simultaneous, non-Gaussian noise).

## Ground truth
All test systems have rational coefficients; the exact symmetry algebra is the
exact rational nullspace of `theta -> coef([xi_theta, F])`. True defects
`rho(theta; beta_true)` are computed in closed form from exact monomial moments.
