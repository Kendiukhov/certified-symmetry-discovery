# Verification checklist

Everything the paper claims, and where it is checked.

## Mathematics (tests/test_core.py, 30 tests)

| claim | how it is checked |
|---|---|
| the polynomial bracket is the Lie bracket | evaluated against central finite differences of the two vector fields |
| `jac_times` computes `DV W` | same |
| closed-form monomial moments of box / Gaussian / ball / annulus | compared with 4M-sample Monte-Carlo integration on the Cauchy-Schwarz scale |
| the defect equals its defining ratio of integrals | compared with 2M-sample Monte-Carlo evaluation of the numerator and denominator |
| the defect is invariant to rescaling `xi` and `F` | direct |
| a true symmetry has zero defect | exact symmetry algebra of the Hopf form, then evaluate |
| the exact symmetry algebra is right | for five linear systems, compared with the dimension of the commutant of the coefficient matrix computed by an independent Kronecker construction |
| Lorenz has no continuous affine symmetry | exact rational nullspace is empty |
| the sphere flow has so(3) | exact rational nullspace has dimension 3 |
| the trust-region extrema are the true extrema | compared with 200k-sample brute force over the ball, on 12 random instances including rank-deficient ones, and checked against the triangle-inequality bound |
| the simultaneous certificate covers at its nominal level | 300 replications, every direction, coverage measured |
| the subspace certificate dominates every direction in the subspace | 200 random directions per subspace |
| the refutation bound is a valid lower bound | 50 replications against exact ground truth |
| the pointwise certificate covers | 400 replications on a pre-specified direction |
| the Welch--Satterthwaite null used by the baselines | compared with 400k-sample simulation |

## Numbers in the manuscript

`paper/make_tables.py` prints every quoted number and writes each LaTeX table
body from `results/*.json`. No number is transcribed by hand. Run it after any
re-run and compare.

## References

`references/verify_refs.py` queries Crossref, arXiv and DBLP for every entry and
writes `references/verification_report.json`. Entries that did not match at
0.90 similarity were checked by hand against the publisher's record; volume,
issue, page and DOI were taken from the DOI record where one exists.

## Things deliberately *not* claimed

- validity when the fitted model class does not contain the truth (measured, and
  shown to fail, in `exp3`);
- validity when the states are noisy and derivatives are differenced (measured
  in `exp5`; the bias happened to act conservatively there, which is not a
  guarantee);
- that the certificate is tight (measured in `exp8`: it over-covers, by about a
  fifth of its width relative to a calibration-targeting bootstrap);
- that the Le Cam lower bound is attained (it is a lower bound; the constant gap
  is not attributed).
