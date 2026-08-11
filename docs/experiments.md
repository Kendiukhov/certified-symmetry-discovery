# The experiments, and what each one establishes

Every experiment writes a JSON file to `results/`. `paper/make_tables.py` writes
the LaTeX table bodies from those files, and `paper/check_numbers.py` asserts
every number quoted in the manuscript's running text against them.
`experiments/run_all.py` runs the experiments in order and appends to
`results/compute_log.json`.

| script | question | headline outcome |
|---|---|---|
| `exp1_calibration.py` | Under noise and finite samples, how often does each rule declare a symmetry that is not one, and how often does it find the real one? | No fixed threshold controls the error at any tolerance; valid significance tests are among the worst discovery rules; SymCert's error rate is zero in all 240 settings, at a cost in detection. |
| `exp2_coverage.py` | What happens when the data occupy less of the state space than the claim covers? | Spurious symmetry appears exactly where the extrapolation bound allows it and never where the bound forbids it. Trajectory data on an attractor gives an infinite bound. |
| `exp3_modelclass.py` | What if the fitted model class is too small? | The dominant failure mode: a truncated class has symmetries the system does not, and sparse regression truncates automatically. A lack-of-fit gate restores control. |
| `exp4_resolution.py` | How small a symmetry violation can be certified away, and how does that compare to what is information-theoretically possible? | The certified tolerance follows the Le Cam limit's `sigma/sqrt(N)` rate over two decades, at a constant factor. |
| `exp5_robustness.py` | What happens when the error model is wrong? | Coverage survives Student-t, Laplace and heteroskedastic errors, with and without a sandwich covariance. It is *not* claimed for noisy states with differenced derivatives, which is measured separately. |
| `exp6_dimension.py` | How does the problem's size enter, and should one split the sample? | The model class costs; the candidate class is nearly free. The simultaneous certificate beats the split-sample one in every configuration. |
| `exp7_invariance.py` | Does the machinery work for invariances of a *learned model*? | Yes, with the same guarantee. Training inputs on a sector make the threshold rule declare invariances the target does not have. |
| `exp8_tightness.py` | Is the guarantee tight, or vacuous? | The interval over-covers; at the deliberately hardest boundary the error rate is still zero, and a calibration-targeting bootstrap — which has no guarantee, and does err — is only 9% to 18% narrower. |
| `exp9_modelerror.py` | What does honesty cost when the truth is not polynomial at all? | Paying for the error in supremum norm can be vacuous; paying in `L2` against a richer class is informative and improves with the fitted degree. |
| `exp10_noise.py` | Does noise really not manufacture symmetry? | The plug-in defect is biased upward at every noise level tested; the selection optimism of the chosen direction is small and decays with `N`. |
| `exp11_scaling.py` | Does it scale with the state dimension? | Validity is unaffected as the candidate space grows quadratically; the cost is the model class, and one certificate takes 0.031 CPU-seconds at `n = 6`. |
| `exp12_realdata.py` | What does it say on real measurements? | On the Hudson's Bay lynx--hare series, nothing at all can be certified, which is the answer a threshold rule would not give. |
| `exp13_tolerance.py` | Does the comparison depend on the tolerance? | SymCert's error control holds across two decades of it; its detection rises from 0.25 to 1.00 as the tolerance loosens. |

`exp8_tightness.py` also accepts a stage argument (`coverage`, `boundary` or
`combine`) so the two halves can be run separately on a machine where a long
job is inconvenient.

## Conventions shared by all experiments

**Ground truth is exact.** Each benchmark system has rational coefficients, so
its symmetry algebra is the exact rational nullspace of the bracket map
(`hsd.systems.exact_symmetry_algebra`, computed with sympy) and the true defect
of any generator follows in closed form from monomial moments. No tolerance
enters the ground truth. For linear systems the dimension is cross-checked
against the dimension of the commutant of the coefficient matrix, computed by an
independent route.

**False certification** is the event that the declared *subspace* contains a
generator whose true defect exceeds the tolerance. Because the declared object
is a subspace, this is evaluated as a generalised eigenvalue of the true forms
restricted to it (`experiments/common.py:sup_true_defect`), not by sampling
directions — a rule that declares the right dimension but a rotated subspace is
counted as an error, as it should be.

**Detection** is the event that the declared dimension is at least the true one.

**Noise level** is the standard deviation of the additive error divided by the
root-mean-square size of the vector field over the target domain, so it is
comparable across systems.
