# The experiments, and what each one establishes

Every experiment writes a JSON file to `results/`. `paper/make_tables.py` writes
the LaTeX tables of the manuscript and the Supplementary Information from those
files, `figures/make_figures.py` draws every figure, and `paper/check_numbers.py`
asserts every number quoted in the running text against them.
`experiments/run_all.py` runs the experiments in order and appends to
`results/compute_log.json`.

| script | question | headline outcome |
|---|---|---|
| `exp1_calibration.py` | Under noise and finite samples, how often does each rule declare a symmetry that is not one, and how often does it find (and correctly orient) the real one? | None of the thresholds examined controls the worst-case error; valid significance tests are poor discovery rules; SymCert's error rate is zero in all 240 settings, at a cost in detection. |
| `exp2_coverage.py` | What happens when the data occupy less of the state space than the claim covers? | Spurious symmetry appears only where the extrapolation bound allows it. Trajectory data on an attractor gives an infinite bound. |
| `exp3_modelclass.py` | What if the fitted model class is too small, or sparsified? | A truncated class has symmetries the system does not; sparse regression truncates automatically. |
| `exp4_resolution.py` | How small a violation can be certified, against a non-asymptotic Le Cam bound, and how does the gap depend on the model class? | Same `sigma/sqrt(N)` scaling at a fixed class; the gap grows with the number of coefficients. |
| `exp5_robustness.py` | What happens when the error model is wrong, and on trajectory data? | Region coverage is reported alongside certificate coverage; noisy states with differenced derivatives break the region. |
| `exp6_dimension.py` | How does the problem's size enter, and should one split the sample? | The model class costs; the candidate class is nearly free. The simultaneous certificate beats the split-sample one in every configuration. |
| `exp7_invariance.py` | Does the machinery work for invariances of a *learned model*? | Yes. Training inputs on a sector make the threshold rule declare invariances the target does not have. |
| `exp8_tightness.py` | How conservative is the certificate, and what does a finite-sample guarantee cost against a parametric bootstrap? | Coverage, widths (sharp, separate, bootstrap), bootstrap coverage near a symmetry, and false certification at the tolerance boundary. |
| `exp9_modelerror.py` | What does honesty cost when the truth is not polynomial at all? | Oracle and data-driven model-error bounds for two non-polynomial systems. |
| `exp10_noise.py` | Does additive derivative noise manufacture symmetry? | No: the plug-in defect is biased upward; the selection optimism is small and decays with `N`. |
| `exp11_scaling.py` | Does it scale with the state dimension? | Validity is unaffected as the candidate space grows quadratically; the cost is the model class. |
| `exp12_realdata.py` | Lynx--hare pelt records | Nineteen observations support no claim. |
| `exp13_tolerance.py` | Does the comparison depend on the tolerance? | SymCert's error control holds across two decades of tolerance. |
| `exp14_selection.py` | What happens when the model class is chosen from the data (lack-of-fit gate, split, largest class, simultaneous over classes)? | The gate can certify falsely when the omitted term is of high degree; the valid procedures do not. |
| `exp15_dependence.py` | Serially correlated errors along trajectories: which regions cover? | OLS and White regions undercover; cluster-robust and Newey--West regions are designed for this. |
| `exp16_dictionary.py` | Non-polynomial systems fitted with dictionaries that contain them. | Same validity as for polynomials. |
| `exp17_units_closure.py` | Do units matter, and are certified subspaces closed under the bracket? | A change of units can manufacture symmetry under the Euclidean metric; a declared metric removes the effect. |
| `exp18_eiv.py` | Noisy states (errors in variables). | Attenuation makes the fit look symmetric; the certificate can then certify falsely. |
| `exp19_drifters.py` | Real ocean drifter velocities: is near-inertial motion rotation invariant? | Certified to a stated tolerance with a trajectory-clustered region. Needs `data/gdp_np_drogued_2016_2019.csv.gz` (regenerated from NOAA ERDDAP if absent). |

`exp8_tightness.py` accepts a stage argument (`coverage`, `width`, `small`,
`boundary` or `combine`) so the stages can be run separately.

## Conventions shared by all experiments

**Ground truth is exact.** Each polynomial benchmark has rational coefficients,
so its symmetry algebra is the exact rational nullspace of the bracket map
(`hsd.systems.exact_symmetry_algebra`, computed with sympy) and the true defect
of any generator follows in closed form from monomial moments. For linear
systems the dimension is cross-checked against the dimension of the commutant
of the coefficient matrix, computed by an independent route. For the
dictionary families the bilinear forms come from exact (or rounding-level)
quadrature, and the algebra is the numerical nullspace with a gap of many
orders of magnitude.

**False certification** is the event that the declared *subspace* contains a
generator whose true defect exceeds the tolerance, evaluated as a generalised
eigenvalue of the true forms restricted to it
(`experiments/common.py:sup_true_defect`), not by sampling directions.

**Detection** is the event that the declared dimension is at least the true one;
**recovery** additionally requires the declared subspace to contain the true
algebra to within 10 degrees in the `L2(nu)` geometry of generator fields.

**Monte Carlo uncertainty** is reported with exact Clopper--Pearson intervals.

**Noise level** is the standard deviation of the additive error divided by the
root-mean-square size of the vector field over the target domain.

**Confidence region.** The exact least-squares ellipsoid unless stated
otherwise; `exp5` also uses White's sandwich and the cluster-robust sandwich,
`exp15` White, cluster-robust and Newey--West, `exp12` White, `exp19` all four.

**Smoke test.** `HSD_SMOKE=2 HSD_RESULTS=/tmp/x python experiments/run_all.py`
runs every experiment with two replications per cell into another directory.
