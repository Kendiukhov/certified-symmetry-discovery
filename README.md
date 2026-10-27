# Certified Symmetry Discovery

Confidence sets for Lie-algebra generators of polynomial and dictionary vector
fields from noisy data.

Ihor Kendiukhov, Institute of Medical Genetics and Applied Genomics, University
of Tübingen — `kenduhov.ig@gmail.com`

Continuous symmetries of a dynamical system, or invariances of a learned model,
can be found by linear algebra: a candidate generator `xi` is a symmetry exactly
when the Lie derivative `L_xi F` vanishes, so the symmetries form the nullspace
of a linear operator. With real data that operator is estimated, no singular
value is exactly zero, and a symmetry gets declared when one of them is judged
*small*. That judgement has no error control, and its logic runs backwards:
declaring a symmetry because a test failed to reject it controls the chance of
*missing* a symmetry, not the chance of *inventing* one.

This repository implements **SymCert**, which turns symmetry discovery into an
equivalence test — a generator may be declared a `delta`-approximate symmetry
only when the data *rule out* a defect larger than `delta` — by computing the
exact range of the defect over a confidence region for the model coefficients,
with certified numerics, so that the guarantee holds simultaneously for every
candidate generator and every tolerance.

---

## What is here

```
src/hsd/            the library
  polynomials.py    exact polynomial vector-field algebra and closed-form moments
  defect.py         the relative equivariance defect and its bilinear structure
  certify.py        SymCert: simultaneous and pointwise certificates, refutation
  trs.py            certified trust-region extrema and sharp ratio bounds
                    (Dinkelbach iteration with Lagrangian dual certificates)
  coverage.py       the extrapolation factor kappa
  baselines.py      threshold, eigengap, Weyl bound, rank test, split test
  estimation.py     least squares; OLS, White, cluster-robust (CR3) and
                    Newey-West confidence regions; lack-of-fit test
  selection.py      model-class selection: lack-of-fit gate, certificates
                    simultaneous over nested classes, what a test cannot see
  closure.py        how far a certified subspace is from a Lie subalgebra
  dictionary.py     arbitrary linear-in-parameter dictionaries with quadrature
  dict_systems.py   dictionary benchmark families (radial flow, pendulum)
  trajectories.py   RK4 trajectories cut into contiguous in-domain segments
  systems.py        benchmark systems with exactly known symmetry algebras
  invariance.py     the same machinery for invariances of a learned function
  nonpoly.py        non-polynomial reference systems and Monte-Carlo ground truth
experiments/        one script per experiment, plus the runner
figures/            figure script and the compiled figures
paper/              LaTeX source, bibliography, table generator, number checker
results/            raw result files (JSON), the compute log, reported numbers
data/               processed Global Drifter Program subset (CC BY 4.0)
references/         reference-verification harness and its reports
tests/              correctness tests for the mathematical core
docs/               what each experiment establishes, and shared conventions
notes/              research plan and the verification checklist
submission_srep/    the manuscript in the Scientific Reports format, with the
                    Supplementary Information, cover letter and response
submission/         an earlier typesetting for the journal Machine Learning
```

The paper is `paper/main.pdf` (with the Supplementary Information appended);
`paper/main.tex` builds it with `pdflatex`, `bibtex`, `pdflatex`, `pdflatex`.
`submission_srep/` holds the same manuscript typeset with the Springer Nature
template for *Scientific Reports*, with a separate Supplementary Information
file; `submission_srep/build.sh` regenerates everything from the sources in
`paper/`, so the versions cannot diverge.

## Quick start

```bash
pip install -e ".[experiments,dev]"         # the library alone needs only numpy/scipy/sympy
pytest -q                                  # 59 correctness tests
python experiments/run_all.py              # full suite; see results/compute_log.json
python figures/make_figures.py             # rebuild every figure
python paper/make_tables.py                # regenerate every table body in the paper
python paper/check_numbers.py              # assert every number quoted in the prose
```

## Using it on your own problem

```python
import numpy as np
from hsd import DefectProblem, fit_ols
from hsd.certify import certified_dimension, certify_direction, refute_all
from hsd.coverage import extrapolation_factor

# 1. Say where the symmetry is supposed to hold, and in which classes to look.
prob = DefectProblem.build(
    n=2,                 # state dimension
    deg_F=3,             # degree of the polynomial model class for the dynamics
    generators="affine", # candidate generators xi(x) = Sx + b
    measure="box", scale=1.0,   # the target domain: uniform on [-1, 1]^2
)

# 2. Fit the model to pairs (x_i, y_i) with y_i a noisy observation of F(x_i).
#    The region must match the error structure: cov="ols" (exact under Gaussian
#    homoskedastic errors), "hc0" (White), "cluster" with groups=trajectory_ids,
#    or "hac" (Newey-West) for serially correlated errors.
fit = fit_ols(prob, X, Y, cov="ols")

# 3. How much of the target domain did the data actually visit?
print(extrapolation_factor(prob, X))   # 'bound' is sqrt(kappa kappa')

# 4. Certify, refute, and read off the tolerance the data actually support.
d, V = certified_dimension(prob, fit, delta=0.05, alpha=0.05)
print(f"dimension of the certified plug-in subspace: {d}")
print("no delta-symmetry exists for delta below:", refute_all(prob, fit, 0.05))
for k in range(V.shape[1]):
    c = certify_direction(prob, V[:, k], fit, alpha=0.05)
    print(f"  generator {k}: true defect <= {c.upper:.4f} with 95% confidence")
```

`certify_direction` returns a `Certificate` with `upper`, `lower` and `plug_in`.
Report `upper`: it is the smallest tolerance the data support for that
generator, and the calibrated replacement for the singular value one would
otherwise quote. If the coordinates carry different units, pass characteristic
scales as `metric=[s_1, ..., s_n]` to `DefectProblem.build`; with the default
Euclidean metric the defect, and hence the tolerance, depends on the units.

To choose the polynomial degree from the data, use
`hsd.selection.simultaneous_over_classes(probs, X, Y, delta)` with a nested
list of problems; it is valid when the largest class contains the truth. A
lack-of-fit test that fails to reject is not evidence that a class is right.
`hsd.closure.closure_report(prob, V, fit)` measures how far a certified
subspace is from being closed under the bracket. `hsd.dictionary` replaces
polynomials by any dictionary that is linear in its coefficients.

### Invariances of a learned model

`hsd.invariance.InvarianceProblem` exposes the same interface for a scalar
model `f`, where the Lie derivative is `grad f . xi` and the defect is the
root-mean-square cosine of the angle between the model's gradient and the
generator. Every certificate and baseline works unchanged.

## What the guarantee says, and what it needs

**Guarantee.** If the coefficient region covers the true coefficients with
probability at least `1 - alpha`, then with at least that probability *no*
generator whose true defect on the target domain exceeds `delta` is certified —
simultaneously over the whole candidate space and every tolerance, so the
selection rule may be arbitrary and data-dependent. The certificate inherits
the region's coverage and nothing more: exact in finite samples for the OLS
region under homoskedastic Gaussian errors on the observed right-hand sides;
asymptotic for the White, cluster-robust and Newey-West regions, which need
enough independent rows, trajectories or series length respectively.

**Assumptions, most binding first.**

1. *The model class contains the truth*, or you supply a bound on how far
   outside it lies (`eta` / `eta_l2` in `certify_direction`). This is the
   binding assumption. A model fitted in too small a class is *more* symmetric
   than the system it models, and no analysis of that model can tell. A
   lack-of-fit test (`hsd.estimation.lack_of_fit_pvalue`) can miss
   misspecification that matters, so gating on it carries no guarantee; prefer
   a class that is too large, or certify simultaneously over a nested family
   (`hsd.selection.simultaneous_over_classes`) — the certificate then widens
   rather than breaking.
2. *Errors are on the right-hand side, not on the states.* If derivatives are
   differenced from noisy states, the regressors are noisy and the estimator is
   biased, and the fitted model can look more symmetric than the system; no
   covariance correction repairs that.
3. *The claim is about the target domain you chose.* Nothing is asserted
   outside it, and `extrapolation_factor` tells you how far the data are from
   supporting it.
4. *The tolerance depends on units* unless the output metric is declared.
5. *Connected symmetry groups only*, with generators in the class you specify;
   translations are excluded because the relative defect degenerates for them.

## Reproducibility

Every experiment is deterministic given the seeds recorded in the scripts.
`results/compute_log.json` records the CPU and wall time of each experiment;
`results/*.json` are the raw outputs. `paper/make_tables.py` regenerates every
table body in the manuscript from those files, and `paper/check_numbers.py`
asserts the numerical claims in the running text (113 checks) against them,
so the text cannot drift away from the data unnoticed.

Ground truth is exact, not approximate: the symmetry algebra of each benchmark
system is the exact rational nullspace of the bracket map, computed with
symbolic arithmetic, and the true defect of any generator follows in closed
form from monomial moments. The quantity being estimated is itself a
near-degeneracy, so a tolerance-based ground truth would beg the question.

The test suite checks the mathematics rather than the plumbing: the bracket
against finite differences, the closed-form moments against Monte-Carlo
integration, the defect against Monte-Carlo evaluation of its defining
integrals, the exact symmetry algebras against an independent characterisation,
the trust-region extrema and sharp ratio bounds against brute-force search,
the invariance of the defect under changes of units with a transported metric,
the dictionary construction against the polynomial one, the coverage of the
coefficient regions and of the certificate against their nominal levels, and
the closure diagnostic on exact Lie algebras.

## References

The framework this builds on is S. E. Otto, N. Zolman, J. N. Kutz and
S. L. Brunton, *A Unified Framework to Enforce, Discover, and Promote Symmetry
in Machine Learning*, JMLR 26(248):1–83, 2025.
Their conclusion names the problem addressed here: "It will also be important to
study the perturbative effects of noisy data in algorithms to discover and
promote symmetry."

All 67 bibliography entries were checked against Crossref, arXiv and DBLP
metadata by `references/verify_refs.py`, whose entry list is generated from
`paper/refs.bib` itself by `references/sync_entries.py`. The machine report is
`references/verification_report.json`; entries that the automatic sweep could
not match — mostly books, which have no Crossref record at the level of the whole
work — were confirmed individually and the checks recorded in
`references/manual_checks.md`.

## License

MIT. See `LICENSE`.
