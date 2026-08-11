# Certified Symmetry Discovery

Confidence sets for Lie-algebra generators from noisy, limited data.

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
only when the data *rule out* a defect larger than `delta` — and does so in
closed form, with a guarantee that holds simultaneously for every candidate
generator.

---

## What is here

```
src/hsd/            the library
  polynomials.py    exact polynomial vector-field algebra and closed-form moments
  defect.py         the relative equivariance defect and its bilinear structure
  certify.py        SymCert: simultaneous and pointwise certificates, refutation
  trs.py            exact trust-region extrema of ||a + Bu|| over a ball
  coverage.py       the extrapolation factor kappa
  baselines.py      threshold, eigengap, Weyl bound, rank test, split test
  estimation.py     least squares, confidence ellipsoids, lack-of-fit test
  systems.py        benchmark systems with exactly known symmetry algebras
  invariance.py     the same machinery for invariances of a learned function
  nonpoly.py        non-polynomial reference systems and Monte-Carlo ground truth
experiments/        one script per experiment, plus the runner
figures/            figure script and the compiled figures
paper/              LaTeX source, bibliography, table generator, number checker
results/            raw result files (JSON), the compute log, reported numbers
references/         reference-verification harness and its reports
tests/              correctness tests for the mathematical core
docs/               what each experiment establishes, and shared conventions
notes/              research plan and the verification checklist
```

The paper is `paper/main.pdf`; `paper/main.tex` builds it with `pdflatex`,
`bibtex`, `pdflatex`, `pdflatex`.

## Quick start

```bash
pip install -e ".[experiments,dev]"         # the library alone needs only numpy/scipy/sympy
pytest -q                                  # 30 correctness tests
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
fit = fit_ols(prob, X, Y)

# 3. Look before you leap: how much of the target domain did the data visit?
print(extrapolation_factor(prob, X))   # 'bound' is sqrt(kappa kappa')

# 4. Certify, refute, and read off the tolerance the data actually support.
d, V = certified_dimension(prob, fit, delta=0.05, alpha=0.05)
print(f"certified symmetry dimension: {d}")
print("no delta-symmetry exists for delta below:", refute_all(prob, fit, 0.05))
for k in range(V.shape[1]):
    c = certify_direction(prob, V[:, k], fit, alpha=0.05)
    print(f"  generator {k}: true defect <= {c.upper:.4f} with 95% confidence")
```

`certify_direction` returns a `Certificate` with `upper`, `lower` and `plug_in`.
The honest thing to report is `upper`: it is the smallest tolerance the data
support for that generator, and it is the calibrated replacement for the
singular value one would otherwise quote.

### Invariances of a learned model

`hsd.invariance.InvarianceProblem` exposes the same interface for a scalar
model `f`, where the Lie derivative is `grad f . xi` and the defect is the
root-mean-square cosine of the angle between the model's gradient and the
generator. Every certificate and baseline works unchanged.

## What the guarantee says, and what it needs

**Guarantee.** With probability at least `1 - alpha`, *no* generator whose true
defect on the target domain exceeds `delta` is certified — simultaneously over
the whole candidate space, so the selection rule may be arbitrary and
data-dependent. Exact in finite samples under homoskedastic Gaussian errors on
the observed right-hand sides; asymptotic with a sandwich covariance otherwise.

**Assumptions, in the order they bind.**

1. *The model class contains the truth*, or you supply a bound on how far
   outside it lies (`eta` / `eta_l2` in `certify_direction`). This is the
   binding assumption. A model fitted in too small a class is *more* symmetric
   than the system it models, and no analysis of that model can tell. Use the
   lack-of-fit test in `hsd.estimation.lack_of_fit_pvalue`, and prefer a class
   that is too large — the certificate then widens rather than breaking.
2. *Errors are on the right-hand side, not on the states.* If derivatives are
   differenced from noisy states, the regressors are noisy and the estimator is
   biased; no covariance correction repairs that.
3. *The claim is about the target domain you chose.* Nothing is asserted
   outside it, and `extrapolation_factor` tells you how far the data are from
   supporting it.
4. *Connected symmetry groups only*, with generators in the class you specify.

## Reproducibility

Every experiment is deterministic given the seeds recorded in the scripts.
`results/compute_log.json` records the CPU and wall time of each experiment;
`results/*.json` are the raw outputs. `paper/make_tables.py` regenerates every
table body in the manuscript from those files, and `paper/check_numbers.py`
asserts each of the 31 numbers quoted in the running text against them, so the
text cannot drift away from the data unnoticed.

Ground truth is exact, not approximate: the symmetry algebra of each benchmark
system is the exact rational nullspace of the bracket map, computed with
symbolic arithmetic, and the true defect of any generator follows in closed
form from monomial moments. That matters here, because the quantity being
estimated is itself a near-degeneracy, and a tolerance-based ground truth would
beg the question.

The test suite checks the mathematics rather than the plumbing: the bracket
against finite differences, the closed-form moments against Monte-Carlo
integration, the defect against Monte-Carlo evaluation of its defining
integrals, the exact symmetry algebras against an independent characterisation,
the trust-region extrema against brute-force search, and the certificate's
empirical coverage against its nominal level.

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
