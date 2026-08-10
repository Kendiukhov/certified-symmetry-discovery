# Prose numbers that must be refreshed from results/

Everything in tables and figures is generated. These are the numbers that appear
in running text and therefore have to be checked against `paper/make_tables.py`
output after any re-run.

| file | claim | source |
|---|---|---|
| theory.tex, sec:noise-theory | median / 5% quantile of the smallest plug-in defect at high noise | exp10 `bias` |
| theory.tex, sec:noise-theory | selection optimism gap at N=100 and N=6400 | exp10 `optimism` |
| experiments.tex, sec:calibration | worst-case threshold FCR (63%), detection figures | exp1 (also in the generated table) |
| experiments.tex, sec:coverage | 13 / 37 split of coverage settings | exp2 + exact ground truth |
| experiments.tex, sec:modelclass | STLSQ rates at eps = 0.1 | exp3 `sparsity` |
| experiments.tex, sec:resolution | factor 8.1--8.8 vs the Le Cam bound; refutation rates | exp4 |
| experiments.tex, sec:dimension | width ratios 1.5--2.6; pointwise vs simultaneous | exp6, exp4 |
| experiments.tex, sec:tightness | bootstrap narrower by 18--21%; its error rate at alpha = 0.5 | exp8 `boundary` |
| experiments.tex, sec:robustness | number of settings; detection under noisy states | exp5 |
| experiments.tex, sec:robustness | certified tolerance with and without model error | exp9 |
| experiments.tex, sec:scaling | scaling commentary | exp11 |
| experiments.tex, sec:realdata | spectrum, kappa, smallest certified tolerance | exp12 |
| experiments.tex, sec:invariance | sector false-certification rate; data multiplier | exp7 |
| intro.tex, contributions | total CPU-hours | results/compute_log.json |
