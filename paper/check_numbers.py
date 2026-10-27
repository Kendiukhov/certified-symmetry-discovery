"""Cross-check every number quoted in the manuscript's running text.

Tables and figures are generated from ``results/`` and cannot drift. The numbers
that appear in prose (abstract, Introduction, Results, Discussion, Methods,
figure and table captions, Supplementary Notes and captions) can, so this
script asserts each of them against the raw result files. A few quoted numbers
are not stored in a result file (the lynx--hare analysis on the positive
quadrant, the extrapolation factor of the sector inputs, the invisible defect
with an exact denominator, the drifter record lengths); they are recomputed
here from the released code and data.
Design constants quoted in the text are checked against the result files where
these record them, and otherwise against the experiment scripts. Run it after
any re-run; a failure means the text and the data have separated.
"""

from __future__ import annotations

import inspect
import json
import pathlib
import sys
from itertools import product
from math import comb

import numpy as np
import pandas as pd
from scipy import stats

ROOT = pathlib.Path(__file__).resolve().parents[1]
RES = ROOT / "results"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))


def load(name):
    return json.loads((RES / f"{name}.json").read_text())


def near(x, target, tol):
    return abs(float(x) - target) <= tol


def cp(k, T):
    """Exact two-sided 95% Clopper--Pearson interval for k events in T."""
    lo = 0.0 if k == 0 else stats.beta.ppf(0.025, k, T - k + 1)
    hi = 1.0 if k == T else stats.beta.ppf(0.975, k + 1, T - k)
    return float(lo), float(hi)


def main() -> int:
    checks: list[tuple[str, bool, str]] = []
    sg_ = inspect.signature

    def chk(label, ok, detail=""):
        checks.append((label, bool(ok), str(detail)))

    # ------------------------------------------------------------- exp1
    # Abstract, Introduction, Calibration and power, Table 1, Fig. 2a,
    # Supplementary Fig. S4, Methods (benchmarks).
    d1 = load("exp1_calibration")
    df = pd.DataFrame(d1["rows"])
    sym = df[df.d_true > 0]
    T = d1["trials"]
    chk("ten systems, four with symmetry and six without",
        df.system.nunique() == 10 and sym.system.nunique() == 4
        and df[df.d_true == 0].system.nunique() == 6)
    chk("true algebra dimensions: Hopf 1, rotation 2, symmetric top 2, sphere flow 3",
        df.groupby("system").d_true.first()[["hopf", "lin_rot2", "rigid_sym", "sphere_flow3"]].tolist()
        == [1, 2, 2, 3])
    chk("240 settings = 10 systems x 6 sample sizes x 4 noise levels, 300 replications, 72,000",
        len(df) == 240 and len(d1["Ns"]) == 6 and len(d1["sigmas"]) == 4 and T == 300
        and len(df) * T == 72000)
    chk("calibration grid N = 100..3200, noise 0.01, 0.03, 0.1, 0.3",
        d1["Ns"] == [100, 200, 400, 800, 1600, 3200] and d1["sigmas"] == [0.01, 0.03, 0.1, 0.3])
    chk("96 symmetric settings (Table 1 caption)", len(sym) == 96)
    chk("threshold swept over tau in [0.005, 0.4]", min(d1["taus"]) == 0.005 and max(d1["taus"]) == 0.4)
    chk("delta = alpha = 0.05", d1["_meta"]["alpha"] == 0.05 and d1["_meta"]["delta"] == 0.05)
    chk("SymCert false certification is 0 in every setting", df.fc_sym.max() == 0)
    chk("one-sided bound 0.010 per setting", near(1 - 0.05 ** (1 / T), 0.010, 0.0005))
    chk("simultaneous bound 0.028 over 240 settings",
        near(1 - (0.05 / len(df)) ** (1 / T), 0.028, 0.0005))
    worst = df["fc_tau0.05"].max()
    lo, hi = cp(int(round(worst * T)), T)
    chk("threshold worst setting 0.63 [0.58, 0.69] (abstract: 63%)",
        near(worst, 0.633, 0.002) and near(lo, 0.58, 0.005) and near(hi, 0.69, 0.005),
        f"{worst:.3f} [{lo:.3f},{hi:.3f}]")
    chk("every threshold in 0.005-0.4 has worst >= 0.63",
        min(df[f"fc_tau{t}"].max() for t in d1["taus"]) >= 0.633 - 1e-9)
    chk("worst case 0.63 at every threshold up to 0.1 (Fig. S4 caption)",
        all(near(df[f"fc_tau{t}"].max(), 0.633, 0.001) for t in d1["taus"] if t <= 0.1))
    chk("threshold tau=0.05 averages 0.5% (abstract, Introduction)",
        near(df["fc_tau0.05"].mean(), 0.005, 0.0006), f"{df['fc_tau0.05'].mean():.4f}")
    chk("threshold small-tau average 0.005", near(df["fc_tau0.005"].mean(), 0.005, 0.0006))
    chk("detection 0.42 at tau=0.005", near(sym["det_tau0.005"].mean(), 0.42, 0.005))
    chk("SymCert detects 48.5%, threshold at the tolerance 78.4% (abstract, Results)",
        near(sym.det_sym.mean(), 0.485, 0.0005) and near(sym["det_tau0.05"].mean(), 0.784, 0.0005),
        f"{sym.det_sym.mean():.4f} / {sym['det_tau0.05'].mean():.4f}")
    chk("for both rules recovery equals detection",
        abs(sym.det_sym.mean() - sym.rec_sym.mean()) < 1e-12
        and abs(sym["det_tau0.05"].mean() - sym["rec_tau0.05"].mean()) < 1e-12)
    fcs = df[[f"fc_tau{t}" for t in d1["taus"] if t <= 0.05]].max(axis=1)
    off, above = df[fcs > 0], df[fcs > 0.05]
    chk("threshold errors (tau <= delta) all on symmetric systems at high noise (0.1, 0.3);"
        " settings above alpha all at noise 0.3",
        len(off) > 0 and (off.d_true > 0).all() and (off.sigma_rel >= 0.1).all()
        and len(above) > 0 and (above.sigma_rel == 0.3).all())
    chk("threshold at the tolerance never certified falsely on a system without symmetry",
        df[df.d_true == 0]["fc_tau0.05"].max() == 0)
    wsets = {t: df.loc[df[f"fc_tau{t}"].idxmax(), ["system", "sigma_rel", "N"]].tolist()
             for t in d1["taus"] if t <= 0.1}
    chk("worst setting for every threshold up to 0.1: planar rotation, highest noise (0.3), N = 100"
        " (Results; Introduction: 63% on the planar rotation at the highest noise level)",
        all(w == ["lin_rot2", 0.3, 100] for w in wsets.values())
        and max(d1["sigmas"]) == 0.3, wsets)
    ab = {(r_.system, r_.sigma_rel, r_.N) for t in d1["taus"] if t <= 0.05
          for _, r_ in df[df[f"fc_tau{t}"] > 0.05].iterrows()}
    rr5 = df[(df.system == "lin_rot2") & (df.sigma_rel == 0.3)].set_index("N")["fc_tau0.05"]
    chk("tau <= delta: the only settings above alpha are the rotation at noise 0.3, N = 100, 200, 400"
        " (rates 0.63, 0.34, 0.12)",
        ab == {("lin_rot2", 0.3, 100), ("lin_rot2", 0.3, 200), ("lin_rot2", 0.3, 400)}
        and near(rr5[100], 0.63, 0.005) and near(rr5[200], 0.34, 0.005) and near(rr5[400], 0.12, 0.005),
        rr5[[100, 200, 400]].round(3).tolist())
    rem = df[(df["fc_tau0.05"] > 0)
             & ~((df.system == "lin_rot2") & (df.sigma_rel == 0.3) & df.N.isin([100, 200, 400]))]
    chk("remaining errors at tau = delta at most 0.04, on symmetric systems at noise 0.1 and 0.3",
        near(rem["fc_tau0.05"].max(), 0.04, 0.0005) and (rem.d_true > 0).all()
        and set(rem.sigma_rel) == {0.1, 0.3}, f"{rem['fc_tau0.05'].max():.4f}")
    chk("thresholds above the tolerance certify falsely on systems without symmetry such as Lorenz",
        df[(df.d_true == 0) & (df.system == "lorenz")][[f"fc_tau{t}" for t in d1["taus"] if t > 0.05]]
        .max().max() > 0)
    chk("worst threshold settings recover (rec=1) yet err",
        df.loc[df["fc_tau0.05"].idxmax(), "rec_tau0.05"] == 1.0)
    chk("recovery and detection differ by < 0.002 except split test, not at all for SymCert",
        all(abs(sym[f"det_{m}"].mean() - sym[f"rec_{m}"].mean()) < 0.002
            for m in ("gap", "rank", "weyl", "sym", "symp"))
        and all(abs(sym[f"det_tau{t}"].mean() - sym[f"rec_tau{t}"].mean()) < 0.002
                for t in d1["taus"])
        and (sym.det_sym == sym.rec_sym).all())
    chk("split test detects 0.84, recovers 0.22",
        near(sym.det_split.mean(), 0.84, 0.006) and near(sym.rec_split.mean(), 0.22, 0.006))
    chk("rank test worst 1.00, mean 0.070; '7% on average, every replication of worst setting'",
        df.fc_rank.max() == 1.0 and near(df.fc_rank.mean(), 0.070, 0.0006))
    chk("split test worst 0.92, mean 0.070",
        near(df.fc_split.max(), 0.92, 0.002) and near(df.fc_split.mean(), 0.070, 0.0008))
    chk("eigengap reaches 100%", df.fc_gap.max() == 1.0)
    chk("Fig. 2a: rank test lies on the tau = 0.4 point",
        df.fc_rank.max() == df["fc_tau0.4"].max()
        and near(sym.rec_rank.mean(), sym["rec_tau0.4"].mean(), 0.005))
    chk("Fig. 2a: SymCert and splitting variant at zero, variant with lower recovery",
        df.fc_symp.max() == 0 and sym.rec_symp.mean() < sym.rec_sym.mean())
    s01 = sym[sym.sigma_rel == 0.01].groupby("N").det_sym.mean()
    chk("detection reaches 1.00 by N=400 at noise 0.01 (text, Fig. 3a)",
        all(s01[n] == 1.0 for n in s01.index if n >= 400) and s01[200] < 1.0)
    s30 = sym[sym.sigma_rel == 0.3].groupby("N").det_sym.mean()
    chk("at noise 0.3 detection stays below 1", s30.max() < 1.0)
    chk("Weyl certificate never certifies", df.dim_weyl.max() == 0 and df.fc_weyl.max() == 0)
    import exp1_calibration as x1
    deg = {nm: dg for nm, dg, _ in x1.SYSTEMS}
    half = {nm: sc for nm, _, sc in x1.SYSTEMS}
    chk("Methods: class degree 1 rotation, 2 rigid bodies/LV/Lorenz, 3 otherwise",
        deg["lin_rot2"] == 1
        and all(deg[s] == 2 for s in ("rigid_sym", "rigid_asym", "lotka_volterra", "lorenz"))
        and all(deg[s] == 3 for s in ("hopf", "sphere_flow3", "vanderpol", "duffing", "selkov")))
    chk("Methods: box half-width 2 for Duffing and Sel'kov, 20 for Lorenz, 1 otherwise",
        half["duffing"] == 2 and half["selkov"] == 2 and half["lorenz"] == 20
        and all(v == 1 for k, v in half.items() if k not in ("duffing", "selkov", "lorenz")))

    # ------------------------------------------------------------- exp13
    d13 = load("exp13_tolerance")
    e13 = pd.DataFrame(d13["rows"])
    sy13 = e13[e13.d_true > 0]
    chk("tolerance grid (Results, Fig. S4 caption): delta in {0.01, 0.02, 0.05, 0.1, 0.2, 0.4}; Hopf,"
        " rotation, symmetric top (symmetric) and Van der Pol, Lotka-Volterra, asymmetric rigid body;"
        " N = 400 and 3200; noise 0.03 and 0.1; 300 reps",
        d13["deltas"] == [0.01, 0.02, 0.05, 0.1, 0.2, 0.4]
        and set(sy13.system) == {"hopf", "lin_rot2", "rigid_sym"}
        and set(e13[e13.d_true == 0].system) == {"vanderpol", "lotka_volterra", "rigid_asym"}
        and sorted(e13.N.unique()) == [400, 3200] and sorted(e13.sigma_rel.unique()) == [0.03, 0.1]
        and d13["trials"] == 300)
    chk("tolerance sweep: SymCert 0 everywhere, detection on the three symmetric systems 0.25 -> 1.00 at 0.2",
        max(e13[f"symfc_{x}"].max() for x in d13["deltas"]) == 0
        and near(sy13["symdet_0.01"].mean(), 0.25, 0.006) and near(sy13["symdet_0.2"].mean(), 1.0, 0.001))
    s1_in = sym[sym.system.isin(set(sy13.system)) & sym.N.isin([400, 3200]) & sym.sigma_rel.isin([0.03, 0.1])]
    s1_out = sym[(sym.N < 400) | (sym.sigma_rel == 0.3)]
    chk("grid omits N < 400 and noise 0.3, where detection at delta = 0.05 is lower in the main study",
        s1_out.det_sym.mean() < s1_in.det_sym.mean(),
        f"{s1_out.det_sym.mean():.3f} vs {s1_in.det_sym.mean():.3f}")
    chk("Fig. S4: threshold above nominal only at the tightest tolerance",
        e13["taufc_0.01"].max() > 0.05
        and all(e13[f"taufc_{x}"].max() < 0.05 for x in d13["deltas"] if x != 0.01))

    # ------------------------------------------------------------- exp2
    # Fig. 2b, Coverage paragraph, Introduction, Fig. 4, Table S3.
    d2 = load("exp2_coverage")
    e2 = pd.DataFrame(d2["rows"])
    r = e2.kappa / e2.bound_critical
    chk("50 coverage settings, five systems, 200 replications, N=2000, noise 0.02",
        len(e2) == 50 and e2.system.nunique() == 5 and d2["trials"] == 200 and d2["N"] == 2000
        and d2["sigma_rel"] == 0.02)
    chk("data box shrunk from full width to 0.15 of the half-width", max(d2["radii"]) == 1.0
        and min(d2["radii"]) == 0.15)
    chk("13 coverage settings below the bound, none fail; six of them without symmetry",
        (r < 1).sum() == 13 and e2[r < 1].fc_naive_data.max() == 0
        and ((r < 1) & (e2.d_true == 0)).sum() == 6)
    beyond = e2[(r >= 1) & (e2.fc_naive_data > 0)].system.value_counts()
    chk("37 above the bound, 14 fail (6 Van der Pol, 5 Hopf, 3 symmetric top), up to 1.0",
        (r >= 1).sum() == 37 and beyond.sum() == 14 and beyond.get("vanderpol", 0) == 6
        and beyond.get("hopf", 0) == 5 and beyond.get("rigid_sym", 0) == 3
        and e2.fc_naive_data.max() == 1.0, beyond.to_dict())
    chk("SymCert never certifies falsely under coverage loss", e2.fc_sym.max() == 0)
    hop = e2[e2.system == "hopf"].set_index("radius").dim_sym
    chk("Hopf: certifies when covered, almost never below half the domain",
        hop[1.0] == 1.0 and all(hop[x] <= 0.1 for x in hop.index if x <= 0.6))
    chk("Fig. 4: incomplete coverage (data in <= 30% of box width): threshold errs, SymCert 0",
        e2[e2.radius <= 0.3].fc_naive_data.max() > 0.05 and e2[e2.radius <= 0.3].fc_sym.max() == 0)
    tr2 = pd.DataFrame(d2["trajectories"]).set_index(["system", "design"])
    chk("Hopf single trajectory has infinite factor",
        not np.isfinite(tr2.loc[("hopf", "single trajectory"), "kappa"]))
    chk("single Van der Pol and Lorenz trajectories: 34.9 and 231.3",
        near(tr2.loc[("vanderpol", "single trajectory"), "kappa"], 34.9, 0.05)
        and near(tr2.loc[("lorenz", "single trajectory"), "kappa"], 231.3, 0.05))
    chk("eight trajectories 24, uniform 1.1",
        near(tr2.loc[("hopf", "8 trajectories"), "kappa"], 24.0, 0.05)
        and near(tr2.loc[("hopf", "uniform design"), "kappa"], 1.1, 0.05))
    chk("eigenvalue threshold 1e-10 for a singular data Gram matrix",
        "1e-10 * max(wmu.max()" in inspect.getsource(
            __import__("hsd.coverage", fromlist=["x"]).extrapolation_factor))

    # ------------------------------------------------------------- exp3
    d3 = load("exp3_modelclass")
    tr = pd.DataFrame(d3["truncation"])
    sp = pd.DataFrame(d3["sparsity"])
    trunc = tr[tr.fit_degree < tr.true_degree]
    chk("truncation: linear or quadratic fits to cubic systems",
        set(trunc.fit_degree) == {1, 2} and set(trunc.true_degree) == {3})
    chk("truncation: threshold and SymCert reach 100%",
        trunc.fc_naive.max() == 1.0 and trunc.fc_sym.max() == 1.0)
    chk("lack-of-fit rejects 100% in every gross truncation", trunc.lof_reject.min() == 1.0)
    src3 = (ROOT / "experiments" / "exp3_modelclass.py").read_text()
    import exp3_modelclass as x3
    chk("Methods: truncation fits Van der Pol, Duffing, Hopf, Sel'kov on their boxes, degrees 1, 2, 3,"
        " N in {400, 1600}, noise 0.02, 300 reps; lack-of-fit against the cubic class",
        set(tr.system) == {"vanderpol", "duffing", "hopf", "selkov"} and set(tr.fit_degree) == {1, 2, 3}
        and set(tr.true_degree) == {3} and set(tr.N) == {400, 1600} and set(tr.sigma_rel) == {0.02}
        and d3["trials"] == 300 and len(tr) == 24
        and '("vanderpol", 3, 1.0), ("duffing", 3, 2.0)' in src3
        and '("hopf", 3, 1.0), ("selkov", 3, 2.0)' in src3
        and "lack_of_fit_pvalue(prob_fit, prob_true, X, Y)" in src3)
    t1, t2 = tr[tr.fit_degree == 1], tr[tr.fit_degree == 2]
    chk("affine fits: threshold declares the n-dimensional affine algebra in every replication,"
        " SymCert certifies it in up to 100%",
        (t1.fc_naive == 1.0).all() and (t1.dim_naive == 2.0).all() and t1.fc_sym.max() == 1.0)
    t2s = t2[t2.fc_sym > 0]
    chk("quadratic fits: threshold false in 70% to 100%; SymCert certifies one in a single setting"
        " (Van der Pol, N = 1600)",
        near(t2.fc_naive.min(), 0.70, 0.005) and t2.fc_naive.max() == 1.0
        and len(t2s) == 1 and t2s.system.iloc[0] == "vanderpol" and t2s.N.iloc[0] == 1600
        and t2s.dim_sym.iloc[0] == 1.0, f"{t2.fc_naive.min():.3f}")
    chk("Methods: sparsity study eps in {0, 0.02, 0.05, 0.1, 0.2, 0.4}, thresholds 0/0.05/0.1/0.25,"
        " ten STLSQ rounds, N = 1000, noise 0.05, 300 reps, [-1,1]^2, cubic class",
        sorted(sp.eps.unique()) == [0.0, 0.02, 0.05, 0.1, 0.2, 0.4]
        and sorted(sp.threshold.unique()) == [0.0, 0.05, 0.1, 0.25]
        and sg_(x3.stlsq).parameters["iters"].default == 10
        and "N=1000" in inspect.getsource(x3.sparsity_study)
        and "0.05 * rms_field" in inspect.getsource(x3.sparsity_study)
        and 'DefectProblem.build(2, 3, "affine", "box", scale=1.0)' in inspect.getsource(x3.sparsity_study))
    s1 = sp[sp.eps == 0.1].set_index("threshold")
    chk("rotation defect 0.064 at eps=0.1", near(s1.true_rot_defect.iloc[0], 0.064, 0.0005))
    chk("STLSQ threshold 0.25 reports symmetry in 100%", s1.fc_stlsq[0.25] == 1.0)
    chk("no thresholding: 17%", near(s1.fc_stlsq[0.0], 0.17, 0.005))
    chk("unpruned SymCert certifies nothing false",
        sp.dim_symcert[sp.eps >= 0.1].max() == 0.0)

    # ------------------------------------------------------------- exp18 (Fig. 4)
    e18 = pd.DataFrame(load("exp18_eiv")["rows"])
    chk("Fig. 4: state-noise s.d. up to 0.12", e18.state_noise.max() == 0.12)

    # ------------------------------------------------------------- exp10
    d10 = load("exp10_noise")
    bi = pd.DataFrame(d10["bias"])
    op = pd.DataFrame(d10["optimism"])
    chk("Van der Pol, Lotka-Volterra, Duffing, asymmetric rigid body; noise 0.1/0.3/0.5/1.0;"
        " N = 30, 60, 120, 400; 500 reps",
        set(bi.system) == {"vanderpol", "lotka_volterra", "duffing", "rigid_asym"}
        and sorted(bi.sigma_rel.unique()) == [0.1, 0.3, 0.5, 1.0]
        and sorted(bi.N.unique()) == [30, 60, 120, 400] and d10["trials"] == 500)
    tmd = bi.groupby("system").true_min_defect.first()
    chk("smallest true defects 0.21 to 0.29, at least four times the tolerance",
        near(tmd.min(), 0.21, 0.005) and near(tmd.max(), 0.29, 0.005) and tmd.min() >= 4 * 0.05,
        tmd.round(3).to_dict())
    chk("32,000 noise replications", len(bi) * d10["trials"] == 32000)
    bmin = bi.loc[bi.minimum.idxmin()]
    chk("smallest plug-in defect observed 0.066 (Lotka-Volterra, true value 0.271), above the tolerance",
        near(bmin.minimum, 0.066, 0.0006) and bmin.system == "lotka_volterra"
        and near(bmin.true_min_defect, 0.271, 0.0006) and bmin.minimum > 0.05)
    med = bi.groupby("sigma_rel").apply(
        lambda gg: float(np.median(gg["median"] / gg.true_min_defect)), include_groups=False)
    chk("median ratios pooled over systems and sample sizes 1.01, 1.09, 1.27, 1.64",
        all(near(med[s], v, 0.006) for s, v in zip(sorted(med.index), (1.01, 1.09, 1.27, 1.64))),
        med.round(3).to_dict())
    bi["ratio"] = bi["median"] / bi.true_min_defect
    three = bi[bi.system.isin(["vanderpol", "duffing", "rigid_asym"])]
    rises = all(np.all(np.diff(g.sort_values("sigma_rel").ratio.values) > 0)
                for _, g in three.groupby(["system", "N"]))
    top = three[(three.sigma_rel == 1.0) & (three.N == 30)].set_index("system").ratio
    chk("Van der Pol, Duffing, rigid body: median rises with noise, to 2.0-2.4 times at noise 1.0, N = 30",
        rises and near(top.min(), 2.0, 0.05) and near(top.max(), 2.4, 0.05), top.round(3).to_dict())
    lvb = bi[bi.system == "lotka_volterra"]
    chk("Lotka-Volterra: median within 1.00 to 1.14 times, 31% to 59% of replications below the true value",
        near(lvb.ratio.min(), 1.00, 0.005) and near(lvb.ratio.max(), 1.14, 0.005)
        and near(lvb.frac_below_true.min(), 0.31, 0.005) and near(lvb.frac_below_true.max(), 0.59, 0.005),
        f"{lvb.ratio.min():.4f}-{lvb.ratio.max():.4f}, "
        f"{lvb.frac_below_true.min():.3f}-{lvb.frac_below_true.max():.3f}")
    # Near-tolerance study (Results, Methods, Introduction, theory section).
    nr = pd.DataFrame(d10["near"])
    import exp10_noise as x10
    chk("near tolerance: broken-Hopf at 1.2, 1.5, 2, 4 delta and hardening Duffing (0.102);"
        " N = 100/400/1600; noise 0.05/0.1/0.2/0.3; 300 reps; 60 settings; 18,000 in all",
        d10["near_multiples"] == [1.2, 1.5, 2.0, 4.0]
        and all(near(nr[nr.system == str(m)].multiple.iloc[0], m, 1e-6) for m in d10["near_multiples"])
        and near(nr[nr.system == "hard_duffing"].true_min_defect.iloc[0], 0.102, 0.0006)
        and near(nr[nr.system == "hard_duffing"].true_min_defect.iloc[0],
                 e18[(e18.system == "hard_duffing")].true_min_defect.iloc[0], 1e-9)
        and sorted(nr.N.unique()) == [100, 400, 1600] and sorted(nr.sigma_rel.unique()) == [0.05, 0.1, 0.2, 0.3]
        and d10["near_trials"] == 300 and (nr.trials == 300).all() and len(nr) == 60
        and int(nr.trials.sum()) == 18000)
    nsrc = inspect.getsource(x10.near_cell) + inspect.getsource(x10._near_system)
    chk("Methods: near-tolerance systems on [-1,1]^2 in the cubic class, defect in the affine class",
        nsrc.count('DefectProblem.build(2, 3, "affine", "box", scale=1.0)') == 2
        and 'sampler="box", scale=1.0' in nsrc)
    nmax = nr.groupby("system").fc_threshold.max()
    w37 = nr.loc[nr.fc_threshold.idxmax()]
    l37, h37 = cp(int(w37.count_threshold), int(w37.trials))
    chk("threshold up to 3.7% at 1.2 delta (N = 100, noise 0.05; 95% interval 1.8% to 6.5%)",
        near(w37.fc_threshold, 0.037, 0.0005) and w37.system == "1.2" and w37.N == 100
        and w37.sigma_rel == 0.05 and near(l37, 0.018, 0.0005) and near(h37, 0.065, 0.0005),
        f"{int(w37.count_threshold)}/300 [{l37:.4f}, {h37:.4f}]")
    chk("at most 0.7% at 1.5 delta, 0.3% at 2 delta and on the hardening Duffing, never at 4 delta",
        near(nmax["1.5"], 0.007, 0.0005) and near(nmax["2.0"], 0.003, 0.0005)
        and near(nmax["hard_duffing"], 0.003, 0.0005) and nmax["4.0"] == 0, nmax.round(4).to_dict())
    chk("SymCert certified nothing in any of the 18,000 near-tolerance replications",
        nr.count_symcert.sum() == 0 and nr.fc_symcert.max() == 0)
    chk("false declarations only within about a factor of two of delta, never at four times"
        " (Introduction, theory section: at most 3.7%)",
        nr[nr.fc_threshold > 0].multiple.max() < 2.1 and nr[nr.multiple > 3].fc_threshold.max() == 0
        and near(nr.fc_threshold.max(), 0.037, 0.0005))
    wmr = nr.loc[nr.median_ratio.idxmax()]
    chk("smallest plug-in defect below its true value in up to 52%; median up to 3.7 times"
        " at high noise and small N",
        near(nr.frac_below_true.max(), 0.52, 0.005) and near(wmr.median_ratio, 3.7, 0.05)
        and wmr.N == nr.N.min() and wmr.sigma_rel == nr.sigma_rel.max(),
        f"{nr.frac_below_true.max():.3f}, {wmr.median_ratio:.3f}")
    chk("optimism study: same four systems, noise 0.2, N doubling 100 to 6400",
        set(op.system) == set(bi.system) and set(op.sigma_rel) == {0.2}
        and sorted(op.N.unique()) == [100 * 2 ** k for k in range(7)])
    o100 = op[op.N == 100].set_index("system")
    chk("optimism values at N=100 (means +/- standard errors)",
        near(o100.gap_mean["vanderpol"], 0.0150, 0.00006) and near(o100.gap_se["vanderpol"], 0.0017, 0.00006)
        and near(o100.gap_mean["lotka_volterra"], 0.0078, 0.00006)
        and near(o100.gap_se["lotka_volterra"], 0.0004, 0.00006)
        and near(o100.gap_mean["duffing"], 0.0060, 0.00006) and near(o100.gap_se["duffing"], 0.0021, 0.00006)
        and near(o100.gap_mean["rigid_asym"], -0.0073, 0.00006)
        and near(o100.gap_se["rigid_asym"], 0.0009, 0.00006))
    vdp = op[op.system == "vanderpol"].set_index("N")
    chk("Van der Pol gap 0.0002 +/- 0.0002 at N=6400",
        near(vdp.gap_mean[6400], 0.0002, 0.00006) and near(vdp.gap_se[6400], 0.0002, 0.00006))

    # ------------------------------------------------------------- Fig. 1
    f0 = json.loads((ROOT / "figures" / "fig1_example_numbers.json").read_text())
    chk("Fig. 1: Hopf rotation certified, Van der Pol nothing, every generator excluded below 0.137",
        f0["hopf"]["n_cert"] == 1 and f0["vanderpol"]["n_cert"] == 0
        and 0.137 <= f0["vanderpol"]["refute"] < 0.1375, f0["vanderpol"]["refute"])
    fsrc = (ROOT / "figures" / "make_figures.py").read_text()
    chk("Fig. 1: N = 3200, noise level 0.03, six affine generators",
        '("hopf", 3, 3200, 0.03,' in fsrc and '("vanderpol", 3, 3200, 0.03,' in fsrc)

    # ------------------------------------------------------------- exp4
    d4 = load("exp4_resolution")
    e4 = pd.DataFrame(d4["rows"])
    z = e4[e4.eps == 0]
    chk("resolution: two decades of N, three noise levels 0.03/0.1/0.3, degree 3 (Q = 20), 200 reps",
        max(d4["Ns"]) / min(d4["Ns"]) >= 100 and d4["sigmas"] == [0.03, 0.1, 0.3]
        and set(e4.degF) == {3} and set(e4.Q) == {20} and d4["trials"] == 200)
    ratios = [float(np.median(g.upper_med / g.lecam)) for _, g in z.groupby("sigma_rel")]
    chk("Le Cam ratio median 8.0 at each noise level (factor of about 8)",
        all(near(x, 8.0, 0.05) for x in ratios), [round(x, 2) for x in ratios])
    slopes = [np.polyfit(np.log(g.N), np.log(g.upper_med), 1)[0] for _, g in z.groupby("sigma_rel")]
    r3all = z.upper_med / z.lecam
    chk("Discussion: degree-3 factor nearly constant at about 8 across N and sigma",
        r3all.min() > 7.3 and r3all.max() < 8.3, f"{r3all.min():.2f}-{r3all.max():.2f}")
    chk("log-log slopes between -0.52 and -0.49",
        min(slopes) >= -0.525 and max(slopes) <= -0.485, [round(x, 3) for x in slopes])
    q = pd.DataFrame(d4["qsweep"])
    chk("Q-sweep: degrees 3, 5, 7 with Q = 20, 42, 72 at noise 0.1",
        q.groupby("degF").Q.first().to_dict() == {3: 20, 5: 42, 7: 72} and set(q.sigma_rel) == {0.1})
    qm = q[(q.N >= 800) & (q.N <= 12800)]
    fin = (np.isfinite(qm.upper_pointwise_med) & (qm.upper_pointwise_med < 1) & (qm.upper_med < 1)).all()
    coarse = q[q.N < 800]
    chk("N = 800 to 12,800 is where every bound is finite and below 1",
        fin and not (np.isfinite(coarse.upper_pointwise_med) & (coarse.upper_pointwise_med < 1)
                     & (coarse.upper_med < 1)).groupby(coarse.N).all().all())
    qr = {d: float(np.median(g.upper_med / g.lecam)) for d, g in qm.groupby("degF")}
    qp = {d: float(np.median(g.upper_pointwise_med / g.lecam)) for d, g in qm.groupby("degF")}
    chk("Q-sweep median factors 8.0, 10.9, 13.2 (N = 800..12,800)",
        near(qr[3], 8.0, 0.05) and near(qr[5], 10.9, 0.05) and near(qr[7], 13.2, 0.05),
        {k: round(v, 2) for k, v in qr.items()})
    chk("pointwise median factors 8.0, 10.7, 12.9 (N = 800..12,800)",
        near(qp[3], 8.0, 0.05) and near(qp[5], 10.7, 0.05) and near(qp[7], 12.9, 0.05),
        {k: round(v, 2) for k, v in qp.items()})
    infp = q[~np.isfinite(q.upper_pointwise_med)]
    chk("pointwise bound infinite exactly for degree 7 at N <= 200",
        set(infp.degF) == {7} and set(infp.N) == {100, 200})
    chk("pointwise slightly narrower than U from N = 3200 on, wider below",
        (q[q.N >= 3200].upper_pointwise_med < q[q.N >= 3200].upper_med).all()
        and (q[q.N < 3200].upper_pointwise_med > q[q.N < 3200].upper_med).all())
    q12 = q[q.N == 12800].set_index("degF")
    chk("at N = 12,800: pointwise 7.7, 10.1, 11.6 against U 7.9, 11.0, 13.2",
        all(near(q12.upper_pointwise_med[d] / q12.lecam[d], v, 0.05) for d, v in ((3, 7.7), (5, 10.1), (7, 11.6)))
        and all(near(q12.upper_med[d] / q12.lecam[d], v, 0.05) for d, v in ((3, 7.9), (5, 11.0), (7, 13.2))))
    gain = 1 - e4.upper_med / e4.upper_separate_med
    chk("168 resolution settings", len(e4) == 168)
    chk("sharp narrower by median 2.5%, at most a third",
        near(gain.median(), 0.025, 0.0015) and gain.max() < 0.34 and gain.min() > 0)
    big = e4.loc[gain.idxmax()]
    chk("largest gain at N=100, noise 0.3", big.N == 100 and big.sigma_rel == 0.3)
    rr = e4[(e4.N == 1600) & (e4.sigma_rel == 0.1)].set_index("eps")
    lo, hi = cp(int(rr.refute_count[0.0]), d4["trials"])
    chk("refutation 49.5% at defect 0.013, 100% at 0.032, 1.0% (2 of 200, [0.1%, 3.6%]) when symmetric",
        near(rr.refute_rate[0.02], 0.495, 0.001) and rr.refute_rate[0.05] == 1.0
        and rr.refute_count[0.0] == 2 and near(lo, 0.001, 0.0005) and near(hi, 0.036, 0.0005)
        and near(rr.true_defect[0.02], 0.013, 0.0005) and near(rr.true_defect[0.05], 0.032, 0.0005),
        f"[{lo:.4f}, {hi:.4f}]")

    # ------------------------------------------------------------- exp6
    d6 = load("exp6_dimension")
    e6 = pd.DataFrame(d6["rows"])
    g6 = e6.groupby(["system", "degF", "gens"]).agg(
        us=("upper_sim", "mean"), up=("upper_split", "mean"),
        ds=("detect_sim", "mean"), dp=("detect_split", "mean"),
        d=("d_true", "first"), Q=("Q", "first"), P=("P", "first"))
    rat = g6.up / g6.us
    chk("Fig. 3b,c: nine configurations, noise 0.05, 300 reps at N = 200..6400",
        len(g6) == 9 and set(e6.sigma_rel) == {0.05} and d6["trials"] == 300
        and d6["Ns"] == [200, 400, 800, 1600, 3200, 6400])
    chk("six configurations have a true algebra of dimension 2 or 3",
        (g6.d >= 2).sum() == 6 and set(g6.d[g6.d >= 2]) == {2, 3})
    chk("simultaneous narrower and certifies the direction at least as often in all nine",
        (g6.us < g6.up).all() and (g6.ds >= g6.dp).all())
    chk("width ratio 1.6 to 2.6", near(rat.min(), 1.6, 0.05) and near(rat.max(), 2.6, 0.05),
        f"{rat.min():.2f}-{rat.max():.2f}")
    h = g6.loc[("hopf", 3)]
    chk("affine (P = 6) vs quadratic (P = 12) generators 0.0454 vs 0.0447"
        " (median over replications, averaged over N = 200 to 6400)",
        h.P["affine"] == 6 and h.P["quadratic"] == 12
        and "upper_sim=float(np.median(us))" in (ROOT / "experiments" / "exp6_dimension.py").read_text()
        and near(h.us["affine"], 0.0454, 0.00006) and near(h.us["quadratic"], 0.0447, 0.00006))
    h5 = g6.xs(("hopf", "affine"), level=("system", "gens"))
    chk("degree 3 to 5 (Q 20 to 42) widens 0.045 to 0.122",
        h5.Q[3] == 20 and h5.Q[5] == 42 and near(h5.us[3], 0.045, 0.0006) and near(h5.us[5], 0.122, 0.0006))

    # ------------------------------------------------------------- exp14
    d14 = load("exp14_selection")
    e14 = pd.DataFrame(d14["rows"])
    chk("selection: k in {5, 7}, noise 0.1, N = 3200 in Fig. 5, 300 reps, 56 settings",
        d14["degrees"] == [5, 7] and d14["sigma_rel"] == 0.1 and 3200 in d14["Ns"]
        and d14["trials"] == 300 and len(e14) == 56)
    k5 = e14[(e14.k == 5) & (e14.defect_multiple != 1.0)]
    chk("degree 5: gate never certifies falsely when the rotation defect differs from delta",
        k5.fc_gate.max() == 0)
    g5 = e14[(e14.k == 5) & (e14.N == 3200)].set_index("defect_multiple")
    l6, h6 = cp(int(g5.count_gate[1.0]), 300)
    chk("degree 5, defect exactly delta: gate errs in 6% at N=3200 (95% interval 3% to 9%)",
        near(g5.fc_gate[1.0], 0.06, 0.006) and near(l6, 0.03, 0.005) and near(h6, 0.09, 0.005),
        f"[{l6:.3f},{h6:.3f}]")
    i5 = e14[(e14.k == 5) & (e14.N == 3200)].invisible_defect_05.iloc[0]
    i7 = e14[(e14.k == 7) & (e14.N == 3200)].invisible_defect_05.iloc[0]
    chk("invisible defects 0.039 and 0.087 at N=3200", near(i5, 0.039, 0.0006) and near(i7, 0.087, 0.0006))
    chk("0.087 is about 1.7 times the tolerance (Results, Discussion, guarantee section)",
        near(i7 / 0.05, 1.7, 0.05))
    g7 = e14[(e14.k == 7) & (e14.N == 3200)].set_index("defect_multiple")
    l125, h125 = cp(int(g7.count_gate[1.25]), 300)
    l15, h15 = cp(int(g7.count_gate[1.5]), 300)
    l10, h10 = cp(int(g7.count_gate[1.0]), 300)
    chk("degree 7 gate: 46% [40, 52] at 1.25 delta, 27% [22, 33] at 1.5 delta, 39% [33, 45] at delta",
        near(g7.fc_gate[1.25], 0.46, 0.006) and near(l125, 0.40, 0.006) and near(h125, 0.52, 0.006)
        and near(g7.fc_gate[1.5], 0.27, 0.006) and near(l15, 0.22, 0.006) and near(h15, 0.33, 0.006)
        and near(g7.fc_gate[1.0], 0.39, 0.006) and near(l10, 0.33, 0.005) and near(h10, 0.45, 0.005),
        f"[{l125:.3f},{h125:.3f}] [{l15:.3f},{h15:.3f}] [{l10:.3f},{h10:.3f}]")
    sp7 = e14[(e14.k == 7) & (e14.defect_multiple > 1.0)]
    j7 = sp7.fc_split.idxmax()
    ls, hs = cp(int(round(sp7.fc_split[j7] * 300)), 300)
    chk("split up to 11% [8, 15] above delta (degree 7, N=12800, 1.25 delta)",
        near(sp7.fc_split.max(), 0.11, 0.006)
        and sp7.loc[j7, "N"] == 12800 and sp7.loc[j7, "defect_multiple"] == 1.25
        and near(ls, 0.08, 0.005) and near(hs, 0.15, 0.005), f"[{ls:.3f},{hs:.3f}]")
    eq7 = e14[(e14.k == 7) & (e14.defect_multiple == 1.0)].set_index("N")
    le, he = cp(int(round(eq7.fc_split[12800] * 300)), 300)
    chk("split 19% [15, 24] at exactly delta (degree 7, N=12800)", near(eq7.fc_split[12800], 0.19, 0.006)
        and eq7.fc_split.idxmax() == 12800 and near(le, 0.15, 0.005) and near(he, 0.24, 0.005),
        f"[{le:.3f},{he:.3f}]")
    chk("valid procedures never certify falsely in 56 settings",
        e14.fc_largest.max() == 0 and e14.fc_simultaneous.max() == 0)
    z14 = e14[e14.defect_multiple == 0.0]
    chk("price: <=0.02 below 51200 (deg 5), nothing (deg 7); gate from 3200",
        z14[(z14.k == 5) & (z14.N < 51200)][["dim_largest", "dim_simultaneous"]].max().max() <= 0.02
        and z14[z14.k == 7][["dim_largest", "dim_simultaneous"]].max().max() == 0
        and z14[z14.N == 3200].dim_gate.min() > 0.8 and z14[z14.N == 800].dim_gate.max() == 0)
    from hsd.selection import lof_power
    th = []
    for _, r_ in e14.iterrows():
        k_, N_ = int(r_.k), int(r_.N)
        mk = (k_ + 1) * (k_ + 2) // 2
        lam = N_ * r_.eps ** 2 / (2 * k_ + 1) / r_.sigma ** 2
        th.append(lof_power(lam, 2 * (mk - 10), 2 * (N_ - mk)))
    th = np.array(th)
    dif = np.abs(e14.lof_reject.values - th)
    j = int(np.argmax(dif))
    se = np.sqrt(th[j] * (1 - th[j]) / 300)
    chk("lack-of-fit power agrees within 0.049, less than two Monte Carlo SE",
        near(dif.max(), 0.049, 0.0006) and dif[j] < 2 * se, f"{dif[j]:.4f} vs 2 SE {2 * se:.4f}")
    # Invisible defect with the denominator evaluated exactly at F +/- h* (Methods).
    from common import rms_field
    from hsd import DefectProblem, get_system
    from hsd.polynomials import moment_matrix
    from hsd.selection import _embed, lof_noncentrality_for_power
    sig14 = 0.1 * rms_field(get_system("hopf"), np.random.default_rng(0), 1.0)
    p3 = DefectProblem.build(2, 3, "affine", "box", scale=1.0)
    b3 = get_system("hopf").beta(p3)
    rot = np.zeros(p3.P)
    rot[1], rot[2] = 1.0, -1.0
    rot /= np.linalg.norm(rot)
    exact = {}
    for k_ in (5, 7):
        pk = DefectProblem.build(2, k_, "affine", "box", scale=1.0)
        G = np.kron(np.eye(2), moment_matrix(pk.alg, pk.measure, pk.scale, sub=pk.beta_idx))
        E = _embed(p3, pk)
        Pp = np.eye(pk.Q) - E @ np.linalg.solve(E.T @ G @ E, E.T @ G)
        w, U = np.linalg.eigh((G + G.T) / 2)
        Gh = (U * np.sqrt(np.clip(w, 0, None))) @ U.T
        Gih = (U / np.sqrt(np.clip(w, 1e-300, None))) @ U.T
        uu, ss, _ = np.linalg.svd(Gh @ (Pp @ Gih))
        Hb = Gih @ uu[:, ss > 1e-10 * ss.max()]
        v = np.linalg.svd(pk.L_theta(rot) @ Hb)[2][0]
        ms, mb = len(p3.beta_idx), len(pk.beta_idx)
        lam = lof_noncentrality_for_power(0.5, 2 * (mb - ms), 2 * (3200 - mb), 0.05)
        hst = Hb @ v * sig14 * np.sqrt(lam / 3200)
        exact[k_] = sorted(pk.rho(rot, E @ b3 + s * hst) for s in (1.0, -1.0))
    chk("Methods: exact denominators give 0.039-0.040 (degree 5) and 0.085-0.089 (degree 7)",
        near(exact[5][0], 0.039, 0.0005) and near(exact[5][1], 0.040, 0.0005)
        and near(exact[7][0], 0.085, 0.0005) and near(exact[7][1], 0.089, 0.0005),
        {k: [round(x, 4) for x in v_] for k, v_ in exact.items()})
    chk("Table S5 caption: smallest true defect 0.048 to 0.049 where the rotation defect is delta",
        near(e14[e14.defect_multiple == 1.0].true_min_defect.min(), 0.048, 0.0005)
        and near(e14[e14.defect_multiple == 1.0].true_min_defect.max(), 0.049, 0.0005))

    # ------------------------------------------------------------- exp8
    d8 = load("exp8_tightness")
    cov8 = pd.DataFrame(d8["coverage"])
    chk("Fig. S5a: three systems, N = 400, noise 0.03 and 0.1, 80 replications",
        cov8.system.nunique() == 3 and set(cov8.N) == {400} and set(cov8.sigma_rel) == {0.03, 0.1}
        and d8["trials"] == 80)
    chk("nominal levels 0.50 to 0.99", max(d8["alphas"]) == 0.5 and min(d8["alphas"]) == 0.01)
    chk("certificate interval covers at every nominal level",
        all(v == 1.0 for _, r_ in cov8.iterrows() for v in r_["coverage"].values()))
    reg95 = [r_["region_coverage"]["0.05"] for _, r_ in cov8.iterrows()]
    chk("region 0.94-0.96 in every setting at nominal 0.95",
        near(min(reg95), 0.94, 0.0026) and near(max(reg95), 0.96, 0.0026))
    rc8 = pd.DataFrame(list(cov8.region_coverage))
    nom8 = 1 - rc8.columns.astype(float)
    pooled_dev = (rc8.mean().values - nom8).__abs__().max()
    scatter = (rc8 - nom8.values).abs().max().max()
    chk("pooled over the six settings, region within 0.02 of nominal at every level 0.50 to 0.99"
        " (36 setting-level combinations, 80 reps)",
        len(rc8) == 6 and rc8.size == 36 and pooled_dev < 0.02 and min(nom8) == 0.5 and max(nom8) == 0.99,
        f"{pooled_dev:.4f}")
    chk("single settings scatter by up to 0.15 (0.35 to 0.59 at nominal 0.50); standard error up to 0.056",
        near(scatter, 0.15, 0.0026) and near(rc8["0.5"].min(), 0.35, 0.0026)
        and near(rc8["0.5"].max(), 0.59, 0.0026) and near(np.sqrt(0.25 / d8["trials"]), 0.056, 0.0005),
        f"{scatter:.4f}")
    w8 = pd.DataFrame(d8["width"])
    w8["low"] = 1 - w8.upper_boot / w8.upper_sharp
    chk("Fig. S5b: noise 0.1, 200 reps, true defects 0, 1.02 delta, 2 delta",
        d8["sigma_rel"] == 0.1 and d8["width_trials"] == 200
        and sorted(w8.true_defect.round(4).unique()) == [0.0, 0.051, 0.1])
    a05, a20 = w8[w8.alpha == 0.05], w8[w8.alpha == 0.2]
    chk("bootstrap upper bound 5% to 42% lower at alpha 0.05, up to 47% at alpha 0.2",
        near(a05.low.min(), 0.05, 0.005) and near(a05.low.max(), 0.42, 0.005)
        and near(a20.low.max(), 0.47, 0.005),
        f"{a05.low.min():.3f}-{a05.low.max():.3f}, {a20.low.max():.3f}")
    chk("gap largest at a true symmetry and shrinking with N when the defect is not zero",
        w8.loc[w8.low.idxmax(), "true_defect"] == 0.0
        and all(np.all(np.diff(g.sort_values("N").low.values) < 0)
                for (td, _), g in w8[w8.true_defect > 0].groupby(["true_defect", "alpha"])))
    b8 = pd.DataFrame(d8["boundary"])
    b102 = b8[b8.true_defect.round(3) == 0.051]
    mx = b102.groupby("alpha").fc_fix_boot.max()
    chk("boundary (1.02 delta, N 800..51200, 300 reps): bootstrap <= 15%, 4.3%, 0.7%; SymCert never",
        near(mx[0.5], 0.15, 0.002) and near(mx[0.2], 0.043, 0.001) and near(mx[0.05], 0.007, 0.001)
        and all(mx[a] <= a for a in mx.index)
        and b8.fc_fix_sharp.max() == 0 and b102.N.min() == 800 and b102.N.max() == 51200
        and d8["boundary_trials"] == 300)
    cnt = b102.groupby("alpha").count_fix_boot.max()
    bci = {a: cp(int(cnt[a]), 300) for a in (0.5, 0.2, 0.05)}
    chk("bootstrap boundary intervals: 15% [11, 20], 4.3% [2.3, 7.3], 0.7% [0.1, 2.4]",
        near(bci[0.5][0], 0.11, 0.005) and near(bci[0.5][1], 0.20, 0.005)
        and near(bci[0.2][0], 0.023, 0.0005) and near(bci[0.2][1], 0.073, 0.0005)
        and near(bci[0.05][0], 0.001, 0.0005) and near(bci[0.05][1], 0.024, 0.0005),
        {a: (round(x, 4), round(y, 4)) for a, (x, y) in bci.items()})
    s8 = pd.DataFrame(d8["small"])
    s02 = s8[(s8.true_defect.round(3) == 0.02) & (s8.N == 3200)].set_index("alpha")
    l36, h36 = cp(int(s02.count_boot[0.05]), 200)
    chk("bootstrap covers <=1/200 at either level (nominal 90% and 60%) at defect <= 0.01;"
        " at 0.02 (N=3200) 36% [29, 43] against 90% and 8.5% [5.0, 13.3] against 60%",
        s8[s8.true_defect <= 0.0101].count_boot.max() <= 1 and (s8.trials == 200).all()
        and set(s8.alpha) == {0.05, 0.2}
        and near(s02.coverage_boot[0.05], 0.36, 0.0026) and near(l36, 0.29, 0.005) and near(h36, 0.43, 0.005)
        and near(s02.coverage_boot[0.2], 0.085, 0.0005)
        and near(cp(int(s02.count_boot[0.2]), 200)[0], 0.050, 0.0005)
        and near(cp(int(s02.count_boot[0.2]), 200)[1], 0.133, 0.0005)
        and (s8[s8.alpha == 0.05].nominal_boot == 0.9).all() and (s8[s8.alpha == 0.2].nominal_boot == 0.6).all(),
        f"[{l36:.3f},{h36:.3f}]")
    chk("certificate coverage never below 0.93 (nominal 0.8) or 0.985 (nominal 0.95)",
        near(s8[s8.alpha == 0.2].coverage_cert.min(), 0.93, 0.0026)
        and near(s8[s8.alpha == 0.05].coverage_cert.min(), 0.985, 0.0026))
    se8 = pd.DataFrame(d8["selected"])
    import exp8_tightness as x8
    ssrc = inspect.getsource(x8.selected_cell)
    chk("selected generator: Van der Pol, Duffing, Lotka-Volterra, Sel'kov, all on [-1,1]^2,"
        " N 50-400, noise 0.2 and 0.5, 300 reps (Results; Methods)",
        set(se8.system) == {"vanderpol", "duffing", "lotka_volterra", "selkov"}
        and se8.N.min() == 50 and se8.N.max() == 400
        and set(se8.sigma_rel) == {0.2, 0.5} and d8["selected_trials"] == 300
        and '"box", scale=1.0)' in ssrc and 'sampler="box", scale=1.0)' in ssrc)
    stm = se8.groupby("system").true_min_defect.first()
    chk("selected study: smallest true defects 0.25 to 0.64",
        near(stm.min(), 0.25, 0.005) and near(stm.max(), 0.64, 0.005), stm.round(3).to_dict())
    se8["cp_hi"] = [cp(int(k), int(t))[1] for k, t in zip(se8.count_boot, se8.trials)]
    allbelow = se8[se8.cp_hi < se8.nominal]
    chk("whole 95% interval below nominal in 5 settings at each level, nine of these ten on Duffing",
        (allbelow.nominal == 0.8).sum() == 5 and (allbelow.nominal == 0.95).sum() == 5
        and (allbelow.system == "duffing").sum() == 9,
        allbelow.groupby("nominal").system.apply(list).to_dict())
    n80 = int((se8[se8.nominal == 0.8].coverage_upper_boot < 0.8).sum())
    n95 = int((se8[se8.nominal == 0.95].coverage_upper_boot < 0.95).sum())
    w80 = se8[se8.nominal == 0.8].sort_values("coverage_upper_boot").iloc[0]
    w95 = se8[se8.nominal == 0.95].sort_values("coverage_upper_boot").iloc[0]
    chk("bootstrap below nominal in 9 of 32 (0.80) and 8 of 32 (0.95)",
        n80 == 9 and n95 == 8 and (se8.nominal == 0.8).sum() == 32, f"{n80}, {n95}")
    c80, c95 = cp(int(w80.count_boot), 300), cp(int(w95.count_boot), 300)
    chk("worst 0.59 [0.54, 0.65] and 0.90 [0.86, 0.93], Duffing, N=50, noise 0.5",
        near(w80.coverage_upper_boot, 0.59, 0.006) and near(w95.coverage_upper_boot, 0.90, 0.006)
        and near(c80[0], 0.54, 0.005) and near(c80[1], 0.65, 0.005)
        and near(c95[0], 0.86, 0.005) and near(c95[1], 0.93, 0.005)
        and all(w.system == "duffing" and w.N == 50 and w.sigma_rel == 0.5 for w in (w80, w95)),
        f"[{c80[0]:.3f},{c80[1]:.3f}] [{c95[0]:.3f},{c95[1]:.3f}]")
    chk("selection biases the plug-in defect down by up to 0.15 on average",
        near(se8.mean_true_minus_plug.max(), 0.15, 0.006) and se8.mean_true_minus_plug.min() > 0)
    chk("certificate upper bound contained the true defect in every replication; defects far above delta",
        (se8.coverage_upper_sharp == 1.0).all() and se8.true_min_defect.min() > 4 * 0.05)

    # ------------------------------------------------------------- exp5
    d5 = load("exp5_robustness")
    a5 = pd.DataFrame(d5["design"])
    t5 = pd.DataFrame(d5["trajectory"])
    ols = a5[a5["cov"] == "ols"]
    hc0 = a5[a5["cov"] == "hc0"]
    chk("OLS region at least 0.938 under t and Laplace",
        near(ols[ols.noise.isin(["t", "laplace"])].region_coverage.min(), 0.938, 0.0006))
    chk("OLS region 0.82 under heteroskedasticity",
        near(ols[ols.hetero == 1.0].region_coverage.min(), 0.82, 0.006))
    chk("White 0.77 to 0.88 at N=400, 0.93 to 0.96 at N=3200 (Discussion: down to 0.77)",
        near(hc0[hc0.N == 400].region_coverage.min(), 0.77, 0.006)
        and near(hc0[hc0.N == 400].region_coverage.max(), 0.88, 0.006)
        and near(hc0[hc0.N == 3200].region_coverage.min(), 0.93, 0.006)
        and near(hc0[hc0.N == 3200].region_coverage.max(), 0.96, 0.0026))
    chk("certificate always covers, never false, under error variations",
        a5.coverage.min() == 1.0 and a5.fc.max() == 0)
    cl = t5[t5["mode"] == "clean_states"]
    chk("trajectories (exact states): White 0.90-0.98", near(cl.reg_hc0.min(), 0.90, 0.006)
        and near(cl.reg_hc0.max(), 0.98, 0.006))
    chk("trajectories (exact states): cluster-robust 0.76-0.91 with 48 trajectories",
        d5["n_segments"] == 48 and near(cl.reg_cluster.min(), 0.76, 0.006)
        and near(cl.reg_cluster.max(), 0.91, 0.006))
    # (state dimension, class degree) of each trajectory system; Q = n C(n+d, d).
    nd = {"hopf": (2, 3), "vanderpol": (2, 3), "rigid_asym": (3, 2)}
    qtr = {s: nd[s][0] * comb(sum(nd[s]), nd[s][1]) for s in t5.system.unique()}
    chk("48 trajectories are 1.6 to 2.4 times the coefficients (Q = 30 or 20)",
        sorted(set(qtr.values())) == [20, 30] and near(48 / 30, 1.6, 0.05) and near(48 / 20, 2.4, 0.05),
        qtr)
    chk("state noise 0.05% to 0.5% of the box half-width",
        min(d5["state_noise"]) == 0.0005 and max(d5["state_noise"]) == 0.005)
    chk("noisy states: White overcovers (1.00), certificate never fails",
        t5[t5["mode"] != "clean_states"].reg_hc0.min() == 1.0
        and t5[["cert_hc0", "cert_cluster"]].min().min() == 1.0
        and t5[["fc_hc0", "fc_cluster"]].max().max() == 0)
    chk("trajectories of 100 steps of 0.01", round(d5["t_segment"] / d5["dt"]) == 100 and d5["dt"] == 0.01)

    # ------------------------------------------------------------- exp15
    d15 = load("exp15_dependence")
    e15 = pd.DataFrame(d15["rows"])
    chk("Fig. S1: Hopf and asymmetric rigid body, 12/48/192 trajectories, noise 0.05, 200 reps, 100 steps",
        set(e15.system) == {"hopf", "rigid_asym"} and d15["groups"] == [12, 48, 192]
        and d15["sigma_rel"] == 0.05 and d15["trials"] == 200
        and round(d15["t_segment"] / d15["dt"]) == 100 and d15["dt"] == 0.01)
    chk("OLS/White regions <= 7% once phi >= 0.5",
        near(e15[e15.phi >= 0.5][["reg_ols", "reg_hc0"]].max().max(), 0.065, 0.001))
    h15 = e15[e15.phi >= 0.9]
    chk("OLS/White certificates cover 1% to 27% for phi >= 0.9",
        near(h15[["cert_ols", "cert_hc0"]].min().min(), 0.01, 0.0006)
        and near(h15[["cert_ols", "cert_hc0"]].max().max(), 0.27, 0.006))
    c48, c192 = e15[e15.G == 48], e15[e15.G == 192]
    chk("cluster 73-92% (48), 91-95% (192), certificate always covers",
        near(c48.reg_cluster.min(), 0.73, 0.006) and near(c48.reg_cluster.max(), 0.92, 0.006)
        and near(c192.reg_cluster.min(), 0.91, 0.006) and near(c192.reg_cluster.max(), 0.95, 0.006)
        and e15[e15.G >= 48].cert_cluster.min() == 1.0)
    c12 = e15[e15.G == 12]
    chk("12 trajectories, fewer than the coefficients: cluster region unidentified, certifies nothing",
        (c12.G < c12.Q).all() and (c12.identified_cluster == 0).all() and c12.dim_cluster.max() == 0)
    nw = e15[e15.phi >= 0.5]
    rng_nw = {G: (nw[nw.G == G].reg_hac.min(), nw[nw.G == G].reg_hac.max()) for G in (12, 48, 192)}
    chk("Newey-West for phi >= 0.5: 0.14-0.80 (192), 0.01-0.81 (48), 0-0.46 (12)",
        near(rng_nw[192][0], 0.14, 0.006) and near(rng_nw[192][1], 0.80, 0.006)
        and near(rng_nw[48][0], 0.01, 0.006) and near(rng_nw[48][1], 0.81, 0.006)
        and near(rng_nw[12][0], 0.0, 0.006) and near(rng_nw[12][1], 0.46, 0.006),
        {k: (round(a, 3), round(b, 3)) for k, (a, b) in rng_nw.items()})
    p5 = e15[(e15.phi == 0.5) & (e15.G == 192)]
    chk("Newey-West at phi = 0.5 with 192 trajectories: 0.75 to 0.80",
        near(p5.reg_hac.min(), 0.75, 0.006) and near(p5.reg_hac.max(), 0.80, 0.006))
    chk("no false certification under dependence",
        e15[[c for c in e15.columns if c.startswith("fc_")]].max().max() == 0)
    lowc = e15[(e15.G == 12) & (e15.phi == 0.98)].set_index("system").cert_hac
    rest = e15[~((e15.G == 12) & (e15.phi == 0.98))].cert_hac
    chk("Newey-West certificate >= 98% except 80% (Hopf) and 91% (rigid body) at 12 trajectories, phi = 0.98"
        " (Results; Discussion: as few as 80%)",
        near(rest.min(), 0.98, 0.0026) and near(lowc["hopf"], 0.80, 0.006)
        and near(lowc["rigid_asym"], 0.91, 0.006) and near(e15.cert_hac.min(), 0.80, 0.006),
        f"{rest.min():.3f}, {lowc.round(3).to_dict()}")
    import hsd.trajectories as htraj
    chk("dependence study: trajectories of at most 100 points (Discussion, drifter paragraph)",
        round(d15["t_segment"] / d15["dt"]) == 100
        and "out = np.empty((n_steps,) + X.shape)" in inspect.getsource(htraj.integrate))

    # ------------------------------------------------------------- exp18
    du = e18[e18.system == "hard_duffing"].set_index(["state_noise", "N"])
    import exp18_eiv as x18
    hd = x18.hard_duffing().terms
    chk("hardening Duffing x' = y, y' = -x - 0.3x^3 - 0.2y",
        float(hd[(0, (0, 1))]) == 1 and float(hd[(1, (1, 0))]) == -1
        and float(hd[(1, (3, 0))]) == -0.3 and float(hd[(1, (0, 1))]) == -0.2)
    d18 = load("exp18_eiv")
    chk("Fig. S3: right-hand-side noise 0.02, 200 reps, three sample sizes, 10^6 reference states",
        d18["sigma_y_rel"] == 0.02 and d18["trials"] == 200 and len(d18["Ns"]) == 3
        and "size=(1_000_000, 2)" in inspect.getsource(x18.pseudo_true_min_defect))
    chk("Duffing: true 0.102, pseudo-true 0.038 at state noise 0.08",
        near(du.true_min_defect.iloc[0], 0.102, 0.0006)
        and near(du.loc[(0.08, 3200)].pseudo_true_min_defect, 0.038, 0.0006))
    chk("false certification in every replication at N = 25,600 and 204,800, none at 3200",
        du.loc[(0.08, 25600)].fc == 1.0 and du.loc[(0.08, 204800)].fc == 1.0
        and du.loc[(0.08, 3200)].fc == 0 and du.loc[(0.08, 3200)].mean_dim == 0)
    chk("region never covers with state noise (Duffing)",
        du[du.index.get_level_values(0) > 0].region_coverage.max() == 0)
    chk("threshold false at every N at 0.08, identical at every sample size",
        all(du.loc[(0.08, n)].fc_threshold == 1.0 for n in d18["Ns"])
        and du.groupby(level=0).fc_threshold.nunique().max() == 1)
    chk("0.04 and 0.12 leave the pseudo-true defect above tolerance, no false certification",
        du.loc[(0.04, 3200)].pseudo_true_min_defect > 0.05 and du.loc[(0.12, 3200)].pseudo_true_min_defect > 0.05
        and du[du.index.get_level_values(0).isin([0.04, 0.12])].fc.max() == 0)
    hp = e18[e18.system == "hopf"]
    chk("Hopf: detection lost from state noise 0.08 (not at 0.04); region misses whenever sigma_x > 0",
        hp[hp.state_noise >= 0.08].detect.max() == 0 and hp[hp.state_noise == 0.04].detect.max() > 0
        and hp[hp.state_noise > 0].region_coverage.max() == 0)

    # ------------------------------------------------------------- exp9
    d9 = load("exp9_modelerror")
    e9 = pd.DataFrame(d9["rows"]).set_index(["system", "degree"])
    chk("model error: N = 40,000, noise 0.02, one data set per system and degree",
        d9["N"] == 40000 and d9["sigma_rel"] == 0.02 and len(e9) == 2 * len(d9["degrees"]))
    chk("L2 form against a class four degrees larger",
        "DefectProblem.build(2, deg + 4," in (ROOT / "experiments" / "exp9_modelerror.py").read_text())
    r5 = e9.loc[("rot_nonpoly", 5)]
    chk("degree 5: 0.050 ignored, 0.65 L2 oracle, 2.65 sup, 0.68 data-driven",
        near(r5.upper_no_model_error, 0.050, 0.0006) and near(r5.upper_l2, 0.65, 0.006)
        and near(r5.upper_sup_norm, 2.65, 0.006) and near(r5.upper_l2_datadriven, 0.68, 0.006))
    r9 = e9.loc[("rot_nonpoly", 9)]
    chk("degree 9: rel error 0.5%, data-driven 0.27",
        near(r9.rel_l2_error, 0.005, 0.0006) and near(r9.upper_l2_datadriven, 0.27, 0.006))
    rot9 = e9.loc["rot_nonpoly"]
    chk("Discussion: narrowest bound accounting for model error is 0.27",
        near(rot9[["upper_l2", "upper_sup_norm", "upper_l2_datadriven"]].min().min(), 0.27, 0.006))
    chk("Fig. S2a: true defect of the selected generator small but not zero",
        (rot9.true_defect > 0).all() and rot9.true_defect.max() < 0.05)
    import exp9_modelerror as x9
    import hsd.nonpoly as hnp
    src9 = (ROOT / "experiments" / "exp9_modelerror.py").read_text()
    chk("Fig. S2 caption: true defect from 2e6 uniform points (SE below 2e-4), central differences with"
        " step 1e-6 for the radial flow, oracle sizes as maxima over 4e5 uniform points",
        "mc_defect(sm, prob, theta, rng, n_mc=2_000_000)" in src9
        and e9.true_defect_se.max() < 2e-4
        and sg_(hnp._rot_jac).parameters["h"].default == 1e-6
        and hnp.SMOOTH_SYSTEMS["rot_nonpoly"].jac is hnp._rot_jac
        and sg_(x9.measure_model_error).parameters["n_mc"].default == 400_000
        and "measure_model_error(prob, fit, sm, rng, scale)" in src9,
        f"max SE {e9.true_defect_se.max():.2e}")
    pe = e9.loc["pendulum"]
    chk("pendulum: true 0.204, bounds 0.23-0.39, rel error 0.07% at degree 5",
        near(pe.true_defect.iloc[0], 0.204, 0.0006)
        and near(pe[["upper_l2", "upper_sup_norm", "upper_l2_datadriven"]].min().min(), 0.23, 0.006)
        and near(pe[["upper_l2", "upper_sup_norm", "upper_l2_datadriven"]].max().max(), 0.39, 0.006)
        and near(pe.loc[5].rel_l2_error, 0.0007, 0.00006))

    # ------------------------------------------------------------- exp16
    d16 = load("exp16_dictionary")
    e16 = pd.DataFrame(d16["rows"])
    chk("two non-polynomial families, 45 settings, 300 reps, N 200/800/3200, noise 0.03/0.1/0.3",
        e16.family.nunique() == 2 and len(e16) == 45 and d16["trials"] == 300
        and d16["Ns"] == [200, 800, 3200] and d16["sigmas"] == [0.03, 0.1, 0.3])
    chk("45 dictionary settings, SymCert never false, threshold up to 6.3%",
        e16.fc_sym.max() == 0 and near(e16.fc_thr.max(), 0.063, 0.0006))
    r0 = e16[(e16.family == "radial") & (e16.eps == 0)]
    chk("exact rotation certified at noise 0.03, median bound 0.008 at N=3200",
        r0[r0.sigma_rel == 0.03].dim_sym.min() == 1.0
        and near(r0[(r0.sigma_rel == 0.03) & (r0.N == 3200)].upper_med.iloc[0], 0.008, 0.0006))
    r3 = e16[(e16.family == "radial") & (e16.eps == 0.03)]
    chk("defect 0.024 certified from N=800 at noise 0.03",
        near(r3.true_rot_defect.iloc[0], 0.024, 0.0006)
        and r3[(r3.sigma_rel == 0.03) & (r3.N >= 800)].dim_sym.min() == 1.0
        and r3[(r3.sigma_rel == 0.03) & (r3.N < 800)].dim_sym.max() < 1.0)
    r12 = e16[(e16.family == "radial") & (e16.eps == 0.12)]
    chk("defect 0.096: nothing certified", near(r12.true_rot_defect.iloc[0], 0.096, 0.0006)
        and r12.dim_sym.max() == 0)
    pdl = e16[e16.family == "pendulum"]
    u213 = pdl[(pdl.N == 3200) & (pdl.sigma_rel == 0.03)].upper_med.iloc[0]
    chk("pendulum refuted at N=3200 for noise <= 0.1; bound 0.213 within 5% of 0.204",
        pdl[(pdl.N == 3200) & (pdl.sigma_rel <= 0.1)].refute.min() == 1.0
        and pdl[(pdl.N == 3200) & (pdl.sigma_rel > 0.1)].refute.max() < 1.0
        and near(u213, 0.213, 0.0006) and near(pdl.true_min_defect.iloc[0], 0.204, 0.0006)
        and u213 / pdl.true_min_defect.iloc[0] - 1 < 0.05)
    import hsd.certify as hcert
    import hsd.dict_systems as hds
    import hsd.trs as htrs
    sg = inspect.signature
    chk("Methods: quadrature 40 Gauss-Legendre radial nodes, 24 angles; 32 x 32 for the pendulum",
        sg(hds.radial_problem).parameters["Mr"].default == 40
        and sg(hds.radial_problem).parameters["Mphi"].default == 24
        and sg(hds.pendulum_problem).parameters["M"].default == 32)
    tsrc = (ROOT / "tests" / "test_extensions.py").read_text()
    ref_t = tsrc[tsrc.index("def test_dictionary_quadrature_refinement"):tsrc.index("def _analytic_field")]
    mc_t = tsrc[tsrc.index("def test_dictionary_defects_match_monte_carlo"):tsrc.index("def test_closure_diagnostic")]
    chk("Methods: doubling both rules changes no defect by more than a relative 1e-10; Monte Carlo"
        " with 1e6 points agrees within three standard errors",
        "40 * refine, 24 * refine" in tsrc and "32 * refine" in tsrc
        and "rtol=1e-10" in ref_t and "rel=1e-10" in ref_t
        and "1_000_000" in mc_t and "<= 3 * se" in mc_t)

    # ------------------------------------------------------------- numerics (Methods)
    chk("Methods: relative tolerance 1e-7, safety margin 1e-9, zero threshold 1e-12",
        htrs.RATIO_RTOL == 1e-7 and htrs.REL_SAFETY == 1e-9 and htrs.ZERO_RTOL == 1e-12)
    chk("Methods: parametric bootstrap with 400 draws",
        sg(hcert.certify_direction_bootstrap).parameters["n_draw"].default == 400)

    # ------------------------------------------------------------- exp11
    d11 = load("exp11_scaling")
    e11 = pd.DataFrame(d11["rows"])
    chk("scaling: n = 2..6, N = 200/800/3200/12800, 20 cells, d_true = n, noise 0.05, 200 reps",
        sorted(e11.n.unique()) == [2, 3, 4, 5, 6] and sorted(e11.N.unique()) == [200, 800, 3200, 12800]
        and len(e11) == 20 and (e11.d_true == e11.n).all() and d11["sigma_rel"] == 0.05
        and d11["trials"] == 200)
    chk("scaling FCR 0 in all 20 cells", e11.fc.max() == 0)
    chk("P = n^2 from 4 to 36, Q = 6 at n=2 and 42 at n=6",
        (e11.P == e11.n ** 2).all() and e11.P.min() == 4 and e11.P.max() == 36
        and set(e11[e11.n == 2].Q) == {6} and set(e11[e11.n == 6].Q) == {42})
    w12800 = e11[e11.N == 12800].set_index("n").upper_med
    chk("width 0.0017 -> 0.0050, factor 2.9, somewhat more than the sqrt(Q) factor 2.6",
        near(w12800[2], 0.0017, 0.00006) and near(w12800[6], 0.0050, 0.00006)
        and near(w12800[6] / w12800[2], 2.9, 0.05) and near(np.sqrt(42 / 6), 2.6, 0.05))
    dd = e11.set_index(["n", "N"]).detect
    chk("full algebra from N=800 at n=4, N=3200 at n=6",
        dd[(4, 800)] == 1.0 and dd[(4, 200)] < 1.0 and dd[(6, 3200)] == 1.0 and dd[(6, 800)] < 1.0
        and dd[(4, 3200)] == 1.0 and dd[(6, 12800)] == 1.0)
    chk("0.03 to 0.11 CPU-s per certified subspace",
        near(e11.cpu_seconds_per_certificate.min(), 0.03, 0.002)
        and near(e11.cpu_seconds_per_certificate.max(), 0.11, 0.002))

    # ------------------------------------------------------------- exp17
    d17 = load("exp17_units_closure")
    u17 = pd.DataFrame(d17["units"])
    bh = u17[u17.system == "broken_hopf"]
    lv = u17[u17.system == "lotka_volterra"].set_index("unit_factor")
    chk("units c from 0.01 to 100", u17.unit_factor.min() == 0.01 and u17.unit_factor.max() == 100)
    chk("rotation defect 0.051-0.076 (0.064 in original units); nondimensional fixed",
        near(bh.defect_euclidean.min(), 0.051, 0.0006) and near(bh.defect_euclidean.max(), 0.076, 0.0006)
        and near(bh.defect_reference.iloc[0], 0.064, 0.0006)
        and near(bh[bh.unit_factor == 1].defect_euclidean.iloc[0], 0.064, 0.0006)
        and bh.defect_nondimensional.std() < 1e-5)
    chk("LV smallest Euclidean defect 0.27 -> 0.0035 at units a hundred times smaller; nondim fixed",
        near(lv.min_defect_euclidean[1.0], 0.27, 0.0015) and near(lv.min_defect_euclidean[0.01], 0.0035, 0.00006)
        and lv.min_defect_euclidean.max() == lv.min_defect_euclidean[1.0]
        and lv.min_defect_nondimensional.std() < 1e-5)
    us = pd.DataFrame(d17["units_sim"])
    lvs = us[(us.system == "lotka_volterra") & (us.unit_factor != 1.0)]
    chk("LV rescaled: Euclidean certifies falsely always, declared metric never",
        lvs[lvs.metric == "euclidean"].fc.min() == 1.0
        and us[us.metric == "nondimensional"].certify_rate.max() == 0)
    src17 = (ROOT / "experiments" / "exp17_units_closure.py").read_text()
    chk("units certification at N = 12800, noise 0.05, 300 reps (Table S1 caption)",
        "N=12800" in src17 and "0.05 * rms_field" in src17 and (us.trials == 300).all())
    c17 = pd.DataFrame(d17["closure"])
    so3 = c17[c17.system == "sphere_flow3"]
    small_noisy = (so3.N == so3.N.min()) & (so3.sigma_rel == so3.sigma_rel.max())
    chk("so(3): dimension 3 in every replication except smallest N with larger noise",
        (so3[~small_noisy].mean_dim == 3.0).all() and (so3[small_noisy].mean_dim < 3.0).all())
    chk("so(3): closure <= 0.4%, every commutator certified",
        so3.max_closure.max() <= 0.004 and so3.commutators_certified.dropna().min() == 1.0)
    chk("abelian algebras: commutators negligible",
        c17[c17.system != "sphere_flow3"].runs_with_commutators.max() == 0)
    chk("closure: 100 replications per row (Table S2 caption)", (c17.runs == 100).all())

    # ------------------------------------------------------------- exp19
    d19 = load("exp19_drifters")
    rows = {(r["degree"], r["cov"]): r for r in d19["rows"]}
    r1c = rows[(1, "cluster")]
    chk("31 drifters, 46,733 pairs", r1c["drifters"] == 31 and r1c["N"] == 46733)
    chk("fitted linear field 0.994 and -1.028",
        near(r1c["linear"][0][2], 0.994, 0.0006) and near(r1c["linear"][1][1], -1.028, 0.0006))
    chk("plug-in rotation defect 0.025", near(r1c["plug_rot"], 0.025, 0.0006))
    chk("velocity error 0.014, rms 0.109",
        near(d19["median_velocity_error"], 0.014, 0.0006) and near(d19["rms_velocity"], 0.109, 0.0006))
    chk("cluster-robust: U 0.074, L 0, not certified at 0.05 (linear class)",
        near(r1c["upper_rot"], 0.074, 0.0006) and r1c["lower_rot"] < 5e-4 and r1c["upper_rot"] > 0.05)
    wu = {w["window"]: w["upper_rot"] for w in d19["windows"]}
    chk("filter windows (49 h default): 0.067 to 0.074",
        49 in wu and near(min(wu.values()), 0.067, 0.0006) and near(max(wu.values()), 0.074, 0.0006))
    r1o, r1h = rows[(1, "ols")], rows[(1, "hac")]
    chk("OLS and Newey-West: U 0.040 and 0.045, L 0.010 and 0.005 (both certify, exclude exact)",
        near(r1o["upper_rot"], 0.040, 0.0006) and near(r1h["upper_rot"], 0.045, 0.0006)
        and near(r1o["lower_rot"], 0.010, 0.0006) and near(r1h["lower_rot"], 0.005, 0.0006))
    chk("quadratic and cubic: 0.11 and 0.21", near(rows[(2, "cluster")]["upper_rot"], 0.11, 0.005)
        and near(rows[(3, "cluster")]["upper_rot"], 0.21, 0.005))
    simc = [s for s in d19["simultaneous"] if s["cov"] == "cluster"][0]
    chk("simultaneous over degrees 1 to 3 (cluster-robust): 0.25", near(simc["upper_min"], 0.25, 0.005))
    chk("31 clusters: five times the 6 coefficients, 2.6 and 1.6 times 12 and 20",
        r1c["Q"] == 6 and rows[(2, "cluster")]["Q"] == 12 and rows[(3, "cluster")]["Q"] == 20
        and int(r1c["drifters"] / r1c["Q"]) == 5
        and round(r1c["drifters"] / rows[(2, "cluster")]["Q"] + 1e-9, 1) == 2.6
        and round(r1c["drifters"] / rows[(3, "cluster")]["Q"] + 1e-9, 1) == 1.6)
    chk("extrapolation factor 1.02 for the linear class", near(r1c["kappa"], 1.02, 0.006))
    import exp19_drifters as x19
    dpar = sg(x19.design).parameters
    chk("Methods: segments of at least 150 h, running mean of 49 h",
        dpar["window"].default == 49 and dpar["min_len"].default == 150)
    chk("Methods: 30-45 N, 180-140 W, 1 June to 30 September 2016-2019, gaps of at most 6 h",
        all(s in x19.ERDDAP for s in ("latitude%3E=30", "latitude%3C=45", "longitude%3E=-180",
                                      "longitude%3C=-140", "{y}-06-01T", "{y}-09-30T"))
        and x19.YEARS == [2016, 2017, 2018, 2019]
        and "df.gap <= 6 * 3600" in inspect.getsource(x19.fetch))
    if x19.DATA.exists():
        _, _, grp, _ = x19.design(x19.load_data())
        per = np.unique(grp, return_counts=True)[1]
        chk("median 1,486 pairs per drifter record; most records much longer than 100 points",
            len(per) == 31 and np.median(per) == 1486 and per.sum() == 46733 and (per > 300).mean() > 0.5,
            f"median {np.median(per)}, share > 300: {(per > 300).mean():.2f}")
    else:
        chk("median 1,486 pairs per drifter record (data file missing)", False, str(x19.DATA))
    lin_dims = {int(r["certified_dim"]) for r in d19["rows"] + d19["windows"] if r["degree"] == 1}
    lin_dims |= {int(y["certified_dim"]) for y in d19["years"]}
    chk("Table S8 caption: degree-1 rows certify 0, 1 or 2 of the trivial generators",
        lin_dims == {0, 1, 2}, sorted(lin_dims))
    yrs = {y["year"]: y for y in d19["years"]}
    chk("Table S8 caption: 2016/2017 unidentified (3, 4 drifters < 6); 2018 (10), 2019 (15): U 0.117, 0.109",
        not yrs[2016]["cluster_identified"] and not yrs[2017]["cluster_identified"]
        and yrs[2016]["drifters"] == 3 and yrs[2017]["drifters"] == 4
        and yrs[2018]["drifters"] == 10 and yrs[2019]["drifters"] == 15
        and near(yrs[2018]["upper_rot_cluster"], 0.117, 0.0006)
        and near(yrs[2019]["upper_rot_cluster"], 0.109, 0.0006))

    # ------------------------------------------------------------- exp7 (Supplementary Note 1)
    d7 = load("exp7_invariance")
    e7 = pd.DataFrame(d7["rows"])
    chk("SN1: degree 6, noise 0.01/0.05/0.2, N 1000 to 64,000, 300 reps, five targets",
        d7["degree"] == 6 and sorted(e7.sigma_rel.unique()) == [0.01, 0.05, 0.2]
        and e7.N.min() == 1000 and e7.N.max() == 64000 and d7["trials"] == 300
        and len(e7[~e7.sector].groupby(["kind", "eps"])) == 5
        and "np.abs(ang) < np.pi / 3" in (ROOT / "experiments" / "exp7_invariance.py").read_text())
    ni = e7[e7.kind == "near_invariant"].groupby("eps").true_rot_defect.first()
    chk("SN1: eps 0.02, 0.05, 0.2 give rotation defects 0.025, 0.063, 0.23",
        list(ni.index) == [0.02, 0.05, 0.2] and near(ni[0.02], 0.025, 0.0006)
        and near(ni[0.05], 0.063, 0.0006) and near(ni[0.2], 0.23, 0.006))
    inv = e7[(e7.kind == "invariant") & (~e7.sector) & (e7.sigma_rel == 0.05)].set_index("N")
    chk("SN1: threshold from N=1000, SymCert 0.97 at 4000, all from 16000 (about four times the data)",
        inv.dim_naive_t[1000] == 1.0 and near(inv.dim_sym[4000], 0.97, 0.006)
        and inv.dim_sym[16000] == 1.0 and inv.dim_sym[64000] == 1.0 and inv.dim_sym[1000] == 0.0)
    n02 = e7[(e7.kind == "near_invariant") & (e7.eps == 0.02)]
    chk("SN1: defect 0.025 certified at noise 0.01 for every N and at 0.05 from N=16000",
        n02[n02.sigma_rel == 0.01].dim_sym.min() == 1.0
        and n02[(n02.sigma_rel == 0.05) & (n02.N >= 16000)].dim_sym.min() == 1.0
        and n02[(n02.sigma_rel == 0.05) & (n02.N < 16000)].dim_sym.max() < 1.0)
    x7_sig = sorted(e7.sigma_rel.unique())
    chk("SN1: defect 0.025 declared by the threshold at every noise level in at least 97% from N = 16,000",
        near(n02[n02.N >= 16000].dim_naive_t.min(), 0.97, 0.0075)
        and n02[n02.N >= 16000].dim_naive_t.min() >= 0.97
        and set(n02[n02.N >= 16000].sigma_rel) == set(x7_sig),
        f"{n02[n02.N >= 16000].dim_naive_t.min():.4f}")
    g02 = n02[n02.sigma_rel == 0.2].set_index("N")
    chk("SN1: at noise 0.2 SymCert certifies the defect-0.025 invariance in no replication"
        " up to N = 16,000 and in 8% (25 of 300) at N = 64,000",
        g02[g02.index <= 16000].dim_sym.max() == 0 and round(g02.dim_sym[64000] * 300) == 25
        and max(x7_sig) == 0.2,
        f"{n02[n02.sigma_rel == 0.2].dim_sym.max():.4f}")
    sq7 = e7[~e7.sector]
    f4 = sq7[(sq7.sigma_rel == 0.2) & (sq7.N == 4000)]
    chk("SN1: square inputs, noise 0.2, N = 4000: threshold false 3.3% (defect 0.025), 1.3% (invariant);"
        " SymCert never false with these inputs",
        near(f4[(f4.kind == "near_invariant") & (f4.eps == 0.02)].fc_naive_t.iloc[0], 0.033, 0.0005)
        and near(f4[f4.kind == "invariant"].fc_naive_t.iloc[0], 0.013, 0.0005)
        and sq7.fc_sym.max() == 0)
    above = e7[(e7.kind == "near_invariant") & (e7.eps >= 0.05)]
    chk("SN1: defects above tolerance declared by threshold in at most 0.7%, never by SymCert",
        near(above.dim_naive_t.max(), 0.007, 0.0006) and above.dim_sym.max() == 0)
    sec = e7[e7.sector]
    chk("SN1: sector threshold up to 0.72 at noise 0.05 and 0.76 overall, both targets; SymCert never",
        near(sec[sec.sigma_rel == 0.05].fc_naive_t.max(), 0.72, 0.006)
        and near(sec.fc_naive_t.max(), 0.76, 0.006)
        and sec[sec.kind == "invariant"].fc_naive_t.max() > 0.5
        and sec[sec.kind == "anisotropic"].fc_naive_t.max() > 0.5
        and e7.fc_sym.max() == 0 and sec.dim_sym.max() == 0)
    s05 = sec[sec.sigma_rel == 0.05].pivot_table(index="N", columns="kind", values="fc_naive_t")
    chk("Fig. S6b: invariant and non-invariant targets almost indistinguishable (difference < 0.03)",
        (s05.invariant - s05.anisotropic).abs().max() < 0.03)
    # Extrapolation factor of the inputs, recomputed on the first replication's
    # training inputs of every setting (the text quotes the range over settings).
    import exp7_invariance as x7
    from hsd.coverage import extrapolation_factor
    from hsd.invariance import InvarianceProblem
    iprob = InvarianceProblem.build(2, x7.DEG, "linear", "box", scale=1.0)
    specs = ([("invariant", 0.0, s) for s in (False, True)]
             + [("near_invariant", e, False) for e in (0.02, 0.05, 0.2)]
             + [("anisotropic", 0.0, s) for s in (False, True)])
    fac = []
    for i, ((kind, eps, sector), N, srel) in enumerate(product(specs, x7.NS, x7.SIGMAS)):
        rng = np.random.default_rng(7000 + 1637 * i)
        rng.uniform(-1, 1, size=(200_000, 2))
        X = rng.uniform(-1, 1, size=(N, 2))
        if sector:
            keep = np.abs(np.arctan2(X[:, 1], X[:, 0])) < np.pi / 3
            while keep.sum() < N:
                Zs = rng.uniform(-1, 1, size=(N, 2))
                X = np.vstack([X[keep], Zs[np.abs(np.arctan2(Zs[:, 1], Zs[:, 0])) < np.pi / 3]])
                keep = np.ones(len(X), bool)
            X = X[:N]
        ef = extrapolation_factor(iprob, X)
        fac.append(dict(sector=sector, N=N, bound=ef["bound"], kappa=ef["kappa"]))
    fac = pd.DataFrame(fac)
    fs, fq = fac[fac.sector], fac[~fac.sector]
    chk("SN1: sector inputs give a factor between 2e4 and 3e4 (kappa between 1.4e8 and 2.6e8) at every N",
        len(fs) == 6 * len(x7.NS) and 2e4 <= fs.bound.min() < 2.1e4 and 2.9e4 < fs.bound.max() <= 3e4
        and 1.4e8 <= fs.kappa.min() < 1.5e8 and 2.5e8 < fs.kappa.max() <= 2.6e8,
        f"bound {fs.bound.min():.4g}-{fs.bound.max():.4g}, kappa {fs.kappa.min():.4g}-{fs.kappa.max():.4g}")
    sqmax = fq.groupby("N").bound.max()
    chk("SN1: largest factor 2.2, 1.5, 1.2 and 1.1 at N = 1000, 4000, 16,000, 64,000 for inputs covering the square"
        " (first replication of each setting)",
        len(fq) == 15 * len(x7.NS) and x7.NS == [1000, 4000, 16000, 64000]
        and [round(sqmax[n], 1) for n in x7.NS] == [2.2, 1.5, 1.2, 1.1],
        sqmax.round(3).to_dict())

    # ------------------------------------------------------------- exp12 (Supplementary Note 2)
    d12 = load("exp12_realdata")
    import exp12_realdata as x12
    chk("lynx-hare: 1900-1920, 21 counts, 19 pairs, scale 77.4 thousand",
        x12.YEAR[0] == 1900 and x12.YEAR[-1] == 1920 and len(x12.YEAR) == 21
        and d12["n_points"] == 19 and near(d12["scale"], 77.4, 1e-9))
    Zl = np.stack([x12.HARE, x12.LYNX], axis=1) / d12["scale"]
    Xl = Zl[1:-1]
    Yl = (Zl[2:] - Zl[:-2]) / 2.0
    chk("lynx-hare: states in [0.10, 1] x [0.08, 0.77]",
        near(Xl[:, 0].min(), 0.10, 0.005) and near(Xl[:, 0].max(), 1.0, 0.005)
        and near(Xl[:, 1].min(), 0.08, 0.005) and near(Xl[:, 1].max(), 0.77, 0.005))
    spec = d12["plug_in_spectrum_target"]
    chk("lynx-hare: Q = 12, spectrum (0.18, 0.33, 0.68, 1.17, 1.31, 1.40), factor 58, tolerance 0.65",
        DefectProblem.build(2, 2, "affine", "box", scale=1.0).Q == 12
        and all(near(a, b, 0.006) for a, b in zip(spec, (0.18, 0.33, 0.68, 1.17, 1.31, 1.40)))
        and near(d12["kappa_bound"], 58, 0.5) and near(d12["smallest_certified_tolerance"], 0.65, 0.006))
    chk("lynx-hare: a threshold at 0.3 would report a one-dimensional algebra",
        sum(v <= 0.3 for v in spec) == 1)
    # Positive quadrant [0,1]^2 as target: translate the states by -1/2 and use
    # the box of half-width 1/2 (the defect is invariant under translation).
    from hsd import fit_ols
    from hsd.certify import certify_direction
    pq = DefectProblem.build(2, 2, "affine", "box", scale=0.5)
    fq = fit_ols(pq, Xl - 0.5, Yl, robust=True)
    rq, Thq = pq.spectrum(fq.beta)
    kq = extrapolation_factor(pq, Xl - 0.5)["bound"]
    uq = min(certify_direction(pq, Thq[:, k], fq, 0.05).upper for k in range(pq.P))
    chk("lynx-hare on [0,1]^2: factor 7.2, spectrum begins (0.20, 0.63), tolerance 0.84",
        near(kq, 7.2, 0.05) and near(rq[0], 0.20, 0.006) and near(rq[1], 0.63, 0.006) and near(uq, 0.84, 0.006),
        f"{kq:.2f}, ({rq[0]:.3f}, {rq[1]:.3f}), {uq:.3f}")

    # ------------------------------------------------------------- Methods: Monte Carlo error
    def max_halfwidth(Tn):
        return max(max(k / Tn - cp(k, Tn)[0], cp(k, Tn)[1] - k / Tn) for k in range(Tn + 1))
    chk("Clopper-Pearson interval extends at most 0.058 (T=300), 0.072 (T=200), 0.116 (T=80)",
        near(max_halfwidth(300), 0.058, 0.0006) and near(max_halfwidth(200), 0.072, 0.0006)
        and near(max_halfwidth(80), 0.116, 0.0006),
        [round(max_halfwidth(t), 4) for t in (300, 200, 80)])
    reps = [d1["trials"], d2["trials"], d3["trials"], d4["trials"], d5["trials"],
            int(t5.n_traj_trials.min()), d6["trials"], d7["trials"], d8["trials"],
            d8["width_trials"], d8["boundary_trials"], d8["selected_trials"], d10["trials"],
            d11["trials"], d13["trials"], d14["trials"], d15["trials"], d16["trials"],
            d17["trials"], int(c17.runs.min()), d18["trials"]]
    chk("80 is the smallest number of replications used", min(reps) == 80)
    hw = {t: max_halfwidth(t) for t in set(reps)}
    chk("Methods: the bounds hold for T >= 300, T >= 200 and T >= 80 (every T used)",
        all(hw[t] <= (0.0585 if t >= 300 else 0.0725 if t >= 200 else 0.1165) for t in hw),
        {t: round(v, 4) for t, v in sorted(hw.items())})

    # ------------------------------------------------------------- Discussion: problem sizes
    allQ = (list(e6.Q) + list(q.Q) + list(e11.Q) + [r["Q"] for r in d19["rows"]]
            + [2 * comb(max(d9["degrees"]) + 4 + 2, 2)])
    allP = list(e6.P) + list(e11.P) + list(df.P)
    chk("Discussion: n <= 6, Q <= 210, P <= 36",
        e11.n.max() == 6 and max(allQ) == 210 and max(allP) == 36)

    # ------------------------------------------------------------- compute
    clog = json.loads((ROOT / "results" / "compute_log.json").read_text())
    exps = [k for k in clog if k.startswith("exp")]
    chk("about 3.6 CPU-hours for one complete pass of the 19 experiments",
        len(exps) == 19 and near(clog["total_cpu_hours"], 3.6, 0.05))

    bad = [c for c in checks if not c[1]]
    for lab, ok, det in checks:
        print(("  OK  " if ok else "  !!  ") + lab + (f"   [{det}]" if det else ""))
    print(f"\n{len(checks) - len(bad)}/{len(checks)} numeric claims verified against results/")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
