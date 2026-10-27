"""Write every table of the manuscript and the Supplementary Information from
the saved results, and print the diagnostics that the running text quotes.

Each table is emitted as a complete LaTeX environment, so the manuscript never
carries a number by hand.  ``HSD_RESULTS`` redirects the input directory (used
to test the script on smoke-test output).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
RES = Path(os.environ.get("HSD_RESULTS", ROOT / "results"))
OUT = ROOT / "paper" / "tables"
ALPHA = DELTA = 0.05


def lof_theory(r):
    """Power of the lack-of-fit test of degree 3 against degree k, from the
    stored parameters (noncentrality N eps^2 / ((2k+1) sigma^2))."""
    import sys as _s
    _s.path.insert(0, str(ROOT / "src"))
    from hsd.selection import lof_power
    k, N = int(r.k), int(r.N)
    m3, mk = 10, (k + 1) * (k + 2) // 2
    lam = N * float(r.eps) ** 2 / (2 * k + 1) / float(r.sigma) ** 2
    return lof_power(lam, 2 * (mk - m3), 2 * (N - mk), 0.05)


def load(n):
    return json.loads((RES / f"{n}.json").read_text())


def have(n):
    return (RES / f"{n}.json").exists()


def cp(k, T, level=0.95):
    a = 1 - level
    lo = 0.0 if k == 0 else stats.beta.ppf(a / 2, k, T - k + 1)
    hi = 1.0 if k == T else stats.beta.ppf(1 - a / 2, k + 1, T - k)
    return float(lo), float(hi)


def fmt_ci(rate, T):
    k = int(round(rate * T))
    lo, hi = cp(k, T)
    return f"{rate:.3f} [{lo:.3f}, {hi:.3f}]"


def table(name, colspec, header, body, caption, label, size=r"\small", maxh="0.9"):
    OUT.mkdir(exist_ok=True)
    (OUT / f"{name}.tex").write_text(
        # adjustbox shrinks a table only if it is wider than the text block.
        # Supplementary tables may sit where they are input, so that none floats
        # above the "Supplementary Tables" heading.
        "\\begin{table}" + ("[!htbp]" if label.startswith("stab:") else "[!tbp]") +
        "\n\\centering" + size + "\n"
        "\\begin{adjustbox}{max width=\\textwidth, max totalheight=" + maxh + "\\textheight}\n"
        "\\begin{tabular}{" + colspec + "}\n"
        "\\toprule\n" + header + "\n\\midrule\n" + "\n".join(body) +
        "\n\\bottomrule\n\\end{tabular}\n\\end{adjustbox}\n\\caption{" + caption +
        "}\n\\label{" + label + "}\n\\end{table}\n")


# ------------------------------------------------------------ calibration
def calibration():
    d1 = load("exp1_calibration")
    df = pd.DataFrame(d1["rows"])
    T = d1["trials"]
    sym = df[df.d_true > 0]
    rows = [(rf"fixed threshold $\tau={t}$", f"fc_tau{t}", f"det_tau{t}", f"rec_tau{t}")
            for t in d1["taus"]]
    rows += [("largest eigengap", "fc_gap", "det_gap", "rec_gap"),
             ("nullspace rank test", "fc_rank", "det_rank", "rec_rank"),
             ("split significance test", "fc_split", "det_split", "rec_split"),
             ("Weyl perturbation bound", "fc_weyl", "det_weyl", "rec_weyl")]
    ours = [(r"\symcert{}", "fc_sym", "det_sym", "rec_sym"),
            (r"\symcert{} + sample splitting", "fc_symp", "det_symp", "rec_symp")]
    body = []
    for block in (rows, ours):
        for lab, fc, det, rec in block:
            f = df[fc]
            body.append(f"{lab} & {f.mean():.3f} & {fmt_ci(f.max(), T)} & "
                        f"{int((f > ALPHA).sum())} & {sym[det].mean():.3f} & "
                        f"{sym[rec].mean():.3f} \\\\")
        body.append(r"\midrule")
    body.pop()
    table("calibration", "lrrrrr",
          r"& \multicolumn{3}{c}{false certification} & \multicolumn{2}{c}{symmetric systems} \\"
          "\n" r"\cmidrule(lr){2-4}\cmidrule(lr){5-6}" "\n"
          r"method & mean & worst setting [95\% CI] & settings $>\alpha$ & detection & recovery \\",
          body,
          r"\textbf{Calibration and power over " + f"{len(df)}" + r" settings.} Ten systems,"
          r" sample sizes $N = " + ", ".join(f"{N:g}" for N in d1["Ns"][:-1]) + r"$ and $"
          + f"{d1['Ns'][-1]:g}" + r"$, and noise levels $" + ", ".join(f"{s:g}" for s in d1["sigmas"][:-1])
          + r"$ and $" + f"{d1['sigmas'][-1]:g}" + r"$; " + f"{T}" + r" replications per setting."
          r" ``Mean'' is the average false-certification rate over settings and ``worst"
          r" setting'' its maximum, with an exact Clopper--Pearson interval for the Monte"
          r" Carlo error of that setting; the next column counts settings whose rate exceeds"
          r" $\alpha = 0.05$ (out of " + f"{len(df)}" + r"). Detection (declared dimension at"
          r" least the true one; for the split-sample test, which declares at most one"
          r" direction, any declaration) and recovery (detection, with the true algebra"
          r" inside the declared subspace to within $10^\circ$) are averaged over the "
          + f"{len(sym)}" + r" settings of the four symmetric systems. Among the thresholds"
          r" examined none keeps the worst-setting rate near $\alpha$; the rank and"
          r" split-sample tests are valid tests of exact symmetry and still poor discovery"
          r" rules; the Weyl certificate is valid but never certifies.",
          "tab:calibration")


# ------------------------------------------------------------- robustness
def robustness():
    d5 = load("exp5_robustness")
    a = pd.DataFrame(d5["design"])
    t = pd.DataFrame(d5["trajectory"])
    names = {("gauss", 0.0): "Gaussian", ("t", 0.0): r"Student $t_4$",
             ("laplace", 0.0): "Laplace", ("gauss", 1.0): "heteroskedastic Gaussian"}
    lab = {"ols": "OLS (exact)", "hc0": "White", "cluster": "cluster-robust",
           "hac": "Newey--West"}
    body = [r"\multicolumn{5}{l}{\emph{Independent rows, error distribution varied}} \\"]
    for key in names:
        g = a[(a.noise == key[0]) & (a.hetero == key[1])]
        for k, c in enumerate(("ols", "hc0")):
            s = g[g["cov"] == c]
            body.append(f"{names[key] if k == 0 else ''} & {lab[c]} & "
                        f"{s.region_coverage.min():.3f} & {s.coverage.min():.3f} & "
                        f"{s.fc.max():.3f} \\\\")
    body.append(r"\midrule")
    body.append(r"\multicolumn{5}{l}{\emph{" + f"{d5['n_segments']}" + r" trajectories}} \\")
    for mode, ml in (("clean_states", "exact states"),
                     ("noisy_states_fd", "noisy states, differenced")):
        g = t[t["mode"] == mode]
        for k, c in enumerate(("hc0", "cluster")):
            first = ml if k == 0 else ""
            reg, cert, fcm = g["reg_" + c].min(), g["cert_" + c].min(), g["fc_" + c].max()
            body.append(f"{first} & {lab[c]} & {reg:.3f} & {cert:.3f} & {fcm:.3f} " + "\\\\")
    if have("exp15_dependence"):
        e = pd.DataFrame(load("exp15_dependence")["rows"])
        Gmax = e.G.max()
        body.append(r"\midrule")
        body.append(r"\multicolumn{5}{l}{\emph{AR(1) errors, "
                    + f"{Gmax}" + r" trajectories}} \\")
        for phi in (0.9, 0.98):
            g = e[(e.phi == phi) & (e.G == Gmax)]
            plab = "$\\phi = " + f"{phi}" + "$"
            for k, c in enumerate(("ols", "hc0", "cluster", "hac")):
                first = plab if k == 0 else ""
                reg, cert, fcm = (g["reg_" + c].min(), g["cert_" + c].min(),
                                  g["fc_" + c].max())
                body.append(f"{first} & {lab[c]} & {reg:.3f} & {cert:.3f} & {fcm:.3f} " + "\\\\")
    words = {2: "two", 3: "three", 4: "four"}
    pairs = a[["N", "sigma_rel"]].drop_duplicates().sort_values("N")
    pairs_txt = " and ".join(f"$({int(p.N)}, {p.sigma_rel:g})$" for p in pairs.itertuples())
    # Every row of the independent-rows block covers each system at each pair.
    assert (a.groupby(["noise", "hetero", "cov"]).size()
            == a.system.nunique() * len(pairs)).all()
    sn = d5["state_noise"]
    sn_txt = ", ".join(f"{s:g}" for s in sn[:-1]) + f" and {sn[-1]:g}"
    ar_txt = ""
    if have("exp15_dependence"):
        d15 = load("exp15_dependence")
        ar_txt = (r"; AR(1): " + words[e.system.nunique()] + r" systems, "
                  + f"{d15['trials']}" + r" replications each")
    n_steps = int(round(d5["t_segment"] / d5["dt"]))
    table("robustness", "llrrr",
          r"setting & region & region coverage & certificate coverage & false cert.\ \\",
          body,
          r"\textbf{What the certificate inherits from the coefficient region.} Nominal"
          r" coverage $0.95$. Region coverage is the fraction of replications in which the"
          r" coefficient region contains the true coefficients; certificate coverage the"
          r" fraction in which the interval contains the true defect of every plug-in"
          r" direction; both are minima, and the false-certification rate a maximum, over"
          r" the settings in each row (independent rows: " + words[a.system.nunique()]
          + r" systems, each at ($N$, noise level) $=$ " + pairs_txt + ", "
          + f"{d5['trials']}" + r" replications each; trajectories: "
          + words[t.system.nunique()] + r" systems, with exact states or with state noise of"
          r" standard deviation " + sn_txt + r" (" + f"{100 * sn[0]:g}" + r"\% to "
          + f"{100 * sn[-1]:g}" + r"\% of the box half-width), "
          + f"{int(t.n_traj_trials.iloc[0])}" + r" replications each" + ar_txt + r")."
          r" Each trajectory starts from a uniform initial condition in the box and runs for "
          + f"{n_steps}" + r" steps of " + f"{d5['dt']:g}" + r"; only its part inside the box is"
          r" kept (at most " + f"{n_steps}" + r" points), and clusters and Newey--West groups"
          r" are trajectories. With noisy states the regressors carry error and no region"
          r" is guaranteed to cover; the state noise here is small, so the bias is"
          r" negligible and the differenced derivatives dominate the error (main text)."
          r" With serially correlated errors only the cluster-robust and Newey--West"
          r" regions are designed to cover.",
          "tab:robustness")


# --------------------------------------------------------- supplementary
def trajectory():
    d2 = load("exp2_coverage")
    t = pd.DataFrame(d2["trajectories"])
    pretty = {"hopf": "Hopf", "vanderpol": "Van der Pol", "lorenz": "Lorenz"}
    body = []
    for _, r in t.iterrows():
        kap = r"$\infty$" if not np.isfinite(r.kappa) else f"{r.kappa:.1f}"
        body.append(f"{pretty[r.system]} & {r.design} & {int(r.n_points)} & {kap} & "
                    f"{int(r.d_true)} & {int(r.dim_naive_data)} & {int(r.dim_sym)} \\\\")
    table("trajectory", "llrrrrr",
          r"system & design & points & $\sqrt{\kappa\kappa'}$ & true dim.\ & nullspace dim.\ & \symcert{} dim.\ \\",
          body,
          r"\textbf{Trajectory data and the extrapolation factor.} One realisation per row."
          r" Target measure uniform on $[-3,3]^2$ (Van der Pol, degree-3 class), $[-1.5,1.5]^2$"
          r" (Hopf, degree 3) and $[-20,20]^3$ (Lorenz, degree 2); right-hand side measured with"
          r" noise level $0.02$, states exact. The single trajectory starts at $0.7$ times the"
          r" half-width in every coordinate ($(1,1,20)$ for Lorenz) and is sampled at $4000$"
          r" times over $[0,60]$ ($[0,40]$ for Lorenz), the first $1000$ discarded as transient;"
          r" the eight trajectories start uniformly in the box and are sampled at $400$ times"
          r" over $[0,8]$; the uniform design has $3000$ points drawn from the target measure."
          r" The single Hopf trajectory converges onto the unit circle, an algebraic curve of"
          r" low degree, so every field divisible by $x^2 + y^2 - 1$ in the space that contains"
          r" every bracket vanishes on the data to within the eigenvalue threshold, and the"
          r" factor is reported as infinite; on the Van der Pol limit cycle and the Lorenz"
          r" attractor the data lie only near the zero sets of other nonzero fields of that space, and the factor is"
          r" finite but large. Either way a defect measured on them says little about the target"
          r" domain."
          r" The nullspace rule happens to give the right dimension on every row here, but"
          r" on the single Hopf trajectory it does so without support from the data;"
          r" \symcert{} declines to certify exactly there.",
          "stab:trajectory")


def qsweep():
    d4 = load("exp4_resolution")
    q = pd.DataFrame(d4["qsweep"])
    def num(x, spec):
        return r"$\infty$" if not np.isfinite(x) else f"{x:{spec}}"

    body = []
    for (deg, Q), g in q.groupby(["degF", "Q"]):
        for _, r in g.sort_values("N").iterrows():
            body.append(f"{int(deg)} & {int(Q)} & {int(r.N)} & {num(r.upper_med, '.4f')} & "
                        f"{num(r.upper_pointwise_med, '.4f')} & {r.lecam:.5f} & "
                        f"{num(r.upper_med / r.lecam, '.1f')} & "
                        f"{num(r.upper_pointwise_med / r.lecam, '.1f')} \\\\")
        body.append(r"\addlinespace")
    body.pop()
    # The main text quotes medians of the two ratios over the sample sizes at
    # which every bound is finite and below 1; find that range from the data.
    fine = (np.isfinite(q.upper_pointwise_med) & (q.upper_pointwise_med < 1)
            & (q.upper_med < 1))
    ok = fine.groupby(q.N).all()
    Nok = sorted(int(N) for N in ok.index if all(ok[ok.index >= N]))
    inf_cells = q[~np.isfinite(q.upper_pointwise_med)]
    assert set(inf_cells.degF) == {7}
    Ninf = int(inf_cells.N.max())
    table("qsweep", "rrrrrrrr",
          r"degree & $Q$ & $N$ & $U$ (median) & pointwise (median) & Le~Cam & $U$/Le~Cam & pointwise/Le~Cam \\",
          body,
          r"\textbf{The gap to the Le~Cam bound grows with the model class.} The exactly"
          r" symmetric Hopf system at noise level $0.1$ fitted in polynomial classes of"
          r" degree 3, 5 and 7; medians over " + f"{d4['trials']}" + r" replications. $U$ is"
          r" the simultaneous certificate for the rotation generator, ``pointwise'' the"
          r" Gaussian-concentration bound (valid because the rotation is fixed in advance),"
          r" and Le~Cam the bound of the main text at power $0.5$ evaluated on each design."
          r" The pointwise bound is infinite ($\infty$) for degree 7 at $N \le " + f"{Ninf}"
          + r"$: its lower bound on the denominator of the defect reached zero in at least"
          r" half of the replications. The median factors quoted in the main text are over"
          r" $N = " + f"{Nok[0]:,}".replace(",", "{,}") + r"$ to $"
          + f"{Nok[-1]:,}".replace(",", "{,}") + r"$, where every bound is finite and"
          r" below 1.",
          "stab:qsweep")


def scaling():
    d11 = load("exp11_scaling")
    words = {3: "three", 4: "four", 5: "five", 6: "six"}
    e = pd.DataFrame(d11["rows"])
    Ns = sorted(e.N.unique())
    k = len(Ns)
    body = []
    for n, g in e.groupby("n"):
        gi = g.set_index("N")
        tol = " & ".join(f"{gi.upper_med.get(N, float('nan')):.4f}" for N in Ns)
        det = " & ".join(f"{gi.detect.get(N, float('nan')):.2f}" for N in Ns)
        body.append(f"{n} & {int(gi.P.iloc[0])} & {int(gi.Q.iloc[0])} & "
                    f"{int(gi.d_true.iloc[0])} & {tol} & {det} & "
                    f"{gi.cpu_seconds_per_certificate.max():.3f} \\\\")
    hdr = " & ".join(rf"$N{{=}}{N}$" for N in Ns)
    assert (e.fc == 0).all()
    table("scaling", "r" * (5 + 2 * k),
          r"$n$ & $P$ & $Q$ & true dim.\ & \multicolumn{" + f"{k}" + r"}{c}{median $U$, minimal-defect direction}"
          r" & \multicolumn{" + f"{k}" + r"}{c}{detection rate} & CPU-s / cert.\ \\"
          "\n" + rf"\cmidrule(lr){{5-{4 + k}}}\cmidrule(lr){{{5 + k}-{4 + 2 * k}}}" + "\n"
          r" & & & & " + hdr + " & " + hdr + r" & \\",
          body,
          r"\textbf{Scaling in the state dimension.} Linear systems built from rotation"
          r" blocks, whose symmetry algebra is the commutant of the coefficient matrix and"
          r" has dimension $n$; noise level $" + f"{d11['sigma_rel']:g}" + r"$, "
          + f"{d11['trials']}" + r" replications per cell. The first block is the median"
          r" certified upper bound $U$ on the defect of the plug-in direction with the"
          r" smallest defect; the detection rate is the fraction of replications in which"
          r" the dimension of the certified subspace is at least $n$. The candidate search"
          r" space $P = n^2$ grows quadratically with no cost to validity; the width grows,"
          r" but because $P$ and $Q$ grow together this design does not separate their"
          r" effects. False certification was zero in all " + f"{len(e)}" + r" cells"
          r" (" + words[e.n.nunique()] + r" dimensions, " + words[k] + r" sample sizes)."
          r" CPU time is per certificate of the plug-in subspace, on one core.",
          "stab:scaling")


def closure():
    c = pd.DataFrame(load("exp17_units_closure")["closure"])
    pretty = {"sphere_flow3": r"sphere flow, $\mathfrak{so}(3)$",
              "rigid_sym": "symmetric top", "lin_rot2": "planar rotation"}
    body = []
    for _, r in c.iterrows():
        mc = "---" if r.median_closure is None or pd.isna(r.median_closure) else f"{r.median_closure:.3f}"
        xc = "---" if r.max_closure is None or pd.isna(r.max_closure) else f"{r.max_closure:.3f}"
        nwc = int(r.get("runs_with_commutators", 0) or 0)
        if int(r.runs_dim_ge2) == 0:
            cc = "---"
        else:
            cc = "negligible" if nwc == 0 else f"{r.commutators_certified:.2f}"
        body.append(f"{pretty[r.system]} & {int(r.N)} & {r.sigma_rel} & {r.mean_dim:.2f} & "
                    f"{int(r.runs_dim_ge2)}/{int(r.runs)} & {mc} & {xc} & {cc} \\\\")
    table("closure", "lrrrrrrr",
          r"system & $N$ & noise & mean dim.\ & runs dim.\ $\ge 2$ & closure (median) & closure (max) & commutators certified \\",
          body,
          r"\textbf{Closure of certified subspaces under the bracket.} ``Mean dim.'' is the"
          r" mean dimension of the certified plug-in subspace. For each certified subspace of"
          r" dimension at least two, the closure defect is the largest, over pairs of an"
          r" $L^2(\nu)$-orthonormal basis, of the $L^2(\nu)$ norm of the part of the commutator"
          r" outside the subspace relative to the size of its two terms (0 for a Lie"
          r" subalgebra, at most $\sqrt 2$). Affine generators are closed under the bracket,"
          r" so each commutator is itself a candidate generator; the last column is the"
          r" fraction of those runs in which every commutator larger than $5\%$ of its two"
          r" terms is certified at $\delta = 0.05$ (``negligible'': all commutators smaller,"
          r" as for an abelian algebra; ``---'': no subspace of dimension two or more was"
          r" certified). $100$ replications per row.",
          "stab:closure")


def dictionary():
    e = pd.DataFrame(load("exp16_dictionary")["rows"])
    body = []
    for (fam, eps), g in e.groupby(["family", "eps"], sort=False):
        name = "radial flow" if fam == "radial" else "damped pendulum"
        tr = g.true_rot_defect.iloc[0] if fam == "radial" else None
        trs = f"{tr:.3f}" if tr is not None and not pd.isna(tr) else "---"
        best = g[(g.N == g.N.max()) & (g.sigma_rel == g.sigma_rel.min())].iloc[0]
        body.append(f"{name} & {eps:g} & {trs} & {g.true_min_defect.iloc[0]:.3f} & "
                    f"{int(g.d_true.iloc[0])} & {g.fc_sym.max():.3f} & {g.fc_thr.max():.3f} & "
                    f"{best.dim_sym:.2f} & {best.upper_med:.3f} & {g.refute.max():.2f} \\\\")
    table("dictionary", "lrrrrrrrrr",
          r"family & $\varepsilon$ & rotation defect & min.\ defect & true dim.\ & \multicolumn{2}{c}{worst false cert.} & dim.\ & $U$ & refuted \\"
          "\n" r"\cmidrule(lr){6-7}" "\n"
          r" & & & & & \symcert{} & threshold & & & \\",
          body,
          r"\textbf{Non-polynomial systems fitted with dictionaries that contain them.} The"
          r" planar radial flow $\dot r = r(1-r)$, $\dot\varphi = 1/(1+r^2)$ plus"
          r" $\varepsilon x^2\partial_x$ (exactly rotation invariant at $\varepsilon = 0$), on"
          r" the unit disc, is fitted with quadratic polynomials and four radial functions;"
          r" the damped pendulum $\dot\theta = \omega$, $\dot\omega = -\sin\theta - 0.1\,\omega$,"
          r" on $[-1.5, 1.5]^2$, with a polynomial and trigonometric library. Affine"
          r" candidate generators; the target measure is uniform on the domain. Worst"
          r" false-certification rates are over $N = " + ", ".join(f"{N:g}" for N in sorted(e.N.unique()))
          + r"$ and noise levels $" + ", ".join(f"{s:g}" for s in sorted(e.sigma_rel.unique()))
          + r"$, " + f"{e.trials.iloc[0]}" +
          r" replications each; ``dim.''\ (the mean dimension of the certified plug-in"
          r" subspace) and the median upper bound $U$ on the minimal-defect generator are at"
          r" the largest sample size and smallest noise level; ``refuted'' is the largest"
          r" fraction of replications in which the lower certificate excludes every"
          r" $\delta$-approximate symmetry.",
          "stab:dictionary")


def units():
    d = load("exp17_units_closure")
    u = pd.DataFrame(d["units"])
    body = []
    for sysname, g in u.groupby("system", sort=False):
        nm = "broken Hopf" if sysname == "broken_hopf" else "Lotka--Volterra"
        for k, (_, r) in enumerate(g.iterrows()):
            body.append(f"{nm if k == 0 else ''} & {r.unit_factor:g} & {r.defect_euclidean:.4f} & "
                        f"{r.defect_nondimensional:.4f} & {r.min_defect_euclidean:.4f} & "
                        f"{r.min_defect_nondimensional:.4f} \\\\")
        body.append(r"\addlinespace")
    body.pop()
    s = pd.DataFrame(d["units_sim"])
    # Summarise the certification runs in one sentence; the wording below is
    # asserted against the result file so that it cannot drift.
    hit = s[s.certify_rate > 0]
    assert set(hit.metric) == {"euclidean"} and set(hit.system) == {"lotka_volterra"}
    assert (hit.fc == hit.certify_rate).all()
    cs = " and ".join(f"$c = {c:g}$" for c in sorted(hit.unit_factor))
    rate = f"{hit.certify_rate.min():.2f}" if hit.certify_rate.nunique() == 1 else \
        f"{hit.certify_rate.min():.2f} to {hit.certify_rate.max():.2f}"
    rate = "every" if rate == "1.00" else f"a fraction {rate} of"
    cvals = ", ".join(f"{c:g}" for c in sorted(s.unit_factor.unique()))
    sim_text = (f"With the Euclidean metric, Lotka--Volterra was certified in {rate} "
                f"replication{'s' if rate != 'every' else ''} at {cs}, and every certification was false;"
                " in every other case, and always with the nondimensional metric, nothing was certified.")
    table("units", "lrrrrr",
          r"system & unit factor $c$ & \multicolumn{2}{c}{defect of fixed generator} & \multicolumn{2}{c}{smallest defect in class} \\"
          "\n" r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}" "\n"
          r" & & Euclidean & nondimensional & Euclidean & nondimensional \\",
          body,
          r"\textbf{The units of the state matter unless the metric is declared.} One state"
          r" coordinate is re-expressed in units $c$ times larger ($x = \mathrm{diag}(c,1)z$)."
          r" The fixed generator is the rotation for the broken-Hopf system and the"
          r" minimal-defect generator for Lotka--Volterra, transported to the new"
          r" coordinates; ``nondimensional'' uses $W = \mathrm{diag}(1/s_i^2)$ with $s_i$ the"
          r" root-mean-square size of coordinate $i$ over the target box, which coincides with"
          r" the transported metric. Certification in the new units at $N = 12800$, noise"
          r" level $0.05$, judged by the nondimensional defect of the physical system ("
          + f"{s.trials.iloc[0]}" + r" replications at $c = " + cvals + r"$, both systems,"
          r" both metrics). " + sim_text,
          "stab:units")


def selection():
    d14 = load("exp14_selection")
    e = pd.DataFrame(d14["rows"])
    # The rotation defect is a design value (a multiple of delta, matched by
    # bisection); print the multiple, not the floating-point estimate.
    assert np.allclose(e.true_defect, e.defect_multiple * DELTA, atol=1e-9)
    body = []
    for (k, N), g in e.groupby(["k", "N"]):
        for _, r in g.sort_values("defect_multiple").iterrows():
            body.append(f"{int(k)} & {int(N)} & {r.defect_multiple:g} & {r.true_min_defect:.4f} & "
                        f"{r.lof_reject:.2f} & {lof_theory(r):.2f} & {r.fc_small:.2f} & "
                        f"{r.fc_gate:.2f} & {r.fc_split:.2f} & {r.fc_largest:.2f} & "
                        f"{r.fc_simultaneous:.2f} & {r.dim_gate:.2f} & {r.dim_split:.2f} & "
                        f"{r.dim_largest:.2f} & {r.dim_simultaneous:.2f} \\\\")
        body.append(r"\addlinespace")
    body.pop()
    b = e[e.defect_multiple == 1.0]
    lo_b, hi_b = b.true_min_defect.min(), b.true_min_defect.max()
    table("selection", "r" * 15,
          r"$k$ & $N$ & rotation & min.\ true & \multicolumn{2}{c}{lack-of-fit rejection} & \multicolumn{5}{c}{false certification} & \multicolumn{4}{c}{certified subspace dim.} \\"
          "\n" r"\cmidrule(lr){5-6}\cmidrule(lr){7-11}\cmidrule(lr){12-15}" "\n"
          r" & & defect$/\delta$ & defect & observed & theory & degree 3 & gate & split & largest & simult.\ & gate & split & largest & simult.\ \\",
          body,
          r"\textbf{Choosing the model class from the data: all " + f"{len(e)}" + r" settings.}"
          r" Hopf normal form plus $\varepsilon P_k(x)\partial_x$, with $P_k$ the Legendre"
          r" polynomial of degree $k$ and $\varepsilon$ set so that the true defect of the"
          r" rotation is the stated multiple of $\delta = 0.05$; noise level $"
          + f"{d14['sigma_rel']:g}" + r"$, " + f"{e.trials.iloc[0]}" + r" replications per row."
          r" ``Min.\ true defect'' is the smallest true defect of any generator in the"
          r" degree-$k$ class; ``theory'' is the power of the lack-of-fit test from the"
          r" noncentral $F$ distribution. Procedures: certify in degree 3; the lack-of-fit"
          r" gate; the gate applied on one half of the data with certification on the"
          r" other (split); certify in degree $k$ (largest); simultaneous over both classes."
          r" A false certification certifies a subspace containing a generator whose true"
          r" defect exceeds $\delta$; the last four columns give the mean dimension of the"
          r" certified subspace. Where the rotation defect equals $\delta$, the smallest true"
          r" defect is " + f"{lo_b:.3f}" + r" to " + f"{hi_b:.3f}" + r", so a certified"
          r" direction near the rotation can have a true defect just above $\delta$; such"
          r" certifications count as false.",
          "stab:selection", size=r"\scriptsize", maxh="0.74")


def drifters():
    d = load("exp19_drifters")
    lab = {"ols": "OLS (exact)", "hc0": "White", "cluster": "cluster-robust",
           "hac": "Newey--West"}
    body = []
    for r in d["rows"]:
        body.append(f"{r['degree']} ({r['Q']}) & {lab[r['cov']]} & {r['plug_rot']:.3f} & "
                    f"{r['upper_rot']:.3f} & {r['lower_rot']:.3f} & {r['certified_dim']} & "
                    f"{r['kappa']:.2f} \\\\")
        if r["cov"] == "hac" and r["degree"] < 3:
            body.append(r"\addlinespace")
    body.append(r"\midrule")
    for r in d["windows"]:
        body.append(f"1 (6), window {r['window']} h & cluster-robust & {r['plug_rot']:.3f} & "
                    f"{r['upper_rot']:.3f} & {r['lower_rot']:.3f} & {r['certified_dim']} & "
                    f"{r['kappa']:.2f} \\\\")
    body.append(r"\midrule")
    for r in d["years"]:
        body.append(f"1 (6), summer {r['year']} & Newey--West & {r['plug_rot']:.3f} & "
                    f"{r['upper_rot']:.3f} & {r['lower_rot']:.3f} & {r['certified_dim']} & "
                    f"{r['kappa']:.2f} \\\\")
    # For the summers with more drifters than coefficients the experiment also
    # stores the cluster-robust upper bound (the fit, hence the plug-in value
    # and the extrapolation factor, is the same).
    ident = [r for r in d["years"] if r.get("cluster_identified")]
    unident = [r for r in d["years"] if not r.get("cluster_identified")]
    for r in ident:
        body.append(f"1 (6), summer {r['year']} & cluster-robust & {r['plug_rot']:.3f} & "
                    f"{r['upper_rot_cluster']:.3f} & --- & --- & {r['kappa']:.2f} \\\\")
    body.append(r"\midrule")
    for r in d["simultaneous"]:
        body.append(f"1--3, simultaneous & {lab[r['cov']]} & --- & {r['upper_min']:.3f} & --- & --- & --- \\\\")
    r0 = d["rows"][0]
    # In the degree-1 class two plug-in defects vanish whatever the data (the
    # dilation about the fixed point and the fitted field itself).
    lin = [r for r in d["rows"] + d["windows"] if r["degree"] == 1]
    assert all(r["spectrum"][1] < 1e-6 for r in lin + d["years"])
    # These two come first in the nested plug-in subspaces; in the degree-1 rows
    # the column counts how many of them the region certifies (never more).
    lin_dims = sorted({int(r["certified_dim"]) for r in lin + d["years"]})
    assert lin_dims[-1] <= 2
    lin_txt = (", ".join(str(k) for k in lin_dims[:-1]) + " or " + str(lin_dims[-1])
               if len(lin_dims) > 1 else str(lin_dims[0]))
    assert all(r["Q"] == 6 for r in d["years"]) and len(unident) == 2 and len(ident) == 2
    cl_txt = (r" For single summers the cluster-robust region is unidentified in "
              + " and ".join(str(r["year"]) for r in unident) + r", which have fewer drifters ("
              + " and ".join(str(r["drifters"]) for r in unident) + r") than coefficients (6);"
              r" for " + " and ".join(f"{r['year']} ({r['drifters']} drifters)" for r in ident)
              + r" it gives $U = " + "$ and $".join(f"{r['upper_rot_cluster']:.3f}" for r in ident)
              + r"$ ($L$ and the certified subspace dimension are not reported for these two"
              r" rows); the Newey--West rows cover all four summers.")
    table("drifters", "llrrrrr",
          r"degree ($Q$) & region & plug-in & $U$ & $L$ & certified subspace dim.\ & $\sqrt{\kappa\kappa'}$ \\",
          body,
          r"\textbf{Rotation invariance of near-inertial ocean currents.} Hourly velocities"
          r" of " + f"{r0['drifters']}" + r" drogued drifters of the Global Drifter Program"
          r" (North Pacific, 30--45$^\circ$N, 180--140$^\circ$W, June--September"
          r" 2016--2019; " + f"{r0['N']:,}".replace(",", "{,}") + r" state--derivative pairs),"
          r" near-inertial band isolated by removing a running mean (49 hours unless stated),"
          r" time in units of the local inertial period over $2\pi$ and velocity in units of"
          r" its root-mean-square. $U$ and $L$ bound the defect of the rotation generator,"
          r" fixed in advance by the physics; the target measure is the standard Gaussian in"
          r" the normalised velocity. The cluster-robust region treats each drifter as a cluster;"
          r" the Newey--West region allows serial correlation within each drifter's record. The"
          r" certified subspace dimension is that of the certified plug-in subspace at"
          r" $\delta = 0.05$. In the degree-1 class every fitted field $Ax + b$ commutes with"
          r" the dilation about its fixed point and with itself, so two plug-in defects are"
          r" zero whatever the data. These generators come first in the nested plug-in"
          r" subspaces, and in the degree-1 rows the column counts how many of them the"
          r" region is narrow enough to certify (" + lin_txt + r" here). The column is"
          r" therefore not evidence about the rotation, which is judged by $U$ and $L$, and where it is"
          r" positive while $U > \delta$ the certified subspace does not contain the"
          r" rotation." + cl_txt +
          r" The last rows bound the rotation defect simultaneously over"
          r" the classes of degree 1 to 3, assuming the largest contains the truth. Velocities"
          r" carry measurement error (median standard error "
          + f"{d['median_velocity_error']:.3f}" + r" m\,s$^{-1}$ against a root-mean-square of "
          + f"{d['rms_velocity']:.3f}" + r" m\,s$^{-1}$), which the guarantee does not cover.",
          "stab:drifters", size=r"\footnotesize")


def dependence():
    d = load("exp15_dependence")
    e = pd.DataFrame(d["rows"])
    est = ("ols", "hc0", "cluster", "hac")
    pretty = {"hopf": "Hopf", "rigid_asym": "asymmetric rigid body"}
    assert (e[[f"fc_{c}" for c in est]] == 0).all().all()
    body = []
    for name, g in e.groupby("system", sort=False):
        for k, (_, r) in enumerate(g.sort_values(["phi", "G"]).iterrows()):
            cells = []
            for pre in ("reg", "cert"):
                for c in est:
                    ok = r[f"identified_{c}"] == 1.0
                    assert ok or r[f"identified_{c}"] == 0.0
                    cells.append(f"{r[f'{pre}_{c}']:.3f}" if ok else "---")
            nm = f"{pretty[name]} ($Q = {int(r.Q)}$)" if k == 0 else ""
            body.append(f"{nm} & {r.phi:g} & {int(r.G)} & " + " & ".join(cells) + r" \\")
        body.append(r"\addlinespace")
    body.pop()
    hdr = r"OLS & White & cluster & N--W"
    table("dependence", "lrrrrrrrrrr",
          r"system & $\phi$ & trajectories & \multicolumn{4}{c}{region coverage} & \multicolumn{4}{c}{certificate coverage} \\"
          "\n" r"\cmidrule(lr){4-7}\cmidrule(lr){8-11}" "\n"
          r" & & & " + hdr + " & " + hdr + r" \\",
          body,
          r"\textbf{Serially correlated errors along trajectories.} Errors on the right-hand"
          r" side follow a stationary AR(1) process with coefficient $\phi$ along each"
          r" trajectory, with exact states and noise level " + f"{d['sigma_rel']:g}" + r";"
          r" each trajectory starts from a uniform initial condition in the box and runs for "
          + f"{int(round(d['t_segment'] / d['dt']))}" + r" steps of " + f"{d['dt']:g}" + r", only"
          r" its part inside the box is kept, and the process restarts in each contiguous"
          r" part. Regions: OLS (exact under independence), White, cluster-robust over"
          r" trajectories (jackknife form) and Newey--West (N--W) within trajectories;"
          r" nominal coverage $0.95$, " + f"{d['trials']}" + r" replications per row. Region"
          r" coverage is the fraction of replications in which the coefficient region contains"
          r" the true coefficients; certificate coverage the fraction in which the interval"
          r" contains the true defect of every plug-in direction. ``---'': with fewer"
          r" trajectories than coefficients the cluster-robust region is not identified and"
          r" certifies nothing. No region gave a false certification in any setting.",
          "stab:dependence", size=r"\footnotesize")


def main():
    calibration()
    robustness()
    for fn, need in ((trajectory, "exp2_coverage"), (qsweep, "exp4_resolution"),
                     (scaling, "exp11_scaling"), (closure, "exp17_units_closure"),
                     (dictionary, "exp16_dictionary"), (units, "exp17_units_closure"),
                     (selection, "exp14_selection"), (drifters, "exp19_drifters"),
                     (dependence, "exp15_dependence")):
        if have(need):
            fn()
        else:
            print(f"skipping {fn.__name__}: {need} missing")
    print("wrote LaTeX tables to", OUT)


if __name__ == "__main__":
    main()
