"""Build every figure of the manuscript and the Supplementary Information from
the saved experiment results.

Main text: fig1_example ... fig5_selection (Figs. 1-5).
Supplementary: figS1_dependence ... figS6_invariance (Supplementary Figs. S1-S6).
The number in each file name is the figure's number in the manuscript.

``HSD_RESULTS`` redirects the input directory (used to test the script on
smoke-test output).  Lettering follows the journal's figure guidelines: a
sans-serif face throughout and bold lower-case panel letters.

Every figure is drawn at the physical width at which the manuscript includes
it (``\\textwidth`` = 372 pt times the factor in its ``\\includegraphics``), and
is saved without cropping, so the PDF has exactly that width and the lettering
prints at its nominal size: axis labels 7.5 pt, tick labels 7 pt, legends and
annotations at least 6.5 pt.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
RES = Path(os.environ.get("HSD_RESULTS", ROOT / "results"))
OUT = Path(__file__).resolve().parent
ALPHA, DELTA = 0.05, 0.05

# Text width of the manuscript template (372 pt) in inches; each figure is drawn
# at the fraction of it given in its \includegraphics.
TW = 372.0 / 72.27
WIDTH = {"fig1_example": 0.82, "fig2_headline": 1.0, "fig3_power": 1.0,
         "fig4_mechanisms": 0.85, "fig5_selection": 1.0, "figS1_dependence": 1.0,
         "figS2_modelerror": 0.85, "figS3_eiv": 0.85, "figS4_tolerance": 0.78,
         "figS5_tightness": 1.0, "figS6_invariance": 0.78}
SMALL = 6.5          # smallest lettering used anywhere (pt)

# A colour-blind-safe palette (Okabe--Ito), used consistently across figures.
C = {"sym": "#0072B2", "naive": "#D55E00", "gap": "#CC79A7", "rank": "#E69F00",
     "split": "#009E73", "weyl": "#56B4E9", "grey": "#7F7F7F", "boot": "#8C564B",
     "sep": "#56B4E9", "gate": "#E69F00", "largest": "#009E73", "small": "#D55E00",
     "simul": "#0072B2", "splitsel": "#CC79A7"}

plt.rcParams.update({
    "figure.dpi": 160, "savefig.dpi": 300, "font.size": 7.5,
    "axes.titlesize": 7.5, "axes.labelsize": 7.5, "legend.fontsize": 6.8,
    "xtick.labelsize": 7.0, "ytick.labelsize": 7.0, "axes.grid": True,
    "grid.alpha": 0.25, "grid.linewidth": 0.5, "axes.spines.top": False,
    "axes.spines.right": False, "legend.frameon": False,
    "lines.linewidth": 1.3, "lines.markersize": 4.0,
    "axes.linewidth": 0.7, "xtick.major.width": 0.7, "ytick.major.width": 0.7,
    "xtick.major.size": 2.8, "ytick.major.size": 2.8,
    "xtick.minor.size": 1.6, "ytick.minor.size": 1.6,
    "axes.titlepad": 4.0, "axes.labelpad": 2.5,
    "legend.handlelength": 1.8, "legend.borderaxespad": 0.3,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "mathtext.fontset": "dejavusans", "pdf.fonttype": 42,
})


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


def load(name):
    return json.loads((RES / f"{name}.json").read_text())


def have(name):
    return (RES / f"{name}.json").exists()


def new_fig(name, height):
    """A figure exactly as wide as it is printed, with constrained layout."""
    return plt.figure(figsize=(WIDTH[name] * TW, height), layout="constrained")


def panel(ax, letter, title=""):
    """Bold lower-case panel letter followed by a short title."""
    ax.set_title(r"$\bf{" + letter + "}$   " + title, loc="left")


def save(fig, name):
    # No cropping: the saved page is exactly the figure size set in new_fig.
    fig.savefig(OUT / f"{name}.pdf")
    plt.close(fig)


# ------------------------------------------------------------ Fig. 1 example
def fig_example():
    """What the method outputs, on two systems at one setting."""
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from hsd import DefectProblem, fit_ols, get_system, simulate_design
    from hsd.certify import certify_direction, refute_all
    fig = new_fig("fig1_example", 2.55)
    ax = fig.subplots(1, 2, sharey=True)
    info = {}
    for k, (name, degF, N, srel, title) in enumerate(
            [("hopf", 3, 3200, 0.03, "Hopf normal form"),
             ("vanderpol", 3, 3200, 0.03, "Van der Pol")]):
        rng = np.random.default_rng(3)
        sys_ = get_system(name)
        prob = DefectProblem.build(sys_.n, degF, "affine", "box", scale=1.0)
        beta0 = sys_.beta(prob)
        Xr = rng.uniform(-1, 1, size=(200_000, sys_.n))
        rms = float(np.sqrt(np.mean(np.sum(sys_.rhs(Xr) ** 2, axis=1))))
        X, Y = simulate_design(sys_, rng, N=N, sigma=srel * rms, scale=1.0)
        fit = fit_ols(prob, X, Y)
        rho_hat, Theta = prob.spectrum(fit.beta)
        lo, hi, tru = [], [], []
        for j in range(prob.P):
            c = certify_direction(prob, Theta[:, j], fit, ALPHA)
            lo.append(c.lower); hi.append(min(c.upper, 3.0))
            tru.append(prob.rho(Theta[:, j], beta0))
        info[name] = dict(refute=refute_all(prob, fit, ALPHA),
                          n_cert=int(np.sum(np.asarray(hi) <= DELTA)))
        idx = np.arange(prob.P)
        lo, hi, tru = map(np.asarray, (lo, hi, tru))
        a = ax[k]
        a.axhspan(1e-4, DELTA, color=C["sym"], alpha=0.07, lw=0)
        a.errorbar(idx, 0.5 * (lo + hi), yerr=[0.5 * (hi - lo), 0.5 * (hi - lo)],
                   fmt="none", ecolor=C["sym"], elinewidth=2.0, capsize=3, capthick=1.3,
                   label="95% certificate interval")
        a.plot(idx, rho_hat, "o", color=C["naive"], markersize=4.2, label="Plug-in defect")
        a.plot(idx, tru, "kx", markersize=5.5, markeredgewidth=1.3, label="True defect")
        a.axhline(DELTA, ls="--", color=C["grey"], lw=0.9)
        if k == 0:
            a.text(-0.3, DELTA * 1.35, r"Tolerance $\delta = 0.05$", fontsize=SMALL,
                   color="#555555")
        a.set_yscale("log")
        a.set_ylim(3e-4, 4.0)
        a.set_xlim(-0.6, prob.P - 0.4)
        a.set_xticks(idx)
        a.set_xticklabels([str(i + 1) for i in idx])
        panel(a, "ab"[k], title)
        n_cert = info[name]["n_cert"]
        a.annotate(f"{n_cert} generator{'s' if n_cert != 1 else ''}\ncertified",
                   (0.97, 0.05), xycoords="axes fraction", ha="right", va="bottom",
                   fontsize=7.0, color=C["sym"], weight="bold")
    ax[0].set_ylabel(r"Relative equivariance defect $\rho$")
    fig.supxlabel("Candidate generator, ordered by plug-in defect", fontsize=7.5)
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, loc="outside upper center", ncol=3, columnspacing=1.2)
    save(fig, "fig1_example")
    (OUT / "fig1_example_numbers.json").write_text(json.dumps(info, indent=1))


# ----------------------------------------------------------- Fig. 2 headline
def fig_headline():
    d1 = load("exp1_calibration")
    df = pd.DataFrame(d1["rows"])
    sym = df[df.d_true > 0]
    fig = new_fig("fig2_headline", 4.85)
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.25])
    top = gs[0].subgridspec(1, 2, width_ratios=[1.75, 1.0])
    bot = gs[1].subgridspec(1, 2)
    a = fig.add_subplot(top[0])
    la = fig.add_subplot(top[1]); la.axis("off")
    b = fig.add_subplot(bot[0]); c = fig.add_subplot(bot[1])

    xs = [sym[f"rec_tau{t}"].mean() for t in d1["taus"]]
    ys = [df[f"fc_tau{t}"].max() for t in d1["taus"]]
    a.plot(xs, ys, "-o", color=C["naive"], label="Fixed threshold (swept)", zorder=5,
           markersize=4.0)
    # Each label is placed next to its own point: below the threshold curve for
    # tau = 0.005 and 0.1, below and to the right for tau = 0.4.
    offsets = {0.005: ((0, -10), "center"), 0.1: ((0, -10), "center"),
               0.4: ((-2, -11), "left")}
    for t, x, y in zip(d1["taus"], xs, ys):
        if t in offsets:
            (dx, dy), ha = offsets[t]
            a.annotate(rf"$\tau={t}$", (x, y), textcoords="offset points",
                       xytext=(dx, dy), fontsize=SMALL, color=C["naive"], ha=ha,
                       va="center")
    # Hollow baseline markers, so a threshold point that coincides with one
    # (the rank test lies on the tau = 0.4 point) stays visible.
    pts = [("Eigengap", "gap", C["gap"], "s", True),
           ("Nullspace rank test", "rank", C["rank"], "^", False),
           ("Split significance test", "split", C["split"], "v", True),
           ("Weyl perturbation bound", "weyl", C["weyl"], "D", True),
           ("SymCert", "sym", C["sym"], "*", True),
           ("SymCert + sample splitting", "symp", C["sym"], "P", True)]
    for lab, key, col, m, filled in pts:
        if f"rec_{key}" not in sym:
            continue
        a.plot([sym[f"rec_{key}"].mean()], [df[f"fc_{key}"].max()], m, color=col, label=lab,
               markersize=8.5 if m == "*" else (7.5 if not filled else 5.0),
               mfc=col if filled else "none", mew=1.2, zorder=4, clip_on=False)
    a.axhline(ALPHA, ls="--", color=C["grey"], lw=0.9)
    a.text(0.005, ALPHA + 0.045, r"Nominal level $\alpha=0.05$", fontsize=SMALL,
           color="#555555", ha="left")
    a.set_xlabel("Recovery rate (true algebra found)")
    a.set_ylabel(f"Worst-case false-certification\nrate (max over {len(df)} settings)")
    panel(a, "a", "Calibration and recovery")
    a.set_xlim(-0.03, 1.05); a.set_ylim(-0.05, 1.1)
    h, l = a.get_legend_handles_labels()
    la.legend(h, l, loc="center left", borderaxespad=0.0, handlelength=1.6)

    d2 = load("exp2_coverage")
    e2 = pd.DataFrame(d2["rows"])
    sysmark = {"vanderpol": ("o", "Van der Pol"), "hopf": ("s", "Hopf"),
               "rigid_sym": ("^", "Symmetric top"),
               "lotka_volterra": ("v", "Lotka-Volterra"),
               "rigid_asym": ("D", "Asymmetric rigid body")}
    for name, (m, _) in sysmark.items():
        sub = e2[e2.system == name].sort_values("kappa")
        x = sub.kappa / sub.bound_critical
        b.plot(x, sub.fc_naive_data, "-" + m, color=C["naive"], alpha=0.85, markersize=3.4)
        b.plot(x, sub.fc_sym, ":" + m, color=C["sym"], alpha=0.85, markersize=3.4)
    b.axvspan(1e-4, 1.0, color=C["grey"], alpha=0.10, lw=0)
    b.axvline(1.0, color=C["grey"], lw=0.9, ls="--")
    b.text(0.03, 0.95, "Bound excludes\nextra\ngenerators\n(population\nlevel)",
           transform=b.transAxes, fontsize=SMALL, color="#444444", va="top")
    b.set_xscale("log")
    b.set_xlabel(r"Extrapolation bound / threshold, $\sqrt{\kappa\kappa'}\,/\,(\rho^\star/\delta)$")
    b.set_ylabel("False-certification rate")
    panel(b, "b", "Incomplete coverage")
    b.set_ylim(-0.04, 1.08); b.set_xlim(2e-2, 4e2)
    # Method key (line styles only) and system key (markers only), below the axes.
    hs = [Line2D([], [], color=C["naive"], ls="-"), Line2D([], [], color=C["sym"], ls=":")]
    ls_ = ["Nullspace on the data", "SymCert"]
    hs += [Line2D([], [], color="0.35", ls="none", marker=m, markersize=3.6)
           for m, _ in sysmark.values()]
    ls_ += [lab for _, lab in sysmark.values()]
    b.legend(hs, ls_, loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=2,
             columnspacing=1.0, handlelength=1.8, borderaxespad=0.0)

    e4 = pd.DataFrame(load("exp4_resolution")["rows"])
    e4 = e4[e4.eps == 0.0]
    cols = {0.03: "#1b7837", 0.1: "#762a83", 0.3: "#b35806"}
    for lvl, col in cols.items():
        sub = e4[e4.sigma_rel == lvl].sort_values("N")
        c.plot(sub.N, sub.upper_med, "-o", color=col, markersize=3.0)
        c.plot(sub.N, sub.lecam, "--", color=col, alpha=0.8)
        c.annotate(f"Noise {lvl}", (sub.N.iloc[-1], sub.upper_med.iloc[-1]),
                   textcoords="offset points", xytext=(4, 0), fontsize=SMALL,
                   color=col, va="center")
    c.set_xscale("log"); c.set_yscale("log")
    c.set_xlim(70, 9e4); c.set_ylim(5e-4, 8.0)
    c.set_xlabel("Sample size $N$")
    c.set_ylabel(r"Certifiable tolerance $\delta$")
    panel(c, "c", "Resolution at a fixed class")
    hc = [Line2D([], [], color="k", ls="-", marker="o", markersize=3.0),
          Line2D([], [], color="k", ls="--")]
    c.legend(hc, ["SymCert (achieved)", "Le Cam bound (any procedure)"],
             loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=1, borderaxespad=0.0)
    save(fig, "fig2_headline")


# --------------------------------------------------------- Fig. 4 mechanisms
def fig_mechanisms():
    df1 = pd.DataFrame(load("exp1_calibration")["rows"])
    e2 = pd.DataFrame(load("exp2_coverage")["rows"])
    d3 = load("exp3_modelclass")
    tr = pd.DataFrame(d3["truncation"])
    sp = pd.DataFrame(d3["sparsity"])
    tight = e2[e2.radius <= 0.3]
    trunc = tr[tr.fit_degree < tr.true_degree]
    spars = sp[(sp.eps >= 0.1) & (sp.threshold == 0.25)]
    # SymCert in the sparse-regression group certifies from the unpruned
    # least-squares fit.  exp3 stores its mean certified dimension per setting,
    # not a false-certification count; since the dimension is a non-negative
    # integer, the rate of certifying anything, and hence of certifying falsely,
    # is at most that mean.  Every setting plotted has mean 0, so the rate is 0.
    fc_sym_sparse = float(spars.dim_symcert.max())
    groups = [
        ("Derivative\nnoise, finite\nsamples", df1[f"fc_tau{DELTA}"].max(), df1.fc_sym.max()),
        ("Incomplete\ncoverage", tight.fc_naive_data.max(), tight.fc_sym.max()),
        ("Truncated\nmodel class", trunc.fc_naive.max(), trunc.fc_sym.max()),
        ("Sparse\nregression\n(STLSQ)", spars.fc_stlsq.max(), fc_sym_sparse),
    ]
    if have("exp18_eiv"):
        e18 = pd.DataFrame(load("exp18_eiv")["rows"])
        g = e18[e18.system == "hard_duffing"]
        groups.append(("Noisy states\n(errors in\nvariables)",
                       g.fc_threshold.max() if "fc_threshold" in g else np.nan, g.fc.max()))
    fig = new_fig("fig4_mechanisms", 2.75)
    ax = fig.subplots()
    x = np.arange(len(groups)); w = 0.36
    labs = ("Threshold rule on the fitted nullspace\n"
            "(sparse regression: on the STLSQ fit)",
            "SymCert\n(sparse regression: on the unpruned least-squares fit)")
    for off, vals, col, lab in ((-w / 2, [g[1] for g in groups], C["naive"], labs[0]),
                                (w / 2, [g[2] for g in groups], C["sym"], labs[1])):
        vals = np.asarray(vals, dtype=float)
        ax.bar(x + off, np.nan_to_num(vals), w, color=col, label=lab)
        for i, v in enumerate(vals):
            if np.isnan(v):
                continue
            if v <= 1e-9:
                ax.plot([i + off - w / 2, i + off + w / 2], [0.012, 0.012],
                        color=col, lw=2.2, solid_capstyle="butt")
            # Labels of small bars sit above the alpha line so the two never touch.
            ax.text(i + off, max(v + 0.025, ALPHA + 0.03), f"{v:.2f}", ha="center",
                    va="bottom", fontsize=SMALL, color=col)
    ax.axhline(ALPHA, ls="--", color=C["grey"], lw=0.9)
    ax.text(-0.62, ALPHA + 0.012, r"$\alpha$", fontsize=7.0, color=C["grey"], ha="left")
    ax.set_xlim(-0.65, len(groups) - 0.4)
    ax.set_xticks(x); ax.set_xticklabels([g[0] for g in groups], fontsize=7.0)
    ax.tick_params(axis="x", length=0)
    ax.set_ylabel("Worst-case false-certification rate")
    ax.set_ylim(0, 1.13)
    ax.grid(axis="x", visible=False)
    fig.legend(loc="outside upper center", ncol=2, columnspacing=1.0, handlelength=1.4)
    save(fig, "fig4_mechanisms")


# ---------------------------------------------------------- Fig. 5 selection
def fig_selection():
    e = pd.DataFrame(load("exp14_selection")["rows"])
    # The settings whose rotation defect equals the tolerance exactly are left
    # out of the plot (they are listed in the Supplementary table).
    e = e[e.defect_multiple != 1.0]
    Nshow = 3200 if (e.N == 3200).any() else int(e.N.max())
    fig = new_fig("fig5_selection", 4.6)
    gs = fig.add_gridspec(2, 2)
    ax = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 0])]
    lg = fig.add_subplot(gs[1, 1]); lg.axis("off")
    procs = [("small", "Degree 3 (misspecified)", C["small"], "o", "-"),
             ("gate", "Lack-of-fit gate", C["gate"], "s", "-"),
             ("split", "Gate on a split", C["splitsel"], "v", "-"),
             ("largest", "Largest class", C["largest"], "^", "-"),
             ("simultaneous", "Simultaneous over classes", C["simul"], "*", "-")]
    for j, k in enumerate(sorted(e.k.unique())):
        a = ax[j]
        g = e[(e.k == k) & (e.N == Nshow)].sort_values("defect_multiple")
        x = g.true_defect / DELTA
        for key, lab, col, m, ls in procs:
            a.plot(x, g[f"fc_{key}"], ls + m, color=col, label=lab,
                   markersize=6.5 if m == "*" else 3.6)
        a.plot(x, g.lof_reject, ":", color="k", lw=1.1, label="Lack-of-fit rejection")
        a.plot(x, [lof_theory(r) for _, r in g.iterrows()], "x", color="k", markersize=3.8,
               label="Rejection, theory")
        inv = float(g.invisible_defect_05.iloc[0]) / DELTA
        a.axvline(inv, color=C["grey"], lw=0.9, ls="-.")
        right = inv < 1.5
        a.text(inv + (0.05 if right else -0.05), 1.115, "invisible at\npower 0.5",
               fontsize=SMALL, color="#555555", va="center",
               ha="left" if right else "right", linespacing=0.95)
        a.axvspan(-1, 1.0, color=C["grey"], alpha=0.08, lw=0)
        a.axhline(ALPHA, ls="--", color=C["grey"], lw=0.8)
        a.set_xlabel(r"True rotation defect, in units of $\delta$")
        a.set_ylabel("Rate")
        a.set_xlim(-0.15, 3.15)
        a.set_ylim(-0.03, 1.2)
        a.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        panel(a, "ab"[j], rf"Omitted term of degree {int(k)}, $N = {Nshow}$")
    c = ax[2]
    z = e[e.defect_multiple == 0.0]
    for (k, ls) in zip(sorted(z.k.unique()), ("-", "--")):
        g = z[z.k == k].sort_values("N")
        for key, lab, col, m, _ in procs[1:]:
            c.plot(g.N, g[f"dim_{key}"], ls + m, color=col,
                   markersize=6.5 if m == "*" else 3.6)
    c.set_xscale("log")
    c.set_xlabel("Sample size $N$")
    c.set_ylabel("Mean dimension of the certified\nsubspace (truth symmetric)")
    c.set_ylim(-0.03, 1.3)
    c.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    proxies = [Line2D([], [], color="0.3", ls="-"), Line2D([], [], color="0.3", ls="--")]
    # The top-left corner (N below about 3000, above the gate's peak of 0.86)
    # holds no data, so the key sits there.
    c.legend(proxies, ["Omitted degree 5", "Omitted degree 7"], loc="upper left",
             handlelength=2.2, borderaxespad=0.2)
    panel(c, "c", "The price of validity")
    handles, labels = ax[0].get_legend_handles_labels()
    lg.legend(handles, labels, loc="center", borderaxespad=0.0, handlelength=2.0)
    save(fig, "fig5_selection")


# -------------------------------------------------------------- Fig. 3 power
def fig_power():
    d1, d6 = load("exp1_calibration"), load("exp6_dimension")
    df = pd.DataFrame(d1["rows"]); sym = df[df.d_true > 0]
    e6 = pd.DataFrame(d6["rows"])
    fig = new_fig("fig3_power", 4.85)
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.75])
    top = gs[0].subgridspec(1, 2, width_ratios=[1.9, 1.0])
    bot = gs[1].subgridspec(1, 2)
    a = fig.add_subplot(top[0])
    la = fig.add_subplot(top[1]); la.axis("off")
    b = fig.add_subplot(bot[0]); c = fig.add_subplot(bot[1], sharey=b)

    hs, labs = [], []
    for s, col in zip(d1["sigmas"], ["#1b7837", "#5aae61", "#762a83", "#b35806"]):
        sub = sym[sym.sigma_rel == s].groupby("N")
        a.plot(sub.det_sym.mean().index, sub.det_sym.mean().values, "-o", color=col,
               markersize=3.4)
        a.plot(sub.rec_sym.mean().index, sub.rec_sym.mean().values, ":", color=col, lw=1.6)
        hs.append(Line2D([], [], color=col, marker="o", markersize=3.4, ls="-"))
        labs.append(f"Noise {s}")
    hs += [Line2D([], [], color="0.3", ls="-"), Line2D([], [], color="0.3", ls=":", lw=1.6)]
    labs += ["Detection", "Recovery"]
    a.set_xscale("log"); a.set_xlabel("Sample size $N$")
    a.set_ylabel("Rate on symmetric\nsystems")
    panel(a, "a", "SymCert detection and recovery")
    a.set_ylim(-0.04, 1.06)
    la.legend(hs, labs, loc="center left", borderaxespad=0.0, ncol=1)

    # Panels b and c share one row per configuration, ordered by the number of
    # model coefficients Q (then by P), so every configuration is visible.
    g = e6.groupby(["system", "degF", "gens"]).agg(
        Q=("Q", "first"), P=("P", "first"),
        us=("upper_sim", "mean"), up=("upper_split", "mean"),
        ds=("detect_sim", "mean"), dp=("detect_split", "mean")).reset_index()
    g = g.sort_values(["Q", "P", "system"]).reset_index(drop=True)
    pretty = {"hopf": "Hopf", "lin_rot2": "Planar rotation",
              "rigid_sym": "Symmetric top", "sphere_flow3": "Sphere flow"}
    lbl = [f"{pretty.get(r.system, r.system)}\n$P$ = {int(r.P)}, $Q$ = {int(r.Q)}"
           for _, r in g.iterrows()]
    idx = np.arange(len(g))
    for i, r in g.iterrows():
        b.plot([r.us, r.up], [i, i], "-", color=C["grey"], lw=0.8, alpha=0.7, zorder=1)
    b.plot(g.us, idx, "o", color=C["sym"], markersize=4.2, zorder=3)
    b.plot(g.up, idx, "s", color=C["split"], markersize=4.0, zorder=3)
    b.set_xscale("log")
    b.set_xlabel("Certified tolerance\n(median over replications, mean over $N$)")
    b.set_yticks(idx); b.set_yticklabels(lbl, fontsize=SMALL)
    b.set_ylim(len(g) - 0.5, -0.5)
    b.set_xlim(4e-3, 0.6)
    panel(b, "b", "Splitting widens the certificate")
    c.barh(idx - 0.2, g.ds, 0.4, color=C["sym"])
    c.barh(idx + 0.2, g.dp, 0.4, color=C["split"])
    c.tick_params(axis="y", labelleft=False)
    c.set_xlabel("Rate of certifying the minimal-defect\ndirection (mean over $N$)")
    c.set_xlim(0, 1.05)
    c.grid(axis="y", visible=False); b.grid(axis="y", visible=False)
    panel(c, "c", "... and certifies less often")
    hk = [Line2D([], [], color=C["sym"], marker="o", ls="none", markersize=4.2),
          Line2D([], [], color=C["split"], marker="s", ls="none", markersize=4.0)]
    fig.legend(hk, ["Simultaneous certificate (no splitting)",
                    "Split sample + pointwise bound"],
               loc="outside lower center", ncol=2, columnspacing=1.5)
    save(fig, "fig3_power")


# --------------------------------------------------- Supplementary figures
def figS4_tolerance():
    d13 = load("exp13_tolerance")
    df = pd.DataFrame(d13["rows"]); sym = df[df.d_true > 0]
    fig = new_fig("figS4_tolerance", 2.35)
    ax = fig.subplots(1, 2)
    ds = d13["deltas"]
    a = ax[0]
    a.plot(ds, [df[f"taufc_{d}"].max() for d in ds], "-o", color=C["naive"],
           label=r"Threshold matched to $\delta$")
    a.plot(ds, [df[f"symfc_{d}"].max() for d in ds], "-s", color=C["sym"], label="SymCert")
    a.axhline(ALPHA, ls="--", color=C["grey"], lw=0.9)
    a.set_xscale("log"); a.set_xlabel(r"Tolerance $\delta$")
    a.set_ylabel("Worst-case false-\ncertification rate")
    a.set_ylim(-0.04, 1.05)
    panel(a, "a", r"Error control across $\delta$")
    b = ax[1]
    b.plot(ds, [sym[f"taudet_{d}"].mean() for d in ds], "-o", color=C["naive"],
           label=r"Threshold matched to $\delta$")
    b.plot(ds, [sym[f"symdet_{d}"].mean() for d in ds], "-s", color=C["sym"], label="SymCert")
    b.set_xscale("log"); b.set_xlabel(r"Tolerance $\delta$")
    b.set_ylabel("Detection rate")
    b.set_ylim(-0.04, 1.05)
    panel(b, "b", r"A looser $\delta$ is easier")
    h, l = a.get_legend_handles_labels()
    fig.legend(h, l, loc="outside upper center", ncol=2, columnspacing=1.5)
    save(fig, "figS4_tolerance")


def figS5_tightness():
    d8 = load("exp8_tightness")
    cov = pd.DataFrame(d8["coverage"])
    wid = pd.DataFrame(d8["width"])
    sm = pd.DataFrame(d8["small"])
    bd = pd.DataFrame(d8["boundary"])
    has_sel = "selected" in d8
    fig = new_fig("figS5_tightness", 5.3 if has_sel else 4.7)
    gs = fig.add_gridspec(3 if has_sel else 2, 2,
                          height_ratios=[1, 1, 1.05] if has_sel else [1, 1])
    a, b = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    c, d = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])

    alphas = [float(x) for x in d8["alphas"]]
    for _, r in cov.iterrows():
        a.plot([1 - x for x in alphas], [r["coverage"][str(x)] for x in alphas], "-o",
               alpha=0.75, lw=1.0, markersize=2.8, color=C["sym"])
        a.plot([1 - x for x in alphas], [r["region_coverage"][str(x)] for x in alphas], ":",
               alpha=0.75, lw=1.0, color=C["grey"])
    a.plot([0.5, 1.0], [0.5, 1.0], "--", color="k", lw=0.8, label="Exactly calibrated")
    a.plot([], [], "-o", color=C["sym"], markersize=2.8, label="Certificate interval")
    a.plot([], [], ":", color=C["grey"], label="Coefficient region")
    a.set_xlabel(r"Nominal coverage $1-\alpha$")
    a.set_ylabel("Observed coverage")
    # The lower limit follows the data, so no setting is clipped.
    lowest = min(min(r["region_coverage"].values()) for _, r in cov.iterrows())
    lowest = min(lowest, min(min(r["coverage"].values()) for _, r in cov.iterrows()))
    a.set_ylim(min(0.45, np.floor((lowest - 0.05) * 20) / 20), 1.03)
    a.legend(loc="lower right")
    panel(a, "a", "Coverage")

    for t, ls in zip(sorted(wid.true_defect.unique()), ("-", "--", ":")):
        g = wid[(wid.true_defect == t) & (wid.alpha == 0.05)].sort_values("N")
        for key, col in (("sharp", C["sym"]), ("separate", C["sep"]), ("boot", C["boot"])):
            b.plot(g.N, g[f"excess_{key}"], ls, color=col, marker="o", markersize=2.6)
    b.set_xscale("log"); b.set_yscale("log")
    b.set_xlabel("Sample size $N$")
    b.set_ylabel("Median upper bound\nminus plug-in")
    prox = [Line2D([], [], color=C["sym"]), Line2D([], [], color=C["sep"]),
            Line2D([], [], color=C["boot"])]
    # Extra room below the curves keeps the legend clear of the data.
    lo, hi = b.get_ylim()
    b.set_ylim(lo / 6.0, hi)
    b.legend(prox, ["Sharp certificate", "Separate bound", "Parametric bootstrap"],
             loc="lower left")
    panel(b, "b", r"Width ($\alpha = 0.05$)")

    for (N, a_lvl), g in sm.groupby(["N", "alpha"]):
        g = g.sort_values("true_defect")
        col = "#762a83" if a_lvl == 0.2 else "#1b7837"
        ls = "-" if N == sm.N.min() else "--"
        c.plot(g.true_defect, g.coverage_boot, ls, marker="s", color=col, markersize=2.8)
        c.axhline(1 - 2 * a_lvl, color=col, lw=0.8, ls=":")
    # The colours are named on their nominal lines (as in panel d); a small key
    # for the two line styles sits between the lower nominal line and the
    # curves at zero.
    for a_lvl, col in ((0.05, "#1b7837"), (0.2, "#762a83")):
        c.text(0.0, 1 - 2 * a_lvl + 0.02,
               rf"$\alpha = {a_lvl}$, nominal ${1 - 2 * a_lvl:.2f}$", fontsize=SMALL,
               color=col, ha="left", va="bottom")
    hls = [Line2D([], [], color="0.3", ls="-"), Line2D([], [], color="0.3", ls="--")]
    c.legend(hls, [rf"$N = {int(sm.N.min())}$", rf"$N = {int(sm.N.max())}$"],
             loc="lower left", bbox_to_anchor=(0.0, 0.11), borderaxespad=0.2)
    c.set_xscale("symlog", linthresh=0.002)
    c.set_xlim(-3e-4, 0.06)
    c.set_xlabel("True rotation defect")
    c.set_ylabel("Coverage of the\nbootstrap interval")
    c.set_ylim(-0.03, 1.05)
    panel(c, "c", "Bootstrap near a symmetry")

    for a_lvl, col in zip(sorted(bd.alpha.unique()), ("#1b7837", "#762a83", "#b35806")):
        g = bd[(bd.alpha == a_lvl) & (bd.tau == bd.tau.min())].sort_values("N")
        d.plot(g.N, g.fc_fix_boot, "--s", color=col, markersize=2.8)
        d.plot(g.N, g.fc_fix_sharp, "-o", color=col, markersize=2.8)
        d.axhline(a_lvl, color=col, lw=0.8, ls=":")
        d.text(g.N.min() * 0.85, a_lvl + 0.014, rf"$\alpha = {a_lvl}$", fontsize=SMALL,
               color=col, ha="left", va="bottom")
    d.set_xscale("log")
    d.set_xlabel("Sample size $N$")
    d.set_ylabel("False-certification rate")
    d.set_ylim(-0.02, 0.62)
    prox = [Line2D([], [], color="0.3", ls="-", marker="o", markersize=2.8),
            Line2D([], [], color="0.3", ls="--", marker="s", markersize=2.8)]
    # Between the alpha = 0.5 and alpha = 0.2 reference lines, clear of both.
    d.legend(prox, ["SymCert", "Bootstrap"], loc="center right", bbox_to_anchor=(1.0, 0.58))
    panel(d, "d", r"Boundary, true defect $1.02\delta$")

    if has_sel:
        # (e) Upper-bound coverage for the generator with the smallest plug-in
        # defect, selected on the same data, on four systems without symmetry.
        se = pd.DataFrame(d8["selected"])
        e1, e2 = fig.add_subplot(gs[2, 0]), fig.add_subplot(gs[2, 1])
        sysst = [("vanderpol", "Van der Pol", "#E69F00", "o"),
                 ("duffing", "Duffing", "#D55E00", "s"),
                 ("lotka_volterra", "Lotka-Volterra", "#009E73", "^"),
                 ("selkov", "Sel'kov", "#CC79A7", "D")]
        for ea, nominal in ((e1, 0.80), (e2, 0.95)):
            sub = se[np.isclose(se.nominal, nominal)]
            # Certificate: the lowest coverage over all systems and noise levels.
            cmin = sub.groupby("N").coverage_upper_sharp.min()
            ea.plot(cmin.index, cmin.values, "-", color=C["sym"], lw=3.0, alpha=0.45,
                    zorder=1, solid_capstyle="butt")
            ea.axhline(nominal, color="k", lw=0.8, ls=":", zorder=1)
            # The systems are offset slightly along N (by -0.03 to 0.03 decades)
            # so that coinciding points stay visible.
            for (key, _, col, m), jit in zip(sysst, (-0.03, -0.01, 0.01, 0.03)):
                for srel, ls in ((0.5, "-"), (0.2, "--")):
                    g = sub[(sub.system == key) & np.isclose(sub.sigma_rel, srel)]
                    g = g.sort_values("N")
                    ea.plot(g.N * 10 ** jit, g.coverage_upper_boot, ls, marker=m, color=col,
                            markersize=3.0, lw=1.0, zorder=3,
                            mfc=col if srel == 0.5 else "white")
            ea.set_xscale("log")
            ea.set_xticks(sorted(sub.N.unique()))
            ea.set_xticklabels([str(int(n)) for n in sorted(sub.N.unique())])
            ea.minorticks_off()
            ea.set_xlim(42, 470)
            ea.set_xlabel("Sample size $N$")
        e1.set_ylim(0.5, 1.03)
        e2.set_ylim(0.86, 1.008)
        e1.set_ylabel("Coverage of the\nupper bound")
        panel(e1, "e", "Data-selected generator, nominal 0.80")
        e2.set_title("Data-selected generator, nominal 0.95", loc="left")
        he = [Line2D([], [], color=col, marker=m, markersize=3.0, ls="none")
              for _, _, col, m in sysst]
        le = [lab for _, lab, _, _ in sysst]
        he += [Line2D([], [], color="0.3", ls="-", marker="o", markersize=3.0, lw=1.0),
               Line2D([], [], color="0.3", ls="--", marker="o", mfc="white",
                      markersize=3.0, lw=1.0),
               Line2D([], [], color=C["sym"], lw=3.0, alpha=0.45),
               Line2D([], [], color="k", lw=0.8, ls=":")]
        le += ["Bootstrap, noise 0.5", "Bootstrap, noise 0.2",
               "Certificate (all systems, both noise levels)", "Nominal level"]
        fig.legend(he, le, loc="outside lower center", ncol=3, columnspacing=1.2)
    save(fig, "figS5_tightness")


def figS1_dependence():
    e = pd.DataFrame(load("exp15_dependence")["rows"])
    systems = sorted(e.system.unique())
    fig = new_fig("figS1_dependence", 2.5)
    ax = fig.subplots(1, len(systems), sharey=True)
    est = [("ols", "OLS (exact)", C["naive"], "o"), ("hc0", "White", C["rank"], "s"),
           ("cluster", "Cluster-robust", C["split"], "^"), ("hac", "Newey-West", C["sym"], "D")]
    for j, s in enumerate(systems):
        a = ax[j]
        for G, ls in zip(sorted(e.G.unique()), (":", "--", "-")):
            g = e[(e.system == s) & (e.G == G)].sort_values("phi")
            for key, lab, col, m in est:
                y = g[f"reg_{key}"].where(g[f"identified_{key}"] > 0.5)
                a.plot(g.phi, y, ls + m, color=col, markersize=3.0, lw=1.1)
        a.axhline(0.95, color="k", lw=0.7, ls=":")
        a.set_xlabel(r"AR(1) coefficient $\phi$")
        a.set_ylim(-0.03, 1.03)
        nice = {"hopf": "Hopf", "rigid_asym": "Asymmetric rigid body"}.get(s, s)
        panel(a, "abc"[j], nice)
    ax[0].set_ylabel("Coverage of the coefficient region")
    hs = [Line2D([], [], color=col, marker=m, ls="-", markersize=3.0) for _, _, col, m in est]
    hs += [Line2D([], [], color="0.3", ls=ls) for ls in (":", "--", "-")]
    labs = [lab for _, lab, _, _ in est] + [f"{G} trajectories" for G in sorted(e.G.unique())]
    fig.legend(hs, labs, loc="outside right center")
    save(fig, "figS1_dependence")


def figS2_modelerror():
    e9 = pd.DataFrame(load("exp9_modelerror")["rows"])
    fig = new_fig("figS2_modelerror", 2.95)
    ax = fig.subplots(1, 2)
    for k, (sysname, a) in enumerate(zip(["rot_nonpoly", "pendulum"], ax)):
        sub = e9[e9.system == sysname].sort_values("degree")
        a.plot(sub.degree, sub.upper_no_model_error, "-o", color=C["sym"], markersize=3.4,
               label="Model error ignored (no guarantee)")
        a.plot(sub.degree, sub.upper_l2_datadriven, "-D", color=C["rank"], markersize=3.2,
               label=r"$L^2$ form, $\|h\|$ bounded from the data")
        a.plot(sub.degree, sub.upper_l2, "-s", color=C["split"], markersize=3.2,
               label=r"$L^2$ form, $\|h\|$ known (oracle)")
        a.plot(sub.degree, sub.upper_sup_norm, "-^", color=C["naive"], markersize=3.4,
               label="Supremum form, sizes known (oracle)")
        a.plot(sub.degree, sub.true_defect, "k--", label="True defect of the selected generator")
        a.axhline(DELTA, color=C["grey"], ls=":", lw=1.1,
                  label=r"Tolerance $\delta = 0.05$" if k == 0 else None)
        a.set_yscale("log"); a.set_xlabel("Fitted polynomial degree")
        panel(a, "ab"[k], "Exact rotational symmetry" if k == 0 else "No continuous symmetry")
        a.set_xticks(sub.degree)
    # Panel b spans less than a decade: plain tick labels instead of powers of ten.
    from matplotlib.ticker import FixedLocator, FormatStrFormatter, NullFormatter
    ax[1].yaxis.set_major_locator(FixedLocator([0.05, 0.1, 0.2, 0.3, 0.4]))
    ax[1].yaxis.set_major_formatter(FormatStrFormatter("%.2g"))
    ax[1].yaxis.set_minor_formatter(NullFormatter())
    ax[0].set_ylabel("Upper bound on the defect")
    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=2, columnspacing=1.0)
    save(fig, "figS2_modelerror")


def figS3_eiv():
    e = pd.DataFrame(load("exp18_eiv")["rows"])
    fig = new_fig("figS3_eiv", 4.3)
    gs = fig.add_gridspec(2, 2)
    a, b = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    c = fig.add_subplot(gs[1, 0])
    lg = fig.add_subplot(gs[1, 1]); lg.axis("off")
    d = e[e.system == "hard_duffing"]
    g = d[d.N == d.N.min()].sort_values("state_noise")
    a.plot(g.state_noise, g.pseudo_true_min_defect, "-o", color="k", markersize=3.4,
           label="Pseudo-true")
    a.axhline(g.true_min_defect.iloc[0], color="k", ls="--", lw=0.9, label="True")
    a.axhline(DELTA, color=C["grey"], ls=":", lw=1.1, label=r"Tolerance $\delta$")
    a.set_xlabel(r"State-noise s.d. $\sigma_x$")
    a.set_ylabel("Smallest defect in the class")
    a.set_ylim(0, 0.115)
    a.legend(loc="lower left")
    panel(a, "a", "Duffing: pseudo-true defect")
    # Sample sizes differ by colour and marker only; every SymCert curve is solid.
    Ns = sorted(d.N.unique())
    # Sizes chosen so that coinciding points stay distinguishable: a filled
    # disc inside an open square, with a small diamond on top.
    styles = list(zip(Ns, ("#1b7837", "#762a83", "#b35806"), ("o", "s", "D"),
                      (5.6, 6.6, 3.2)))
    for N, col, m, ms in styles:
        g = d[d.N == N].sort_values("state_noise")
        b.plot(g.state_noise, g.fc, "-", marker=m, markersize=ms, color=col,
               mfc="none" if m == "s" else col)
    if "fc_threshold" in d:
        t = d.groupby("state_noise").fc_threshold.max()
        b.plot(t.index, t.values, "-", color="0.55", lw=4, alpha=0.35, zorder=0)
    b.axhline(ALPHA, color=C["grey"], ls=":", lw=0.9)
    b.set_xlabel(r"State-noise s.d. $\sigma_x$")
    b.set_ylabel("False-certification rate")
    b.set_ylim(-0.04, 1.05)
    panel(b, "b", "Duffing (no symmetry)")
    h = e[e.system == "hopf"]
    for N, col, m, ms in styles:
        g = h[h.N == N].sort_values("state_noise")
        c.plot(g.state_noise, g.region_coverage, ":", color=col, lw=1.1)
        c.plot(g.state_noise, g.detect, "-", marker=m, markersize=ms, color=col,
               mfc="none" if m == "s" else col)
    c.set_xlabel(r"State-noise s.d. $\sigma_x$")
    c.set_ylabel("Detection (solid),\nregion coverage (dotted)")
    c.set_ylim(-0.04, 1.05)
    panel(c, "c", "Hopf (one true symmetry)")
    hs = [Line2D([], [], color=col, marker=m, markersize=ms, ls="-",
                 mfc="none" if m == "s" else col) for N, col, m, ms in styles]
    labs = [f"N = {N:,}" for N in Ns]
    hs += [Line2D([], [], color="0.55", lw=4, alpha=0.35),
           Line2D([], [], color="0.3", ls=":", lw=1.1)]
    labs += ["Threshold rule, every N (b)", "Region coverage (c),\ncolour by N"]
    lg.legend(hs, labs, loc="center", borderaxespad=0.0, title="SymCert (b), detection (c)",
              title_fontsize=6.8)
    save(fig, "figS3_eiv")


def figS6_invariance():
    e7 = pd.DataFrame(load("exp7_invariance")["rows"])
    fig = new_fig("figS6_invariance", 2.45)
    ax = fig.subplots(1, 2)
    a = ax[0]
    inv = e7[(e7.kind == "invariant") & (~e7.sector) & (e7.sigma_rel == 0.05)].sort_values("N")
    # Method by colour and marker, target by line style (as in b); the
    # threshold is drawn on top so that coinciding points both stay visible.
    a.plot(inv.N, inv.dim_sym, "-s", color=C["sym"], markersize=4.6)
    a.plot(inv.N, inv.dim_naive_t, "-o", color=C["naive"], markersize=3.4)
    a.axhline(1.0, color=C["grey"], ls=":", lw=0.9)
    a.text(inv.N.min(), 1.05, "True invariance dimension", fontsize=SMALL,
           color="#555555", va="bottom")
    a.set_xscale("log"); a.set_xlabel("Training-set size $N$")
    a.set_ylabel("Declared invariance dimension")
    a.set_ylim(-0.05, 1.2)
    panel(a, "a", "Inputs on the square")
    b = ax[1]
    for kind, ls in (("invariant", "-"), ("anisotropic", "--")):
        sub = e7[(e7.kind == kind) & (e7.sector) & (e7.sigma_rel == 0.05)].sort_values("N")
        b.plot(sub.N, sub.fc_sym, ls + "s", color=C["sym"], markersize=4.6)
        b.plot(sub.N, sub.fc_naive_t, ls + "o", color=C["naive"], markersize=3.4)
    b.axhline(ALPHA, ls=":", color=C["grey"], lw=0.9)
    b.set_xscale("log"); b.set_xlabel("Training-set size $N$")
    b.set_ylabel("False-certification rate")
    b.set_ylim(-0.04, 1.05)
    panel(b, "b", "Inputs on a 120\u00b0 sector")
    hs = [Line2D([], [], color=C["naive"], ls="none", marker="o", markersize=3.4),
          Line2D([], [], color=C["sym"], ls="none", marker="s", markersize=4.6),
          Line2D([], [], color="0.3", ls="-"), Line2D([], [], color="0.3", ls="--")]
    labs = ["Threshold", "SymCert", "Invariant target", "Non-invariant target (b)"]
    fig.legend(hs, labs, loc="outside upper center", ncol=4, columnspacing=1.0,
               handlelength=2.0)
    save(fig, "figS6_invariance")


FIGURES = [("fig1_example", fig_example, None),
           ("fig2_headline", fig_headline, "exp1_calibration"),
           ("fig3_power", fig_power, "exp6_dimension"),
           ("fig4_mechanisms", fig_mechanisms, "exp3_modelclass"),
           ("fig5_selection", fig_selection, "exp14_selection"),
           ("figS1_dependence", figS1_dependence, "exp15_dependence"),
           ("figS2_modelerror", figS2_modelerror, "exp9_modelerror"),
           ("figS3_eiv", figS3_eiv, "exp18_eiv"),
           ("figS4_tolerance", figS4_tolerance, "exp13_tolerance"),
           ("figS5_tightness", figS5_tightness, "exp8_tightness"),
           ("figS6_invariance", figS6_invariance, "exp7_invariance")]


if __name__ == "__main__":
    import sys
    only = sys.argv[1:]
    made = []
    for name, fn, need in FIGURES:
        if only and name not in only:
            continue
        if need is not None and not have(need):
            print(f"skipping {name}: {need} missing")
            continue
        fn()
        made.append(name)
    print("built:", made)
