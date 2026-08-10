"""Build every figure in the paper from the saved experiment results."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = Path(__file__).resolve().parent
ALPHA, DELTA = 0.05, 0.05

# A colour-blind-safe palette (Okabe--Ito), used consistently across figures.
C = {"sym": "#0072B2", "naive": "#D55E00", "gap": "#CC79A7", "rank": "#E69F00",
     "split": "#009E73", "weyl": "#56B4E9", "grey": "#7F7F7F", "boot": "#8C564B"}

plt.rcParams.update({
    "figure.dpi": 160, "savefig.dpi": 300, "font.size": 9,
    "axes.titlesize": 9.5, "axes.labelsize": 9, "legend.fontsize": 7.6,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.grid": True,
    "grid.alpha": 0.25, "grid.linewidth": 0.5, "axes.spines.top": False,
    "axes.spines.right": False, "legend.frameon": False,
    "lines.linewidth": 1.6, "lines.markersize": 5,
    "font.family": "serif", "mathtext.fontset": "cm",
})


def load(name):
    return json.loads((RES / f"{name}.json").read_text())


# ---------------------------------------------------------------- figure 1
def figure1():
    d1 = load("exp1_calibration")
    df = pd.DataFrame(d1["rows"])
    sym = df[df.d_true > 0]
    fig, ax = plt.subplots(1, 3, figsize=(11.4, 3.3))

    # (a) worst-case error vs detection
    a = ax[0]
    xs = [sym[f"det_tau{t}"].mean() for t in d1["taus"]]
    ys = [df[f"fc_tau{t}"].max() for t in d1["taus"]]
    a.plot(xs, ys, "-o", color=C["naive"], label="fixed threshold (swept)", zorder=3)
    for t, x, y in zip(d1["taus"], xs, ys):
        if t in (0.005, 0.1, 0.4):
            a.annotate(rf"$\tau={t}$", (x, y), textcoords="offset points",
                       xytext=(-6, 10), fontsize=6.5, color=C["naive"], ha="right")
    pts = [("eigengap", sym.det_gap.mean(), df.fc_gap.max(), C["gap"], "s"),
           ("nullspace rank test", sym.det_rank.mean(), df.fc_rank.max(), C["rank"], "^"),
           ("split significance test", sym.det_split.mean(), df.fc_split.max(), C["split"], "v"),
           ("Weyl perturbation bound", sym.det_weyl.mean(), df.fc_weyl.max(), C["weyl"], "D"),
           ("SymCert (ours)", sym.det_sym.mean(), df.fc_sym.max(), C["sym"], "*")]
    for lab, x, y, c, m in pts:
        a.plot([x], [y], m, color=c, label=lab, markersize=9 if m == "*" else 5.5,
               zorder=4, clip_on=False)
    a.axhline(ALPHA, ls="--", color=C["grey"], lw=1)
    a.text(0.005, ALPHA + 0.045, r"nominal level $\alpha=0.05$", fontsize=7,
           color=C["grey"], ha="left")
    a.set_xlabel("detection rate: found the whole true algebra")
    a.set_ylabel("worst-case false-certification rate\n(max over 240 settings)")
    a.set_title("(a) No fixed threshold controls the error", loc="left")
    a.set_xlim(-0.02, 1.05); a.set_ylim(-0.05, 1.08)
    a.legend(loc="lower right", bbox_to_anchor=(1.0, 0.06))

    # (b) coverage: kappa relative to its critical value
    d2 = load("exp2_coverage")
    e2 = pd.DataFrame(d2["rows"])
    b = ax[1]
    sysmark = {"vanderpol": "o", "hopf": "s", "rigid_sym": "^",
               "lotka_volterra": "v", "rigid_asym": "D"}
    for name, m in sysmark.items():
        sub = e2[e2.system == name].sort_values("kappa")
        x = sub.kappa / sub.bound_critical
        b.plot(x, sub.fc_naive_data, "-" + m, color=C["naive"], alpha=0.85,
               markersize=4.2, label="_nolegend_")
        b.plot(x, sub.fc_sym, ":" + m, color=C["sym"], alpha=0.85,
               markersize=4.2, label="_nolegend_")
    b.axvspan(1e-4, 1.0, color=C["grey"], alpha=0.10, lw=0)
    b.axvline(1.0, color=C["grey"], lw=1.0, ls="--")
    b.text(0.03, 0.30, "the coverage bound\nproves no spurious\nsymmetry is possible\nin this region",
           transform=b.transAxes, fontsize=6.8, color="#444444")
    b.plot([], [], "-o", color=C["naive"], label="nullspace on the observed data")
    b.plot([], [], ":s", color=C["sym"], label="SymCert (ours)")
    b.set_xscale("log")
    b.set_xlabel("coverage budget spent,  "
                 r"$\sqrt{\kappa\kappa'} \,/\, (\rho^\star/\delta)$")
    b.set_ylabel("false-certification rate")
    b.set_title("(b) Incomplete coverage manufactures symmetry", loc="left")
    b.set_ylim(-0.04, 1.12); b.set_xlim(2e-2, 3e2)
    b.legend(loc="upper left", bbox_to_anchor=(0.03, 1.0))

    # (c) certified tolerance vs N against the Le Cam limit
    d4 = load("exp4_resolution")
    e4 = pd.DataFrame(d4["rows"])
    e4 = e4[e4.eps == 0.0]
    c = ax[2]
    cols = {0.03: "#1b7837", 0.1: "#762a83", 0.3: "#b35806"}
    for lvl, col in cols.items():
        sub = e4[e4.sigma_rel == lvl].sort_values("N")
        c.plot(sub.N, sub.upper_med, "-o", color=col, markersize=3.6)
        c.plot(sub.N, sub.lecam, "--", color=col, alpha=0.8)
        c.annotate(f"noise {lvl}", (sub.N.iloc[-1], sub.upper_med.iloc[-1]),
                   textcoords="offset points", xytext=(5, 0), fontsize=6.8,
                   color=col, va="center")
    c.plot([], [], "-o", color="k", markersize=3.6, label="SymCert (achieved)")
    c.plot([], [], "--", color="k", label="Le Cam limit (any procedure)")
    c.set_xscale("log"); c.set_yscale("log")
    c.set_xlim(80, 3.2e4)
    c.set_xlabel("sample size $N$")
    c.set_ylabel(r"certifiable tolerance $\delta$")
    c.set_title("(c) The limit of what anyone can certify", loc="left")
    c.legend(loc="lower left", fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "fig1_headline.pdf", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- figure 2
def figure2():
    d1, d2, d3 = load("exp1_calibration"), load("exp2_coverage"), load("exp3_modelclass")
    df1 = pd.DataFrame(d1["rows"])
    e2 = pd.DataFrame(d2["rows"])
    tr = pd.DataFrame(d3["truncation"])
    sp = pd.DataFrame(d3["sparsity"])

    # four mechanisms, worst case within each study
    tight = e2[e2.radius <= 0.3]
    trunc = tr[tr.fit_degree < tr.true_degree]
    spars = sp[(sp.eps >= 0.1) & (sp.threshold == 0.25)]
    groups = [
        ("noise and\nfinite samples", df1[f"fc_tau{DELTA}"].max(), df1.fc_sym.max(), None),
        ("incomplete\ncoverage", tight.fc_naive_data.max(), tight.fc_sym.max(), None),
        ("truncated\nmodel class", trunc.fc_naive.max(), trunc.fc_sym.max(),
         trunc.fc_sym_gated.max()),
        ("sparse regression\n(STLSQ)", spars.fc_stlsq.max(), 0.0, None),
    ]
    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    x = np.arange(len(groups)); w = 0.27
    gated = [g[3] if g[3] is not None else np.nan for g in groups]
    for off, vals, col, lab in ((-w, [g[1] for g in groups], C["naive"],
                                 "nullspace + threshold"),
                                (0.0, [g[2] for g in groups], C["sym"],
                                 "SymCert (ours)"),
                                (w, gated, C["split"],
                                 "SymCert + lack-of-fit gate")):
        vals = np.asarray(vals, dtype=float)
        ax.bar(x + off, np.nan_to_num(vals), w, color=col, label=lab)
        # a zero rate would be invisible as a bar, so mark it explicitly
        for i, v in enumerate(vals):
            if np.isnan(v):
                continue
            if v <= 1e-9:
                ax.plot([i + off - w / 2, i + off + w / 2], [0.012, 0.012],
                        color=col, lw=2.4, solid_capstyle="butt")
            ax.text(i + off, max(v, 0.012) + 0.03, f"{v:.2f}", ha="center",
                    fontsize=6.8, color=col)
    ax.axhline(ALPHA, ls="--", color=C["grey"], lw=1)
    ax.text(-0.48, ALPHA + 0.03, r"$\alpha = 0.05$", fontsize=7, color=C["grey"])
    ax.set_xticks(x); ax.set_xticklabels([g[0] for g in groups])
    ax.set_ylabel("worst-case false-certification rate")
    ax.set_ylim(0, 1.16)
    for i, g in enumerate(groups):
        if g[3] is None:
            ax.text(i + w, 0.06, "n/a", ha="center", fontsize=6.4,
                    color=C["grey"], rotation=90)
    ax.set_title("Where false symmetries actually come from", loc="left")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "fig2_mechanisms.pdf", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- figure 3
def figure3():
    d1, d6 = load("exp1_calibration"), load("exp6_dimension")
    df = pd.DataFrame(d1["rows"]); sym = df[df.d_true > 0]
    e6 = pd.DataFrame(d6["rows"])
    fig, ax = plt.subplots(1, 3, figsize=(11.4, 3.2))

    a = ax[0]
    for s, col in zip(d1["sigmas"], ["#1b7837", "#5aae61", "#762a83", "#b35806"]):
        sub = sym[sym.sigma_rel == s].groupby("N").det_sym.mean()
        a.plot(sub.index, sub.values, "-o", color=col, label=f"noise $={s}$")
    a.set_xscale("log"); a.set_xlabel("sample size $N$")
    a.set_ylabel("detection rate")
    a.set_title("(a) SymCert detects the true algebra\nas data accumulate", loc="left")
    a.set_ylim(-0.03, 1.12); a.legend(loc="upper left", ncol=2, columnspacing=1.0)

    b = ax[1]
    g = e6.groupby(["system", "degF", "gens"]).agg(
        Q=("Q", "first"), P=("P", "first"),
        us=("upper_sim", "mean"), up=("upper_split", "mean")).reset_index()
    b.plot(g.Q, g.us, "o", color=C["sym"], label="simultaneous (no splitting)")
    b.plot(g.Q, g.up, "s", color=C["split"], label="split sample + pointwise bound")
    for _, r in g.iterrows():
        b.plot([r.Q, r.Q], [r.us, r.up], "-", color=C["grey"], lw=0.7, alpha=0.6)
    b.set_xlabel("number of model coefficients $Q$")
    b.set_ylabel("median certified tolerance")
    b.set_xscale("log"); b.set_yscale("log")
    b.set_title("(b) Sample splitting is strictly worse", loc="left")
    b.legend()

    c = ax[2]
    gg = e6.groupby(["system", "degF", "gens"]).agg(
        P=("P", "first"), Q=("Q", "first"), ds=("detect_sim", "mean"),
        dp=("detect_split", "mean")).reset_index()
    idx = np.arange(len(gg))
    pretty = {"hopf": "Hopf", "lin_rot2": "planar rotation",
              "rigid_sym": "symmetric top", "sphere_flow3": "sphere flow"}
    lbl = [f"{pretty.get(r.system, r.system)}\n$P$={int(r.P)}, $Q$={int(r.Q)}"
           for _, r in gg.iterrows()]
    c.barh(idx - 0.2, gg.ds, 0.4, color=C["sym"], label="simultaneous")
    c.barh(idx + 0.2, gg.dp, 0.4, color=C["split"], label="split sample")
    c.set_yticks(idx); c.set_yticklabels(lbl, fontsize=6.2)
    c.set_xlabel("detection rate")
    c.set_xlim(0, 1.28)
    c.set_title("(c) ... in every configuration", loc="left")
    c.legend(loc="lower right", fontsize=7, bbox_to_anchor=(1.0, 0.02))
    fig.tight_layout()
    fig.savefig(OUT / "fig3_power.pdf", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- figure 4
def figure4():
    d8 = load("exp8_tightness")
    cov = pd.DataFrame(d8["coverage"])
    bnd = pd.DataFrame(d8["boundary"])
    fig, ax = plt.subplots(1, 2, figsize=(7.8, 3.2))

    a = ax[0]
    alphas = [float(x) for x in d8["alphas"]]
    for _, r in cov.iterrows():
        y = [r["coverage"][str(x)] for x in alphas]
        a.plot([1 - x for x in alphas], y, "-o", alpha=0.75, lw=1.1, markersize=3.4,
               color=C["sym"])
    a.plot([0.5, 1.0], [0.5, 1.0], "--", color=C["grey"], label="exactly calibrated")
    a.plot([], [], "-o", color=C["sym"], markersize=3.4,
           label="SymCert (one line per setting)")
    a.set_xlabel(r"nominal coverage $1-\alpha$")
    a.set_ylabel("observed coverage of the interval")
    a.set_title("(a) The certificate is conservative", loc="left")
    a.set_ylim(0.45, 1.02); a.legend(loc="lower right")

    b = ax[1]
    for a_lvl, col in zip(sorted(bnd.alpha.unique()), ["#1b7837", "#762a83", "#b35806"]):
        sub = bnd[bnd.alpha == a_lvl].sort_values("tau")
        b.plot(sub.true_defect / DELTA, sub.fc_direction, "-o", color=col,
               label=rf"SymCert, $\alpha={a_lvl}$")
        b.plot(sub.true_defect / DELTA, sub.fc_bootstrap, "--s", color=col, alpha=0.6,
               markersize=4, label=rf"bootstrap, $\alpha={a_lvl}$")
        b.axhline(a_lvl, color=col, lw=0.7, ls=":", alpha=0.7)
    b.set_xlabel(r"true defect / tolerance $\;\defectratio$".replace(
        r"\defectratio", r"(\rho^\star/\delta)"))
    b.set_ylabel("false-certification rate")
    b.set_title("(b) ... even at the hardest boundary", loc="left")
    b.set_xscale("log")
    b.legend(fontsize=6.6, ncol=1)
    fig.tight_layout()
    fig.savefig(OUT / "fig4_tightness.pdf", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- figure 5
def figure5():
    d7 = load("exp7_invariance")
    e7 = pd.DataFrame(d7["rows"])
    fig, ax = plt.subplots(1, 2, figsize=(7.8, 3.2))

    # (a) data cost of certification when the training inputs cover the disc
    a = ax[0]
    inv = e7[(e7.kind == "invariant") & (~e7.sector) & (e7.sigma_rel == 0.05)]
    inv = inv.sort_values("N")
    a.plot(inv.N, inv.dim_naive_t, "-o", color=C["naive"], label="threshold")
    a.plot(inv.N, inv.dim_sym, "-s", color=C["sym"], label="SymCert (ours)")
    a.axhline(1.0, color=C["grey"], ls=":", lw=1)
    a.text(inv.N.min() * 1.1, 1.04, "true invariance dimension", fontsize=7,
           color=C["grey"])
    a.set_xscale("log"); a.set_xlabel("training-set size $N$")
    a.set_ylabel("declared invariance dimension")
    a.set_ylim(-0.05, 1.18)
    a.set_title("(a) Certifying a real invariance\ncosts data", loc="left")
    a.legend(loc="lower right")

    # (b) training inputs restricted to a sector: invariances that are not there
    b = ax[1]
    styles = {"invariant": ("-", "o", "rotation-invariant target"),
              "anisotropic": ("--", "^", "target with no invariance")}
    for kind, (ls, m, lab) in styles.items():
        sub = e7[(e7.kind == kind) & (e7.sector) & (e7.sigma_rel == 0.05)].sort_values("N")
        b.plot(sub.N, sub.fc_naive_t, ls + m, color=C["naive"],
               label=f"threshold, {lab}")
        b.plot(sub.N, sub.fc_sym, ls + m, color=C["sym"],
               label=f"SymCert, {lab}")
    b.axhline(ALPHA, ls=":", color=C["grey"], lw=1)
    b.text(1.2e3, ALPHA + 0.03, r"$\alpha=0.05$", fontsize=7, color=C["grey"])
    b.set_xscale("log"); b.set_xlabel("training-set size $N$")
    b.set_ylabel("false-certification rate")
    b.set_ylim(-0.04, 1.0)
    b.set_title("(b) Training inputs on a $120^\\circ$ sector:\ninvariances that are not there",
                loc="left")
    b.legend(fontsize=6.4, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "fig5_invariance.pdf", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- figure 6
def figure6():
    d9 = load("exp9_modelerror")
    e9 = pd.DataFrame(d9["rows"])
    fig, ax = plt.subplots(1, 2, figsize=(7.8, 3.2), sharey=False)
    for k, (sysname, a) in enumerate(zip(["rot_nonpoly", "pendulum"], ax)):
        sub = e9[e9.system == sysname].sort_values("degree")
        a.plot(sub.degree, sub.upper_no_model_error, "-o", color=C["sym"],
               label="assuming no model error")
        a.plot(sub.degree, sub.upper_l2, "-s", color=C["split"],
               label=r"paying for $\|h\|_{L^2}$")
        a.plot(sub.degree, sub.upper_sup_norm, "-^", color=C["naive"],
               label=r"paying for $\sup|h|$ and $\sup\|Dh\|$")
        a.plot(sub.degree, sub.true_defect, "k--", label="true defect")
        a.axhline(DELTA, color=C["grey"], ls=":", lw=1.2)
        a.text(sub.degree.min(), DELTA * 1.15, r"tolerance $\delta=0.05$",
               fontsize=6.8, color=C["grey"])
        a.set_yscale("log"); a.set_xlabel("degree of the fitted polynomial class")
        a.set_ylabel("certified tolerance")
        a.set_title(("(a) exact rotational symmetry" if k == 0
                     else "(b) no continuous symmetry")
                    + f"\n({sysname.replace('_', ' ')}, not polynomial)", loc="left")
        a.set_xticks(sub.degree)
        if k == 0:
            a.legend(fontsize=6.6, loc="lower left", frameon=True,
                     framealpha=0.92, edgecolor="none")
    fig.tight_layout()
    fig.savefig(OUT / "fig6_modelerror.pdf", bbox_inches="tight")
    plt.close(fig)




# ------------------------------------------------------- figure 0 (example)
def figure0():
    """What the method actually outputs, on one system at one setting."""
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from hsd import DefectProblem, fit_ols, get_system, simulate_design
    from hsd.certify import certify_direction
    fig, ax = plt.subplots(1, 2, figsize=(7.8, 3.2), sharey=True)
    for k, (name, degF, N, srel, title) in enumerate(
            [("hopf", 3, 3200, 0.03, "Hopf normal form: one true symmetry"),
             ("vanderpol", 3, 3200, 0.03, "Van der Pol: none")]):
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
        idx = np.arange(prob.P)
        lo, hi, tru = map(np.asarray, (lo, hi, tru))
        a = ax[k]
        a.axhspan(1e-4, DELTA, color=C["sym"], alpha=0.07, lw=0)
        a.errorbar(idx, 0.5 * (lo + hi), yerr=[0.5 * (hi - lo), 0.5 * (hi - lo)],
                   fmt="none", ecolor=C["sym"], elinewidth=2.4, capsize=4,
                   capthick=1.6,
                   label="95% certificate interval" if k == 0 else None)
        a.plot(idx, rho_hat, "o", color=C["naive"], markersize=5.5,
               label="plug-in defect (what is reported today)" if k == 0 else None)
        a.plot(idx, tru, "kx", markersize=7, markeredgewidth=1.6,
               label="true defect" if k == 0 else None)
        a.axhline(DELTA, ls="--", color=C["grey"], lw=1)
        if k == 0:
            a.text(0.05, DELTA * 1.3, r"tolerance $\delta = 0.05$", fontsize=7,
                   color=C["grey"])
        a.set_yscale("log")
        a.set_ylim(3e-4, 4.0)
        a.set_xticks(idx)
        a.set_xlabel("candidate generator, ordered by plug-in defect")
        a.set_title(f"({'ab'[k]}) {title}", loc="left")
        n_cert = int(np.sum(hi <= DELTA))
        a.annotate(f"{n_cert} generator{'s' if n_cert != 1 else ''} certified",
                   (0.97, 0.05), xycoords="axes fraction", ha="right",
                   fontsize=8, color=C["sym"], weight="bold")
    ax[0].set_ylabel(r"relative equivariance defect $\rho$")
    ax[0].legend(loc="center right", fontsize=6.6, framealpha=0.9,
                 frameon=True, edgecolor="none")
    fig.tight_layout()
    fig.savefig(OUT / "fig0_example.pdf", bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------- figure 7 (tolerance)
def figure7():
    d13 = load("exp13_tolerance")
    df = pd.DataFrame(d13["rows"]); sym = df[df.d_true > 0]
    fig, ax = plt.subplots(1, 2, figsize=(7.8, 3.1))
    ds = d13["deltas"]
    a = ax[0]
    a.plot(ds, [df[f"taufc_{d}"].max() for d in ds], "-o", color=C["naive"],
           label="threshold matched to $\\delta$")
    a.plot(ds, [df[f"symfc_{d}"].max() for d in ds], "-s", color=C["sym"],
           label="SymCert (ours)")
    a.axhline(ALPHA, ls="--", color=C["grey"], lw=1)
    a.set_xscale("log"); a.set_xlabel(r"tolerance $\delta$")
    a.set_ylabel("worst-case false-certification rate")
    a.set_ylim(-0.04, 1.05)
    a.set_title("(a) Error control holds at every\ntolerance; a threshold's does not", loc="left")
    a.legend(loc="upper left", fontsize=7)
    b = ax[1]
    b.plot(ds, [sym[f"taudet_{d}"].mean() for d in ds], "-o", color=C["naive"],
           label="threshold matched to $\\delta$")
    b.plot(ds, [sym[f"symdet_{d}"].mean() for d in ds], "-s", color=C["sym"],
           label="SymCert (ours)")
    b.set_xscale("log"); b.set_xlabel(r"tolerance $\delta$")
    b.set_ylabel("detection rate")
    b.set_ylim(-0.04, 1.05)
    b.set_title("(b) A looser tolerance is easier\nto certify", loc="left")
    b.legend(loc="lower right", fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "fig7_tolerance.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    made = []
    for name, fn in [("fig0", figure0), ("fig1", figure1), ("fig2", figure2),
                     ("fig3", figure3), ("fig4", figure4), ("fig5", figure5),
                     ("fig6", figure6), ("fig7", figure7)]:
        try:
            fn()
            made.append(name)
        except FileNotFoundError as e:
            print(f"skipping {name}: {e}")
    print("built:", made)
