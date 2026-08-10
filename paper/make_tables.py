"""Emit every number quoted in the paper, directly from the saved results."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
ALPHA = DELTA = 0.05
def load(n): return json.loads((RES / f"{n}.json").read_text())

out = {}
d1 = load("exp1_calibration"); df = pd.DataFrame(d1["rows"]); sym = df[df.d_true > 0]
rows = []
for t in d1["taus"]:
    rows.append((f"fixed threshold tau={t}", df[f"fc_tau{t}"].mean(), df[f"fc_tau{t}"].max(),
                 sym[f"det_tau{t}"].mean(), int((df[f"fc_tau{t}"] > ALPHA).sum())))
for lab, fc, det in [("largest eigengap", "fc_gap", "det_gap"),
                     ("nullspace rank test", "fc_rank", "det_rank"),
                     ("split significance test", "fc_split", "det_split"),
                     ("Weyl perturbation bound", "fc_weyl", "det_weyl"),
                     ("SymCert (ours)", "fc_sym", "det_sym"),
                     ("SymCert + sample splitting", "fc_symp", "det_symp")]:
    rows.append((lab, df[fc].mean(), df[fc].max(), sym[det].mean(), int((df[fc] > ALPHA).sum())))
tab1 = pd.DataFrame(rows, columns=["method", "fc_mean", "fc_worst", "detect", "n_above_alpha"])
out["table1"] = tab1.round(4).to_dict("records")
print("=== TABLE 1 (calibration) ===");print(tab1.round(3).to_string(index=False))
print("\nn_settings =", len(df), " trials/cell =", d1["trials"], " total reps =", len(df)*d1["trials"])
print("SymCert detection by N:", sym.groupby("N").det_sym.mean().round(3).to_dict())
print("SymCert detection by N at noise .01:", sym[sym.sigma_rel==0.01].groupby("N").det_sym.mean().round(3).to_dict())
print("worst tau=0.05 cells:"); print(df.nlargest(3,"fc_tau0.05")[["system","N","sigma_rel","d_true","fc_tau0.05","dim_tau0.05"]].to_string(index=False))

d2 = load("exp2_coverage"); e2 = pd.DataFrame(d2["rows"])
sys.path.insert(0, str(ROOT/"src")); from hsd import DefectProblem, get_system
COV = {"hopf":(3,1.0),"vanderpol":(3,1.0),"lotka_volterra":(2,1.0),"rigid_asym":(2,1.0),"rigid_sym":(2,1.0)}
kc = {}
for nm,(dg,sc) in COV.items():
    s=get_system(nm); p=DefectProblem.build(s.n,dg,"affine","box",scale=sc)
    r=p.spectrum(s.beta(p))[0]; above=r[r>DELTA]; kc[nm]=float((above[0]/DELTA)**2)
e2["ratio"]=[r.kappa/kc[r.system] for _,r in e2.iterrows()]
print("\n=== EXP2: FCR vs kappa/kappa_crit ===")
print("cells with ratio<1:", int((e2.ratio<1).sum()), " of which FCR>0:", int(((e2.ratio<1)&(e2.fc_naive_data>0)).sum()))
print("cells with ratio>=1:", int((e2.ratio>=1).sum()), " of which FCR>0:", int(((e2.ratio>=1)&(e2.fc_naive_data>0)).sum()))
print("SymCert FCR max over coverage cells:", e2.fc_sym.max())
print("hopf SymCert dim by radius:", e2[e2.system=="hopf"].set_index("radius").dim_sym.round(2).to_dict())
print("kappa_crit:", {k: round(v,1) for k,v in kc.items()})
print("naive-D worst FCR with radius<=0.3:", e2[e2.radius<=0.3].fc_naive_data.max())

d3 = load("exp3_modelclass"); tr=pd.DataFrame(d3["truncation"]); sp=pd.DataFrame(d3["sparsity"])
trunc = tr[tr.fit_degree < tr.true_degree]
print("\n=== EXP3 ===")
print("truncated: naive FCR max/mean:", trunc.fc_naive.max(), round(trunc.fc_naive.mean(),3),
      "| SymCert FCR max:", trunc.fc_sym.max(), "| gated FCR max:", trunc.fc_sym_gated.max(),
      "| lack-of-fit rejection min:", trunc.lof_reject.min())
ok = tr[tr.fit_degree == tr.true_degree]
print("correct degree: naive FCR max:", ok.fc_naive.max(), "SymCert FCR max:", ok.fc_sym.max(),
      "lof reject max:", ok.lof_reject.max(), "SymCert dim (hopf):", ok[ok.system=="hopf"].dim_sym.tolist())
print("STLSQ eps=0.1:"); print(sp[sp.eps==0.1][["threshold","true_rot_defect","dim_stlsq","fc_stlsq","dim_symcert"]].to_string(index=False))
print("STLSQ eps=0.2:"); print(sp[sp.eps==0.2][["threshold","true_rot_defect","fc_stlsq","dim_symcert"]].to_string(index=False))

d4 = load("exp4_resolution"); e4=pd.DataFrame(d4["rows"]); z=e4[e4.eps==0.0]
print("\n=== EXP4 ===")
for s in sorted(z.sigma_rel.unique()):
    zz=z[z.sigma_rel==s]; print(f"  noise {s}: ratio upper/lecam median {np.median(zz.upper_med/zz.lecam):.2f}",
        f"| slope log-log {np.polyfit(np.log(zz.N),np.log(zz.upper_med),1)[0]:.3f}")
print("coverage min over all exp4 cells:", e4.coverage.min())
r=e4[(e4.N==1600)&(e4.sigma_rel==0.1)][["eps","true_defect","refute_rate"]]
print(r.round(4).to_string(index=False))

d5 = load("exp5_robustness"); a5=pd.DataFrame(d5["design"]); t5=pd.DataFrame(d5["trajectory"])
print("\n=== EXP5 ===")
print("design coverage min:", a5.coverage.min(), "over", len(a5), "settings; FCR max:", a5.fc.max())
print(t5[["system","mode","sigma_rel","coverage","fc","dim"]].to_string(index=False))

d6 = load("exp6_dimension"); e6=pd.DataFrame(d6["rows"])
g=e6.groupby(["system","degF","gens"]).agg(P=("P","first"),Q=("Q","first"),
    us=("upper_sim","mean"),up=("upper_split","mean"),ds=("detect_sim","mean"),dp=("detect_split","mean")).reset_index()
g["ratio"]=g.up/g.us
print("\n=== EXP6 ===");print(g.round(4).to_string(index=False))
print("split/sim width ratio range:", round(g.ratio.min(),2), "-", round(g.ratio.max(),2),
      "| sim wins on width:", int((g.us<g.up).sum()), "/", len(g),
      "| sim wins on detection:", int((g.ds>=g.dp).sum()), "/", len(g))
h=e6[(e6.system=="hopf")&(e6.degF==3)].groupby("gens").upper_sim.mean()
print("hopf deg3 affine vs quadratic generators:", h.round(4).to_dict())
h5=e6[(e6.system=="hopf")].groupby("degF").upper_sim.mean(); print("hopf by model degree:", h5.round(4).to_dict())

if (RES/"exp7_invariance.json").exists():
    e7=pd.DataFrame(load("exp7_invariance")["rows"])
    print("\n=== EXP7 invariance ===")
    print("full-disc FCR max:", e7[~e7.sector][["fc_naive_t","fc_sym"]].max().round(3).to_dict())
    print("sector FCR max:", e7[e7.sector][["fc_naive_t","fc_sym"]].max().round(3).to_dict())
    inv=e7[(e7.kind=="invariant")&(~e7.sector)&(e7.sigma_rel==0.05)].sort_values("N")
    print("dim vs N (sigma .05):", dict(zip(inv.N, zip(inv.dim_naive_t.round(2), inv.dim_sym.round(2)))))

if (RES/"exp8_tightness.json").exists():
    d8=load("exp8_tightness"); cov=pd.DataFrame(d8["coverage"]); bnd=pd.DataFrame(d8["boundary"])
    print("\n=== EXP8 tightness ===")
    for a in d8["alphas"]:
        v=[r[str(a)] for r in cov.coverage]
        print(f"  nominal {1-a:.2f}: observed coverage min {min(v):.3f} max {max(v):.3f}")
    print(bnd[["tau","alpha","true_defect","N","fc_direction","fc_bootstrap","fc_procedure",
               "width_exact","width_bootstrap"]].round(4).to_string(index=False))
    bnd=bnd.assign(narrow=1-bnd.width_bootstrap/bnd.width_exact)
    print("bootstrap narrower by: %.1f%% - %.1f%%" % (100*bnd.narrow.min(), 100*bnd.narrow.max()))
    print("max exact FCR at boundary:", bnd.fc_direction.max(), " max bootstrap:", bnd.fc_bootstrap.max())

if (RES/"exp9_modelerror.json").exists():
    e9=pd.DataFrame(load("exp9_modelerror")["rows"])
    print("\n=== EXP9 model error ===")
    print(e9[["system","degree","Q","true_defect","plug_in","upper_no_model_error",
              "upper_l2","upper_sup_norm","eta0","eta1","eta2","rel_l2_error"]].round(5).to_string(index=False))

if (RES/"exp10_noise.json").exists():
    d10=load("exp10_noise"); b=pd.DataFrame(d10["bias"]); o=pd.DataFrame(d10["optimism"])
    print("\n=== EXP10 noise ===")
    print("frac of cells where the plug-in minimum ever fell below delta:",
          float((b.frac_below_delta>0).mean()), " max frac:", float(b.frac_below_delta.max()))
    print("median plug-in min / true min, by noise level:",
          b.groupby("sigma_rel").apply(lambda g: float(np.median(g["median"]/g.true_min_defect)),
                                       include_groups=False).round(3).to_dict())
    hi=b[(b.sigma_rel>=0.5)]
    print("at noise>=0.5:", hi[["system","N","sigma_rel","true_min_defect","median","q05","minimum",
                               "frac_below_delta"]].round(4).to_string(index=False))
    print("\noptimism gap vs N (vanderpol):")
    print(o[o.system=="vanderpol"][["N","gap_mean","gap_se","frac_positive"]].round(5).to_string(index=False))
    print("all systems gap at N=100:", o[o.N==100][["system","gap_mean","gap_se"]].round(4).to_string(index=False))

if (RES/"exp11_scaling.json").exists():
    e11=pd.DataFrame(load("exp11_scaling")["rows"])
    print("\n=== EXP11 scaling ===")
    print(e11[["n","P","Q","N","d_true","detect","fc","upper_med","seconds_per_certificate"]].round(4).to_string(index=False))
    print("max FCR over all cells:", e11.fc.max())

if (RES/"exp12_realdata.json").exists():
    d12=load("exp12_realdata")
    print("\n=== EXP12 real data ===")
    for k in ("n_points","kappa_bound","naive_dim_target","naive_dim_data","certified_dim",
              "smallest_certified_tolerance","refutation_lower_bound","design_condition"):
        print(f"  {k}: {d12[k]}")
    print("  plug-in spectrum:", [round(v,3) for v in d12["plug_in_spectrum_target"]])

if (RES/"exp13_tolerance.json").exists():
    d13=load("exp13_tolerance"); e13=pd.DataFrame(d13["rows"]); sy=e13[e13.d_true>0]
    print("\n=== EXP13 tolerance sweep ===")
    for d in d13["deltas"]:
        print(f"  delta={d}: worst FCR threshold {e13[f'taufc_{d}'].max():.3f} "
              f"SymCert {e13[f'symfc_{d}'].max():.3f} | detection threshold "
              f"{sy[f'taudet_{d}'].mean():.3f} SymCert {sy[f'symdet_{d}'].mean():.3f}")

if (RES/"compute_log.json").exists():
    cl=load("compute_log")
    print("\n=== COMPUTE ===")
    tot=sum(v["cpu_seconds"] for v in cl.values() if isinstance(v,dict))
    for k,v in cl.items():
        if isinstance(v,dict): print(f"  {k}: {v['cpu_seconds']:.0f} CPU-s")
    print(f"  TOTAL: {tot/3600:.2f} CPU-hours")


# --------------------------------------------------------------- LaTeX tables
def write_latex_tables():
    """Emit each table as a complete LaTeX environment, so the manuscript never
    carries a number by hand."""
    out = ROOT / "paper" / "tables"
    out.mkdir(exist_ok=True)

    # ---- Table: calibration -------------------------------------------
    d1 = load("exp1_calibration"); df = pd.DataFrame(d1["rows"]); sym = df[df.d_true > 0]
    rows = []
    for t in d1["taus"]:
        rows.append((rf"fixed threshold $\tau={t}$", df[f"fc_tau{t}"], sym[f"det_tau{t}"]))
    for lab, fc, det in [("largest eigengap", "fc_gap", "det_gap"),
                         ("nullspace rank test", "fc_rank", "det_rank"),
                         ("split significance test", "fc_split", "det_split"),
                         ("Weyl perturbation bound", "fc_weyl", "det_weyl")]:
        rows.append((lab, df[fc], sym[det]))
    body = []
    for lab, fc, det in rows:
        body.append(f"{lab} & {fc.mean():.3f} & {fc.max():.3f} & {det.mean():.3f} & "
                    f"{int((fc > ALPHA).sum())} / {len(df)} \\\\")
    body.append(r"\midrule")
    for lab, fc, det in [(r"\symcert{} (ours)", "fc_sym", "det_sym"),
                         (r"\symcert{} + sample splitting", "fc_symp", "det_symp")]:
        f, d = df[fc], sym[det]
        body.append(rf"{lab} & \textbf{{{f.mean():.3f}}} & \textbf{{{f.max():.3f}}} & "
                    rf"{d.mean():.3f} & \textbf{{{int((f > ALPHA).sum())} / {len(df)}}} \\")
    (out / "calibration.tex").write_text(r"""\begin{table}[t]
\centering\small
\begin{tabular}{lrrrr}
\toprule
& \multicolumn{2}{c}{false certification} & detection & \\
\cmidrule(lr){2-3}
method & mean & worst setting & mean & settings above $\alpha$ \\
\midrule
""" + "\n".join(body) + r"""
\bottomrule
\end{tabular}
\caption{Calibration and power over """ + f"{len(df)}" + r""" settings, """
        + f"{d1['trials']}" + r""" replications each. ``Worst setting'' is the
maximum false-certification rate over the settings; the last column counts those
exceeding the nominal $\alpha = 0.05$. No fixed threshold controls the error,
and the two significance-test procedures --- both valid tests --- are the worst
offenders relative to their apparent rigour. The Weyl certificate is valid but
never certifies anything.}
\label{tab:calibration}
\end{table}
""")

    # ---- Table: scaling ------------------------------------------------
    if (RES / "exp11_scaling.json").exists():
        e = pd.DataFrame(load("exp11_scaling")["rows"])
        Ns = sorted(e.N.unique())[-3:]
        body = []
        for n, g in e.groupby("n"):
            gi = g.set_index("N")
            cells = " & ".join(f"{gi.upper_med.get(N, float('nan')):.4f}" for N in Ns)
            body.append(f"{n} & {int(gi.P.iloc[0])} & {int(gi.Q.iloc[0])} & "
                        f"{int(gi.d_true.iloc[0])} & {cells} & "
                        f"{gi.seconds_per_certificate.max():.2f} \\\\")
        hdr = " & ".join(rf"$N{{=}}{N}$" for N in Ns)
        (out / "scaling.tex").write_text(r"""\begin{table}[t]
\centering\small
\begin{tabular}{rrrrrrrr}
\toprule
$n$ & $P$ & $Q$ & true dim.\ & \multicolumn{3}{c}{certified tolerance (median)} & s / certificate \\
\cmidrule(lr){5-7}
 & & & & """ + hdr + r""" & \\
\midrule
""" + "\n".join(body) + r"""
\bottomrule
\end{tabular}
\caption{Scaling in the state dimension, on linear systems whose symmetry
algebra is the commutant of the coefficient matrix. The candidate search space
$P = n^2$ grows quadratically with no cost to validity; what costs is the model
class $Q$. False certification was zero in all """ + f"{len(e)}" + r""" cells.}
\label{tab:scaling}
\end{table}
""")

    # ---- Table: robustness ---------------------------------------------
    if (RES / "exp5_robustness.json").exists():
        a = pd.DataFrame(load("exp5_robustness")["design"])
        name = {("gauss", 0.0): "Gaussian", ("t", 0.0): "Student $t_4$",
                ("laplace", 0.0): "Laplace", ("gauss", 1.0): "Gaussian, heteroskedastic"}
        body = []
        for (nz, het, rb), g in a.groupby(["noise", "hetero", "robust"]):
            hi = g[g.N == g.N.max()]
            body.append(f"{name[(nz, het)]} & {'sandwich' if rb else 'exact Gaussian'} & "
                        f"{g.coverage.min():.3f} & {g.fc.max():.3f} & "
                        f"{hi['dim'].mean():.2f} \\\\")
        (out / "robustness.tex").write_text(r"""\begin{table}[t]
\centering\small
\begin{tabular}{llrrr}
\toprule
error distribution & covariance & worst coverage & worst false cert.\ & certified dim.\ \\
\midrule
""" + "\n".join(body) + r"""
\bottomrule
\end{tabular}
\caption{Coverage of the certificate under departures from the Gaussian
homoskedastic model, at nominal $0.95$. Each row spans three systems and two
regimes --- one where nothing is certifiable, so that coverage is all that is
being tested, and one where the true algebra is recovered --- with 400
replications each; the last column is the certified dimension in the informative
regime. The certificate depends on the errors only through a confidence
ellipsoid for a least-squares estimator, and is correspondingly insensitive to
their shape.}
\label{tab:robustness}
\end{table}
""")

    # ---- Table: trajectory designs --------------------------------------
    d2 = load("exp2_coverage")
    t = pd.DataFrame(d2["trajectories"])
    pretty = {"hopf": "Hopf", "vanderpol": "Van der Pol", "lorenz": "Lorenz"}
    body = []
    for _, r in t.iterrows():
        kap = r"$\infty$" if not np.isfinite(r.kappa) else f"{r.kappa:.1f}"
        body.append(f"{pretty[r.system]} & {r.design} & {int(r.n_points)} & {kap} & "
                    f"{int(r.d_true)} & {int(r.dim_naive_data)} & {int(r.dim_sym)} \\\\")
    (out / "trajectory.tex").write_text(r"""\begin{table}[t]
\centering\small
\begin{tabular}{llrrrrr}
\toprule
system & design & points & $\sqrt{\kappa\kappa'}$ & true dim.\ & nullspace dim.\ & \symcert{} dim.\ \\
\midrule
""" + "\n".join(body) + r"""
\bottomrule
\end{tabular}
\caption{Trajectory data. A single trajectory that settles onto an attractor
gives an infinite or large extrapolation factor: the data lie in, or near, the
zero set of a possible bracket, so a defect measured on them says little about
the target domain. \symcert{} declines to certify exactly there.}
\label{tab:trajectory}
\end{table}
""")
    print("wrote LaTeX tables")


if __name__ == "__main__":
    write_latex_tables()
