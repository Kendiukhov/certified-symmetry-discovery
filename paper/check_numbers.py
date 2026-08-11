"""Cross-check every number quoted in the manuscript's running text.

Tables and figures are generated from ``results/`` and cannot drift. The numbers
that appear in prose can, so this script asserts each of them against the raw
result files. Run it after any re-run; a failure means the text and the data
have separated.
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
RES = ROOT / "results"


def load(name):
    return json.loads((RES / f"{name}.json").read_text())


def main() -> int:
    checks: list[tuple[str, bool, str]] = []

    def chk(label, ok, detail=""):
        checks.append((label, bool(ok), detail))

    d1 = pd.DataFrame(load("exp1_calibration")["rows"])
    sym1 = d1[d1.d_true > 0]
    chk("threshold worst-case FCR is 63%", abs(d1["fc_tau0.05"].max() - 0.633) < 0.002,
        f"{d1['fc_tau0.05'].max():.3f}")
    chk("SymCert FCR is 0 across 240 settings", d1.fc_sym.max() == 0 and len(d1) == 240)
    chk("eigengap reaches 100%", d1.fc_gap.max() == 1.0)
    chk("rank test reaches 100%", d1.fc_rank.max() == 1.0)
    chk("split significance reaches 92%", abs(d1.fc_split.max() - 0.92) < 0.01)
    chk("SymCert detection is 0.484", abs(sym1.det_sym.mean() - 0.484) < 0.005)
    chk("72,000 replications", len(d1) * load("exp1_calibration")["trials"] == 72000)

    d2 = pd.DataFrame(load("exp2_coverage")["rows"])
    r = d2.kappa / d2.bound_critical
    chk("13 coverage settings below the bound, none fail",
        (r < 1).sum() == 13 and d2[r < 1].fc_naive_data.max() == 0)
    chk("37 above the bound, 14 fail",
        (r >= 1).sum() == 37 and (d2[r >= 1].fc_naive_data > 0).sum() == 14)

    d4 = pd.DataFrame(load("exp4_resolution")["rows"])
    z = d4[d4.eps == 0]
    rat = [float(np.median(z[z.sigma_rel == s].upper_med / z[z.sigma_rel == s].lecam))
           for s in sorted(z.sigma_rel.unique())]
    chk("Le Cam ratio spans 8.1 to 8.8",
        abs(min(rat) - 8.1) < 0.05 and abs(max(rat) - 8.8) < 0.05,
        f"{min(rat):.2f}-{max(rat):.2f}")
    rr = d4[(d4.N == 1600) & (d4.sigma_rel == 0.1)].set_index("eps")
    chk("refutation rate 0.495 at eps=0.02", abs(rr.refute_rate[0.02] - 0.495) < 0.005)
    chk("false refutation 0.010 at eps=0", abs(rr.refute_rate[0.0] - 0.010) < 0.002)

    d6 = pd.DataFrame(load("exp6_dimension")["rows"])
    g = d6.groupby(["system", "degF", "gens"]).agg(
        us=("upper_sim", "mean"), up=("upper_split", "mean"),
        ds=("detect_sim", "mean"), dp=("detect_split", "mean"))
    chk("simultaneous beats split in all nine configurations",
        (g.us < g.up).all() and (g.ds >= g.dp).all() and len(g) == 9)
    chk("width ratio spans 1.5 to 2.6",
        abs((g.up / g.us).min() - 1.53) < 0.02 and abs((g.up / g.us).max() - 2.60) < 0.02,
        f"{(g.up / g.us).min():.2f}-{(g.up / g.us).max():.2f}")

    d8 = load("exp8_tightness")
    b8 = pd.DataFrame(d8["boundary"])
    chk("boundary FCR is 0 in nine configurations",
        b8.fc_direction.max() == 0 and len(b8) == 9)
    chk("2,700 boundary replications", len(b8) * d8["boundary_trials"] == 2700)
    nar = 1 - b8.width_bootstrap / b8.width_exact
    chk("bootstrap is 9% to 18% narrower",
        abs(nar.min() - 0.089) < 0.005 and abs(nar.max() - 0.178) < 0.005,
        f"{100 * nar.min():.1f}-{100 * nar.max():.1f}%")
    chk("bootstrap FCR 0.157 at alpha=0.5",
        abs(b8[b8.alpha == 0.5].fc_bootstrap.max() - 0.1567) < 0.002)
    cov8 = pd.DataFrame(d8["coverage"])
    chk("interval coverage is 1.000 at every nominal level",
        all(v == 1.0 for _, r_ in cov8.iterrows() for v in r_["coverage"].values()))

    d10 = load("exp10_noise")
    bi = pd.DataFrame(d10["bias"])
    op = pd.DataFrame(d10["optimism"])
    chk("32,000 noise replications", len(bi) * d10["trials"] == 32000)
    chk("smallest plug-in defect ever seen is 0.066",
        abs(bi.minimum.min() - 0.0662) < 0.001)
    med = bi.groupby("sigma_rel").apply(
        lambda gg: float(np.median(gg["median"] / gg.true_min_defect)),
        include_groups=False)
    chk("median ratio 1.64 at noise level 1.0", abs(med[1.0] - 1.64) < 0.01)
    o100 = op[op.N == 100].set_index("system")
    chk("optimism +0.0150 on Van der Pol",
        abs(o100.gap_mean["vanderpol"] - 0.0150) < 0.0005)
    chk("optimism -0.0073 on the asymmetric rigid body",
        abs(o100.gap_mean["rigid_asym"] + 0.0073) < 0.0005)

    d11 = pd.DataFrame(load("exp11_scaling")["rows"])
    chk("scaling FCR is 0 in all 20 cells", d11.fc.max() == 0 and len(d11) == 20)
    chk("certificate costs 0.015 to 0.031 CPU-seconds",
        abs(d11.cpu_seconds_per_certificate.min() - 0.0152) < 0.002
        and abs(d11.cpu_seconds_per_certificate.max() - 0.0313) < 0.002)

    d12 = load("exp12_realdata")
    chk("real data has 19 usable pairs", d12["n_points"] == 19)
    chk("real data smallest certified tolerance 2.3",
        abs(d12["smallest_certified_tolerance"] - 2.32) < 0.02)
    chk("real data extrapolation factor 58", abs(d12["kappa_bound"] - 57.9) < 0.5)

    d7 = pd.DataFrame(load("exp7_invariance")["rows"])
    chk("sector training gives a threshold FCR of 0.76",
        abs(d7[d7.sector].fc_naive_t.max() - 0.76) < 0.01)
    chk("SymCert FCR is 0 on the invariance task", d7.fc_sym.max() == 0)

    bad = [c for c in checks if not c[1]]
    for lab, ok, det in checks:
        print(("  OK  " if ok else "  !!  ") + lab + (f"   [{det}]" if det else ""))
    print(f"\n{len(checks) - len(bad)}/{len(checks)} numeric claims verified "
          f"against results/")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
