"""Experiment 19 -- a physically expected symmetry in real ocean measurements.

Near-inertial currents in the upper ocean obey, to first order, the damped slab
model ``du/dt = f v - r u``, ``dv/dt = -f u - r v`` plus wind forcing, where
``f`` is the local Coriolis parameter.  In units of ``1/f`` the right-hand side
is ``(-r I + J) u`` with ``J`` the planar rotation, which commutes with the
rotation generator: the dynamics are rotation invariant (isotropic).  We ask
whether hourly drifter velocities certify that symmetry, and to what tolerance.

Data: NOAA Global Drifter Program hourly product (Elipot et al., v2.01,
doi:10.25921/x46c-3620, CC BY 4.0), North Pacific 30-45 N, 180-140 W, June to
September 2016-2019, drogued drifters only (``time < drogue_lost_date``),
interpolation gaps of at most six hours.  The processed subset is stored in
``data/`` and regenerated from the ERDDAP server if absent.

Processing, identical for both velocity components so that it cannot create
or destroy rotation invariance: contiguous hourly segments of at least 150
hours; removal of a centred running mean (49 hours by default) to isolate the
near-inertial band; central differences for the derivative; time measured in
units of the local ``1/f`` and velocity in units of its root-mean-square.

The certificate is computed for the rotation generator (fixed in advance by the
physics) and for the plug-in subspace, with four confidence regions (OLS,
White, cluster-robust by drifter, Newey--West), polynomial classes of degree 1
to 3, three filter windows, each summer separately, and simultaneously over the
classes of degree 1 to 3 (``hsd.selection``).  The target measure is the
standard Gaussian in the normalised velocity, close to the distribution of the
data.  Caveats (reported with the results): velocities carry measurement error
(errors in variables), and the class is an assumption.
"""

from __future__ import annotations

import io
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from common import ALPHA, DELTA, ROOT, save
from hsd import DefectProblem, fit_ols
from hsd.certify import certified_dimension, certify_direction, refute_all
from hsd.coverage import extrapolation_factor
from hsd.selection import omitted_part_bound

DATA = ROOT / "data" / "gdp_np_drogued_2016_2019.csv.gz"
ERDDAP = ("https://erddap.aoml.noaa.gov/gdp/erddap/tabledap/drifter_hourly_qc.csv?"
          "ID%2Ctime%2Clatitude%2Clongitude%2Cve%2Cvn%2Cerr_ve%2Cerr_vn%2C"
          "drogue_lost_date%2Cgap&latitude%3E=30&latitude%3C=45&longitude%3E=-180"
          "&longitude%3C=-140&time%3E={y}-06-01T00%3A00%3A00Z"
          "&time%3C={y}-09-30T23%3A00%3A00Z")
YEARS = [2016, 2017, 2018, 2019]
OMEGA = 7.2921e-5
ESTIMATORS = ("ols", "hc0", "cluster", "hac")


def fetch() -> pd.DataFrame:
    """Download the four summers, keep drogued rows with short gaps."""
    frames = []
    for y in YEARS:
        with urllib.request.urlopen(ERDDAP.format(y=y), timeout=600) as r:
            frames.append(pd.read_csv(io.BytesIO(r.read()), skiprows=[1]))
    df = pd.concat(frames)
    df["time"] = pd.to_datetime(df.time)
    df["drogue_lost_date"] = pd.to_datetime(df.drogue_lost_date)
    df = df[(df.time < df.drogue_lost_date) & (df.gap <= 6 * 3600)]
    df = df[["ID", "time", "latitude", "longitude", "ve", "vn", "err_ve", "err_vn"]]
    DATA.parent.mkdir(exist_ok=True)
    df.to_csv(DATA, index=False, compression="gzip")
    return df


def load_data() -> pd.DataFrame:
    if not DATA.exists():
        return fetch()
    return pd.read_csv(DATA, parse_dates=["time"])


def design(df: pd.DataFrame, window: int = 49, min_len: int = 150, years=None):
    """State-derivative pairs in units of the local inertial period."""
    if years is not None:
        df = df[df.time.dt.year.isin(years)]
    Xs, Ys, G, E = [], [], [], []
    for did, g in df.groupby("ID"):
        g = g.sort_values("time")
        t = (g.time - g.time.iloc[0]).dt.total_seconds().values / 3600.0
        brk = np.flatnonzero(np.diff(t) != 1.0)
        for seg in np.split(np.arange(len(g)), brk + 1):
            if len(seg) < min_len:
                continue
            u = g[["ve", "vn"]].values[seg]
            f = 2 * OMEGA * np.sin(np.radians(g.latitude.values[seg])) * 3600.0
            kern = np.ones(window) / window
            low = np.stack([np.convolve(u[:, j], kern, mode="same") for j in range(2)], 1)
            h = window
            up, ff = (u - low)[h:-h], f[h:-h]
            if len(up) < 3:
                continue
            du = (up[2:] - up[:-2]) / 2.0
            Xs.append(up[1:-1])
            Ys.append(du / ff[1:-1, None])
            G.append(np.full(len(du), did))
            E.append(g[["err_ve", "err_vn"]].values[seg][h:-h][1:-1])
    X, Y = np.vstack(Xs), np.vstack(Ys)
    s = float(np.sqrt(np.mean(np.sum(X ** 2, axis=1)) / 2))
    return X / s, Y / s, np.concatenate(G), dict(rms_velocity=s,
                                                  median_velocity_error=float(np.median(np.vstack(E))))


def _rot(prob):
    r = np.zeros(prob.P)
    r[1], r[2] = -1.0, 1.0
    return r / np.linalg.norm(r)


def analyse(X, Y, groups, degree, cov):
    prob = DefectProblem.build(2, degree, "affine", "gauss", scale=1.0)
    fit = fit_ols(prob, X, Y, cov=cov, groups=groups)
    c = certify_direction(prob, _rot(prob), fit, ALPHA)
    d, _ = certified_dimension(prob, fit, DELTA, ALPHA)
    rho, _ = prob.spectrum(fit.beta)
    return dict(degree=degree, Q=prob.Q, cov=cov, N=len(X),
                drifters=int(np.unique(groups).size), plug_rot=c.plug_in,
                upper_rot=c.upper, lower_rot=c.lower, certified_dim=d,
                refute=refute_all(prob, fit, ALPHA),
                spectrum=[float(v) for v in rho],
                kappa=float(extrapolation_factor(prob, X)["bound"]),
                linear=fit.beta.reshape(2, -1)[:, :3].tolist() if degree >= 1 else None)


def simultaneous_rot(X, Y, groups, cov, degrees=(1, 2, 3)):
    """Upper bound on the rotation defect, simultaneously over the classes,
    assuming the largest contains the truth."""
    K = len(degrees)
    a = ALPHA / K
    probs = [DefectProblem.build(2, d, "affine", "gauss", scale=1.0) for d in degrees]
    fits = [fit_ols(p, X, Y, cov=cov, groups=groups) for p in probs]
    ups = []
    for k, (p, f) in enumerate(zip(probs, fits)):
        if k == K - 1:
            ups.append(certify_direction(p, _rot(p), f, a).upper)
        else:
            eta = omitted_part_bound(p, probs[-1], X, fits[-1], a)
            ups.append(certify_direction(p, _rot(p), f, a, eta_l2=(eta, probs[-1])).upper)
    return dict(cov=cov, uppers=ups, upper_min=float(min(ups)),
                best_degree=int(degrees[int(np.argmin(ups))]))


def main():
    w0 = time.time()
    df = load_data()
    X, Y, g, info = design(df)
    rows = [analyse(X, Y, g, d, c) for d in (1, 2, 3) for c in ESTIMATORS]
    windows = []
    for w in (37, 49, 73):
        Xw, Yw, gw, _ = design(df, window=w)
        windows.append(dict(window=w, **analyse(Xw, Yw, gw, 1, "cluster")))
    years = []
    for y in YEARS:
        Xy, Yy, gy, _ = design(df, years=[y])
        r = dict(year=y, **analyse(Xy, Yy, gy, 1, "hac"))
        r["cluster_identified"] = bool(np.unique(gy).size > 6)
        if r["cluster_identified"]:
            r["upper_rot_cluster"] = analyse(Xy, Yy, gy, 1, "cluster")["upper_rot"]
        years.append(r)
    simul = [simultaneous_rot(X, Y, g, c) for c in ("cluster", "hac")]
    save("exp19_drifters", {"rows": rows, "windows": windows, "years": years,
                            "simultaneous": simul, "n_rows_raw": int(len(df)),
                            "n_drifters_raw": int(df.ID.nunique()), **info,
                            "data": str(DATA.relative_to(ROOT)),
                            "dataset_doi": "10.25921/x46c-3620"})
    print(f"exp19 wall {time.time()-w0:.0f}s")


if __name__ == "__main__":
    main()
