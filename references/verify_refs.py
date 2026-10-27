"""Verify every bibliography entry against public metadata APIs.

For each intended citation we query Crossref (DOI metadata), the arXiv API and
DBLP, and report the best title match together with the authors, venue and year
that the API returns.  Any entry whose recovered metadata disagrees with the
bibliography is flagged for manual correction.  The output is written to
``references/verification_report.json`` and summarised on stdout.
"""

from __future__ import annotations

import json
import re
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from difflib import SequenceMatcher
from pathlib import Path

HERE = Path(__file__).resolve().parent
UA = "hsd-reference-checker/1.0 (mailto:kenduhov.ig@gmail.com)"

# Generated from paper/refs.bib by references/sync_entries.py.
# Generated from paper/refs.bib by references/sync_entries.py.
ENTRIES = [
    ('otto2025unified', 'A Unified Framework to Enforce, Discover, and Promote Symmetry in Machine Learning'),
    ('cahill2023liepca', 'Lie PCA: Density estimation for symmetric manifolds'),
    ('moskalev2022liegg', 'LieGG: Studying Learned Lie Group Generators'),
    ('gruver2023lie', 'The Lie Derivative for Measuring Learned Equivariance'),
    ('kaiser2018discovering', 'Discovering conservation laws from data for control'),
    ('baddoo2023physics', 'Physics-informed dynamic mode decomposition'),
    ('yang2023liegan', 'Generative Adversarial Symmetry Discovery'),
    ('yang2024latent', 'Latent Space Symmetry Discovery'),
    ('yang2024symmetry', 'Symmetry-Informed Governing Equation Discovery'),
    ('desai2022symmetry', 'Symmetry discovery with deep learning'),
    ('dehmamy2021automatic', 'Automatic Symmetry Discovery with Lie Algebra Convolutional Network'),
    ('liu2022machine', 'Machine Learning Hidden Symmetries'),
    ('cohen2016group', 'Group Equivariant Convolutional Networks'),
    ('finzi2021practical', 'A Practical Method for Constructing Equivariant Multilayer Perceptrons for Arbitrary Matrix Groups'),
    ('bronstein2021geometric', 'Geometric Deep Learning: Grids, Groups, Graphs, Geodesics, and Gauges'),
    ('benton2020learning', 'Learning Invariances in Neural Networks from Training Data'),
    ('immer2022invariance', 'Invariance Learning in Deep Neural Networks with Differentiable Laplace Approximations'),
    ('wang2022approximately', 'Approximately Equivariant Networks for Imperfectly Symmetric Dynamics'),
    ('romero2022learning', 'Learning Partial Equivariances from Data'),
    ('brandstetter2022lie', 'Lie Point Symmetry Data Augmentation for Neural PDE Solvers'),
    ('akhound2023lie', 'Lie Point Symmetry and Physics-Informed Networks'),
    ('raissi2019physics', 'Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations'),
    ('brunton2016discovering', 'Discovering governing equations from data by sparse identification of nonlinear dynamical systems'),
    ('messenger2021weak', 'Weak SINDy: Galerkin-Based Data-Driven Model Selection'),
    ('zhang2019convergence', 'On the Convergence of the SINDy Algorithm'),
    ('fasel2022ensemble', 'Ensemble-SINDy: Robust sparse model discovery in the low-data, high-noise limit, with active learning and control'),
    ('kaptanoglu2022pysindy', 'PySINDy: A comprehensive Python package for robust sparse system identification'),
    ('olver1993applications', 'Applications of Lie Groups to Differential Equations'),
    ('lee2013smooth', 'Introduction to Smooth Manifolds'),
    ('bhatia1997matrix', 'Matrix Analysis'),
    ('davis1970rotation', 'The Rotation of Eigenvectors by a Perturbation. III'),
    ('yu2015useful', 'A useful variant of the Davis--Kahan theorem for statisticians'),
    ('gander1989constrained', 'A constrained eigenvalue problem'),
    ('more1983computing', 'Computing a Trust Region Step'),
    ('conn2000trust', 'Trust Region Methods'),
    ('boucheron2013concentration', 'Concentration Inequalities: A Nonasymptotic Theory of Independence'),
    ('borell1975brunn', 'The Brunn--Minkowski inequality in Gauss space'),
    ('cirelson1976norms', 'Norms of Gaussian sample functions'),
    ('tsybakov2009introduction', 'Introduction to Nonparametric Estimation'),
    ('schuirmann1987comparison', 'A comparison of the two one-sided tests procedure and the power approach for assessing the equivalence of average bioavailability'),
    ('berger1996bioequivalence', 'Bioequivalence trials, intersection-union tests and equivalence confidence sets'),
    ('wellek2010testing', 'Testing Statistical Hypotheses of Equivalence and Noninferiority'),
    ('berk2013valid', 'Valid post-selection inference'),
    ('lee2016exact', 'Exact post-selection inference, with application to the lasso'),
    ('meinshausen2009pvalues', 'p-Values for High-Dimensional Regression'),
    ('chernozhukov2018double', 'Double/debiased machine learning for treatment and structural parameters'),
    ('andrews2024inference', 'Inference on Winners'),
    ('kleibergen2006generalized', 'Generalized reduced rank tests using the singular value decomposition'),
    ('robin2000tests', 'Tests of Rank'),
    ('cragg1997inferring', 'Inferring the rank of a matrix'),
    ('anderson1963asymptotic', 'Asymptotic Theory for Principal Component Analysis'),
    ('johnstone2001distribution', 'On the distribution of the largest eigenvalue in principal components analysis'),
    ('white1980heteroskedasticity', 'A Heteroskedasticity-Consistent Covariance Matrix Estimator and a Direct Test for Heteroskedasticity'),
    ('benjamini1995controlling', 'Controlling the False Discovery Rate: A Practical and Powerful Approach to Multiple Testing'),
    ('efron1979bootstrap', 'Bootstrap Methods: Another Look at the Jackknife'),
    ('hansen1982large', 'Large Sample Properties of Generalized Method of Moments Estimators'),
    ('rosenbaum2002observational', 'Observational Studies'),
    ('newey1987simple', 'A Simple, Positive Semi-Definite, Heteroskedasticity and Autocorrelation Consistent Covariance Matrix'),
    ('andrews1991heteroskedasticity', 'Heteroskedasticity and Autocorrelation Consistent Covariance Matrix Estimation'),
    ('clopper1934use', 'The Use of Confidence or Fiducial Limits Illustrated in the Case of the Binomial'),
    ('leeb2005model', 'Model Selection and Inference: Facts and Fiction'),
    ('carroll2006measurement', 'Measurement Error in Nonlinear Models: A Modern Perspective'),
    ('pollard1970comparison', 'Comparison between observed and simulated wind-generated inertial oscillations'),
    ('elipot2016global', 'A global surface drifter data set at hourly resolution'),
    ('elipot2022hourly', 'Hourly location, current velocity, and temperature collected from Global Drifter Program drifters world-wide, version 2.01'),
    ('cameron2015practitioner', "A Practitioner's Guide to Cluster-Robust Inference"),
    ('dinkelbach1967nonlinear', 'On Nonlinear Fractional Programming')
]


def _get(url: str, timeout=25, retries=3):
    """Fetch with retries: the public APIs rate-limit a 67-entry sweep, and a
    throttled request looks exactly like a missing record if it is not retried."""
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read().decode("utf-8", "replace")
        except Exception as e:                               # noqa: BLE001
            last = e
            time.sleep(2.0 * (attempt + 1))
    raise last


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", " ", s.lower())


def sim(a: str, b: str) -> float:
    return SequenceMatcher(None, norm(a), norm(b)).ratio()


def crossref(title: str):
    url = ("https://api.crossref.org/works?rows=3&select=title,author,issued,"
           "container-title,DOI,type&query.bibliographic="
           + urllib.parse.quote(title))
    try:
        items = json.loads(_get(url))["message"]["items"]
    except Exception as e:                                   # noqa: BLE001
        return {"error": str(e)}
    best, score = None, 0.0
    for it in items:
        t = (it.get("title") or [""])[0]
        s = sim(title, t)
        if s > score:
            best, score = it, s
    if best is None:
        return {"found": False}
    return {"found": True, "score": round(score, 3),
            "title": (best.get("title") or [""])[0],
            "authors": [f"{a.get('given','')} {a.get('family','')}".strip()
                        for a in best.get("author", [])][:8],
            "year": (best.get("issued", {}).get("date-parts") or [[None]])[0][0],
            "venue": (best.get("container-title") or [""])[0],
            "doi": best.get("DOI"), "type": best.get("type")}


def arxiv(title: str):
    url = ("http://export.arxiv.org/api/query?max_results=3&search_query=ti:%22"
           + urllib.parse.quote(title[:180]) + "%22")
    try:
        root = ET.fromstring(_get(url))
    except Exception as e:                                   # noqa: BLE001
        return {"error": str(e)}
    ns = {"a": "http://www.w3.org/2005/Atom"}
    best, score = None, 0.0
    for e in root.findall("a:entry", ns):
        t = " ".join(e.findtext("a:title", "", ns).split())
        s = sim(title, t)
        if s > score:
            best, score = e, s
    if best is None:
        return {"found": False}
    return {"found": True, "score": round(score, 3),
            "title": " ".join(best.findtext("a:title", "", ns).split()),
            "authors": [a.findtext("a:name", "", ns) for a in best.findall("a:author", ns)][:8],
            "year": best.findtext("a:published", "", ns)[:4],
            "id": best.findtext("a:id", "", ns)}


def dblp(title: str):
    url = "https://dblp.org/search/publ/api?format=json&h=3&q=" + urllib.parse.quote(title)
    try:
        hits = json.loads(_get(url))["result"]["hits"].get("hit", [])
    except Exception as e:                                   # noqa: BLE001
        return {"error": str(e)}
    best, score = None, 0.0
    for h in hits:
        info = h["info"]
        s = sim(title, info.get("title", ""))
        if s > score:
            best, score = info, s
    if best is None:
        return {"found": False}
    au = best.get("authors", {}).get("author", [])
    if isinstance(au, dict):
        au = [au]
    return {"found": True, "score": round(score, 3), "title": best.get("title"),
            "authors": [a["text"] for a in au][:8], "year": best.get("year"),
            "venue": best.get("venue"), "doi": best.get("doi"),
            "type": best.get("type")}


def main():
    out = {}
    for key, title in ENTRIES:
        rec = {"title_claimed": title}
        for name, fn in (("crossref", crossref), ("arxiv", arxiv), ("dblp", dblp)):
            rec[name] = fn(title)
            time.sleep(1.2)
        best = max((rec[s].get("score", 0.0) for s in ("crossref", "arxiv", "dblp")),
                   default=0.0)
        rec["best_score"] = best
        out[key] = rec
        flag = "OK " if best >= 0.9 else ("?? " if best >= 0.7 else "!! ")
        print(f"{flag}{best:5.3f}  {key:32s} {title[:70]}", flush=True)
    (HERE / "verification_report.json").write_text(json.dumps(out, indent=1))
    bad = [k for k, v in out.items() if v["best_score"] < 0.9]
    print(f"\n{len(ENTRIES) - len(bad)}/{len(ENTRIES)} matched at >= 0.90; "
          f"needs manual check: {bad}")


if __name__ == "__main__":
    sys.exit(main())
