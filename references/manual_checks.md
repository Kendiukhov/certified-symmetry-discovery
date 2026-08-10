# Entries confirmed by hand

`verify_refs.py` queries Crossref, arXiv and DBLP. A 67-entry sweep is heavily
rate-limited, and books and conference papers often have no Crossref record at
the level of the whole work, so a low automatic score does not mean an entry is
wrong. The entries below did not clear the 0.90 automatic threshold on at least
one run and were confirmed individually; where a DOI exists, the volume, issue,
page range and year in `paper/refs.bib` were taken from the DOI record itself.

| key | how it was confirmed |
|---|---|
| `rao1999learning` | NIPS 11 table of contents; authors Rao and Ruderman |
| `benton2020learning` | NeurIPS 2020 proceedings page; volume 33, pp. 17605--17616 |
| `vanderwilk2018learning` | arXiv API exact title match (arXiv:1808.05563); NeurIPS 31 |
| `wang2022approximately` | DBLP exact match, ICML 2022, PMLR 162:23078--23091 |
| `brandstetter2022lie` | PMLR v162 landing page; pp. 2241--2256 |
| `olver1993applications` | Crossref DOI 10.1007/978-1-4612-4350-2 (GTM 107, 2nd ed., 1993) |
| `lee2013smooth` | Crossref DOI 10.1007/978-1-4419-9982-5 (GTM 218, 2nd ed., 2013) |
| `bhatia1997matrix` | Crossref DOI 10.1007/978-1-4612-0653-8 (GTM 169, 1997) |
| `gander1989constrained` | Crossref DOI 10.1016/0024-3795(89)90494-1, LAA 114--115:815--839 |
| `conn2000trust` | Crossref DOI 10.1137/1.9780898719857 (SIAM, 2000) |
| `boucheron2013concentration` | Crossref DOI 10.1093/acprof:oso/9780199535255.001.0001 (OUP, 2013) |
| `tsybakov2009introduction` | Crossref DOI 10.1007/b13794 (Springer, 2009) |
| `maurer2009empirical` | DBLP exact match, COLT 2009 |
| `berk2013valid` | Crossref DOI 10.1214/12-AOS1077, Ann. Statist. 41(2):802--837 |
| `meinshausen2009pvalues` | Crossref DOI 10.1198/jasa.2009.tm08647, JASA 104(488):1671--1681 |
| `robin2000tests` | Crossref DOI 10.1017/S0266466600162012, Econometric Theory 16(2):151--175 |

Two entries were corrected as a result of this process: the page range of
`cahill2023liepca` (279--295, not 236--253) and the venue of
`chernozhukov2018double` (The Econometrics Journal 21(1):C1--C68, not the
working-paper version that the automatic search returned first).
