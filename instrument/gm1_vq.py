#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gate GM1-VQ — the fresh calibration pass required for a NEW token space.

  Standing rule: every new token space (image VQ, video VQ) requires the
  nominal-token estimator adaptation AND a fresh GM1 pass before any claim
  built on it leaves.  Estimators transfer between token spaces; their
  calibration does not.

Same four known-answer subjects as `gm1_calibration.py`, same m, same replicate
count, same pre-registered thresholds -- but every observation passes through
the nominal-VQ path of `vq.py`: latent -> nearest-neighbour quantizer built from
the real aMUSEd-256 codebook -> nominal code index -> PC projection.

Two things this pass must establish, and it reports both even when the answer
is inconvenient:
  1. the estimator path is calibrated in the projected space (a)-(c);
  2. WHICH nulls survive the token space and which do not (d).

Run:  python3 gm1_vq.py [--quick]
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
from scipy import stats

from ebf import AR1Copula, EBF, GMRF, LeakyEBF
from estimators import (bh_fdr, calibrated_edge_ci, calibrated_p, fisher_z,
                        pair_corr, partial_corr, rho_by_lag)
from gm1_calibration import ALPHA, NOMINAL_HI, NOMINAL_LO, _cluster_ci, _wilson
from graphs import build_graph, edge_list, nonedge_list
from vq import Quantizer, VQField, codebook_pcs, load_codebook, projected_corr

K_HERMITE = 64


def make_quantizer(n_codes: int | None = None, pc: int = 0,
                   seed: int = 20260727) -> tuple[Quantizer, np.ndarray]:
    """The real aMUSEd codebook, optionally subsampled to `n_codes`."""
    E = load_codebook()
    sc, evr = codebook_pcs(E, 3)
    ids = np.arange(len(sc))
    if n_codes is not None and n_codes < len(sc):
        ids = np.random.default_rng(seed).choice(len(sc), n_codes, replace=False)
    return Quantizer(sc[ids, pc], ids), evr


def _bank(field, m, n_rep, rng, pairs):
    out = np.empty((n_rep, len(pairs)))
    for b in range(n_rep):
        out[b] = pair_corr(field.sample(m, rng), pairs)
    return out


# ----------------------------------------------------------------- (a) coverage
def test_a_coverage(graph, eps, quant, m, n_rep, rng):
    A = build_graph(graph)
    f = VQField(EBF(A, eps), quant, K_HERMITE)
    e, ne = edge_list(A), nonedge_list(A)
    truth = f.truth()
    tgt_e = truth[e[:, 0], e[:, 1]]

    bank_all = _bank(f, m, n_rep, rng, np.vstack([e, ne]))
    bank_e, bank = bank_all[:, :len(e)], bank_all[:, len(e):]
    max_null = np.abs(bank).max(1)

    cov, cov_tb = np.empty(n_rep), np.empty(n_rep)
    fdr_any = np.zeros(n_rep, bool)
    fwer_rej = 0
    infl = float("nan")
    for r in range(n_rep):
        X = f.sample(m, rng)
        rho = pair_corr(X, e)
        half = 1.96 / np.sqrt(m - 3)
        lo_t, hi_t = np.tanh(np.arctanh(rho) - half), np.tanh(np.arctanh(rho) + half)
        cov_tb[r] = ((lo_t <= tgt_e) & (tgt_e <= hi_t)).mean()
        lo, hi, inf_ = calibrated_edge_ci(rho, bank_e, m)
        cov[r] = ((lo <= tgt_e) & (tgt_e <= hi)).mean()
        infl = float(inf_.mean())
        stat = pair_corr(X, ne)
        fwer_rej += int((1.0 + (max_null >= np.abs(stat).max()).sum())
                        / (len(max_null) + 1.0) <= ALPHA)
        fdr_any[r] = bh_fdr(calibrated_p(stat, bank, m)[0], ALPHA).sum() > 0

    ec, fwer = float(cov.mean()), fwer_rej / n_rep
    res = dict(graph=graph, eps=eps, m=m, n_rep=n_rep, q=quant.q,
               target_edge=float(tgt_e[0]), latent_eps=float(eps),
               attenuation=float(tgt_e[0] / eps),
               edge_coverage=ec, edge_coverage_ci=_cluster_ci(cov),
               edge_coverage_textbook=float(cov_tb.mean()),
               ci_inflation_vs_normal_theory=infl,
               fwer_rejection_rate=fwer, fwer_ci=_wilson(fwer_rej, n_rep),
               fdr_any_rejection_rate=float(fdr_any.mean()),
               n_edges=len(e), n_nonedges=len(ne))
    lo_ci, hi_ci = res["edge_coverage_ci"]
    res["pass_edge_cov"] = bool(NOMINAL_LO <= ec <= NOMINAL_HI or lo_ci <= 0.95 <= hi_ci)
    tol = 2 * np.sqrt(ALPHA * (1 - ALPHA) / n_rep)
    res["pass_fwer"] = bool(fwer <= ALPHA + tol)
    res["pass_fdr"] = bool(res["fdr_any_rejection_rate"] <= ALPHA + tol)
    res["pass"] = bool(res["pass_edge_cov"] and res["pass_fwer"] and res["pass_fdr"])
    return res


# -------------------------------------------------------------------- (b) power
def test_b_power(graph, eps, quant, m, n_rep, deltas, n_inject, rng):
    A = build_graph(graph)
    ne = nonedge_list(A)
    f0 = VQField(EBF(A, eps), quant, K_HERMITE)
    bank = _bank(f0, m, max(n_rep, 100), rng, ne)

    dist = np.abs(ne[:, 0] - ne[:, 1])
    pick = ne[np.argsort(-dist)[:n_inject]]
    inj = np.array([np.flatnonzero((ne[:, 0] == i) & (ne[:, 1] == j))[0] for i, j in pick])
    oth = np.setdiff1d(np.arange(len(ne)), inj)

    rows = []
    for d in deltas:
        f = f0 if d == 0 else VQField(LeakyEBF(A, eps, pick, d), quant, K_HERMITE)
        pi, po = np.zeros(n_rep), np.zeros(n_rep)
        for r in range(n_rep):
            rej = bh_fdr(calibrated_p(pair_corr(f.sample(m, rng), ne), bank, m)[0], ALPHA)
            pi[r], po[r] = rej[inj].mean(), rej[oth].mean()
        rows.append(dict(delta=d, delta_projected=float(projected_corr(d, f0.w)),
                         power_injected=float(pi.mean()),
                         false_positive_rate=float(po.mean())))
    pw = [r["power_injected"] for r in rows]
    fp = [r["false_positive_rate"] for r in rows]
    return dict(graph=graph, eps=eps, q=quant.q, curve=rows,
                monotone=bool(all(b >= a - 1e-9 for a, b in zip(pw, pw[1:]))),
                reaches_full_power=bool(pw[-1] >= 0.95), localized=bool(max(fp) <= 0.02),
                pass_=bool(all(b >= a - 1e-9 for a, b in zip(pw, pw[1:]))
                           and pw[-1] >= 0.95 and max(fp) <= 0.02))


# ------------------------------------------------------------------- (c) AR(1)
def test_c_ar1(n, rhos, quant, m, max_lag, n_rep, rng):
    rows, ok = [], True
    for rho in rhos:
        f = VQField(AR1Copula(n, rho), quant, K_HERMITE)
        est = np.array([rho_by_lag(f.sample(m, rng), max_lag) for _ in range(n_rep)])
        for d in range(1, max_lag + 1):
            v = est[:, d]
            lo, hi = np.quantile(v, [0.025, 0.975])
            tgt = float(projected_corr(rho ** d, f.w))   # projected, not rho^d
            hit = bool(lo <= tgt <= hi)
            ok &= hit
            rows.append(dict(rho=rho, lag=d, latent_target=rho ** d, target=tgt,
                             mean=float(v.mean()), ci=[float(lo), float(hi)], covers=hit))
    return dict(n=n, m=m, n_rep=n_rep, q=quant.q, rows=rows, pass_=bool(ok))


# ------------------------------------------------- (d) separator, and its limit
def test_d_separator(graph, eps, quant, m, n_rep, rng, pcs=(1, 3)):
    """
    The GMRF's conditional independence is exact in the LATENT Gaussian.
    Quantisation is a coarsening: conditioning on a token is not conditioning on
    the latent, so the null is only as good as the resolution.  `ebf.py` records
    the extreme case -- binarised (q=2) the same separator reads |pcorr| = 0.0696,
    23% of the edge signal, i.e. no usable null at all.

    This test asks how much of that survives at real codebook resolution, and
    whether conditioning on more principal components recovers it.  Reported as
    a curve, not a verdict, because the answer sets the scope of every future
    conditional claim on image tokens.
    """
    A = build_graph(graph)
    n = A.shape[0]
    triples = [(i, i + 4, list(range(i + 1, i + 4))) for i in range(0, n - 5, 3)]
    E = load_codebook()
    sc, _ = codebook_pcs(E, 3)
    out = {}
    for k in pcs:
        quants = [Quantizer(sc[quant.code_ids, c], quant.code_ids) for c in range(k)]
        sub = {}
        for name, fld in (("gmrf", GMRF(A, eps)), ("ebf", EBF(A, eps))):
            pv, mags = [], []
            for _ in range(n_rep):
                Z = fld.latent(m, rng)
                # k-dim projection: stack PCs, condition on ALL of them per position
                P = [q.h(Z) for q in quants]
                for i, j, S in triples:
                    Scols = [c * n + s for s in S for c in range(k)]
                    X = np.concatenate(P, axis=1) if k > 1 else P[0]
                    idx_i, idx_j = i, j                       # PC1 of i and j
                    r = partial_corr(X, idx_i, idx_j, Scols)
                    mags.append(abs(r))
                    z = float(fisher_z(np.array([r]), m, df_extra=len(Scols))[0])
                    pv.append(2 * stats.norm.sf(abs(z)))
            pv = np.array(pv)
            sub[name] = dict(rejection_rate=float((pv <= ALPHA).mean()),
                             ci=_wilson(int((pv <= ALPHA).sum()), len(pv)),
                             ks_uniform_p=float(stats.kstest(pv, "uniform").pvalue),
                             median_abs_pcorr=float(np.median(mags)), n_tests=len(pv))
        sub["pass_"] = bool(sub["gmrf"]["rejection_rate"] <= 0.08
                            and sub["ebf"]["rejection_rate"] >= 0.5)
        out[f"pc{k}"] = sub
    out["pass_"] = bool(any(out[f"pc{k}"]["pass_"] for k in pcs))
    out["best_pc"] = min((k for k in pcs), key=lambda k: out[f"pc{k}"]["gmrf"]["rejection_rate"])
    return out


# --------------------------------------------------------------------- driver
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default="gm1_vq_report.json")
    a = ap.parse_args()

    m = 2000 if a.quick else 8000
    n_rep = 40 if a.quick else 200
    n_rep_b = 20 if a.quick else 100
    n_rep_c = 10 if a.quick else 50
    rng = np.random.default_rng(20260727)
    t0 = time.time()

    quant, evr = make_quantizer()
    w = quant.weights(K_HERMITE)
    rep = dict(mode="quick" if a.quick else "gate", m=m, n_rep=n_rep, alpha=ALPHA,
               date="2026-07-27", token_space="aMUSEd-256 VQ (8192 codes, 64-d)",
               pc1_evr=float(evr[0]), pc13_evr=float(evr[:3].sum()),
               hermite_K=K_HERMITE, hermite_residual=float(1 - w.sum()))

    print(f"=== GM1-VQ ({rep['mode']}, m={m}, replicates={n_rep}) ===")
    print(f"    token space: {rep['token_space']}   PC1 {evr[0]:.4f}  PC1-3 {evr[:3].sum():.4f}")
    print(f"    quantizer w1={w[0]:.5f}  high-order residual={1-w.sum():.2e}\n")

    print("(0) coupling attenuation induced by the token space (exact, closed form)")
    rep["attenuation"] = []
    for nc in (8192, 1024, 256, 64, 16):
        q2, _ = make_quantizer(nc)
        w2 = q2.weights(K_HERMITE)
        row = dict(n_codes=nc, **{f"eps{e}": float(projected_corr(e, w2) / e)
                                  for e in (0.15, 0.30, 0.45)})
        rep["attenuation"].append(row)
        print(f"    codes={nc:<6} attenuation @eps 0.15/0.30/0.45 = "
              f"{row['eps0.15']:.4f} / {row['eps0.3']:.4f} / {row['eps0.45']:.4f}")

    print("\n(a) exact-field coverage, in the projected space")
    rep["a"] = []
    cases = [("path_b1", 0.45), ("skip2", 0.45), ("grid8x8", 0.24), ("star16", 0.24)]
    if a.quick:
        cases = cases[:2]
    for g, e in cases:
        r = test_a_coverage(g, e, quant, m, n_rep, rng)
        rep["a"].append(r)
        print(f"    {g:<10} eps={e:+.3f} -> target={r['target_edge']:+.4f} "
              f"(x{r['attenuation']:.4f})  cov={r['edge_coverage']:.3f} "
              f"[{r['edge_coverage_ci'][0]:.3f},{r['edge_coverage_ci'][1]:.3f}]  "
              f"FWER={r['fwer_rejection_rate']:.3f}  FDR-any={r['fdr_any_rejection_rate']:.3f}  "
              f"{'PASS' if r['pass'] else 'FAIL'}")

    print("\n(b) injected-leakage power, in the projected space")
    rep["b"] = test_b_power("path_b1", 0.45, quant, m, n_rep_b,
                            [0.0, 0.02, 0.04, 0.06, 0.10, 0.20], 4, rng)
    for row in rep["b"]["curve"]:
        print(f"    delta={row['delta']:.3f} (projected {row['delta_projected']:+.4f})  "
              f"power={row['power_injected']:.3f}  FPR={row['false_positive_rate']:.4f}")
    print(f"    monotone={rep['b']['monotone']} localized={rep['b']['localized']} "
          f"{'PASS' if rep['b']['pass_'] else 'FAIL'}")

    print("\n(c) AR(1) known-answer recovery, against the PROJECTED target")
    rep["c"] = test_c_ar1(32, [0.5, -0.5], quant, m, 6, n_rep_c, rng)
    for row in rep["c"]["rows"]:
        if row["lag"] <= 3:
            print(f"    rho={row['rho']:+.2f} lag={row['lag']}  latent={row['latent_target']:+.4f} "
                  f"-> target={row['target']:+.4f}  mean={row['mean']:+.4f}  "
                  f"{'ok' if row['covers'] else 'MISS'}")
    print(f"    {'PASS' if rep['c']['pass_'] else 'FAIL'}")

    print("\n(d) separator conditional-independence test in token space")
    rep["d"] = test_d_separator("path_b1", 0.45, quant, m, max(n_rep // 10, 5), rng)
    for k in (1, 3):
        s = rep["d"][f"pc{k}"]
        print(f"    condition on PC1..{k}:  GMRF (null TRUE) reject={s['gmrf']['rejection_rate']:.3f} "
              f"median|pcorr|={s['gmrf']['median_abs_pcorr']:.4f}   "
              f"EBF (null FALSE) reject={s['ebf']['rejection_rate']:.3f}  "
              f"{'PASS' if s['pass_'] else 'FAIL'}")
    print(f"    {'PASS' if rep['d']['pass_'] else 'FAIL'} (best: PC1..{rep['d']['best_pc']})")

    gate = (all(r["pass"] for r in rep["a"]) and rep["b"]["pass_"]
            and rep["c"]["pass_"] and rep["d"]["pass_"])
    rep["GM1_VQ"] = "PASS" if gate else "FAIL"
    rep["seconds"] = round(time.time() - t0, 1)
    with open(a.out, "w") as fh:
        json.dump(rep, fh, indent=2)
    print(f"\n=== GATE GM1-VQ: {rep['GM1_VQ']} ===  ({rep['seconds']}s, wrote {a.out})")
    if a.quick:
        print("    (quick mode -- smoke test only, NOT the gate)")


if __name__ == "__main__":
    main()
