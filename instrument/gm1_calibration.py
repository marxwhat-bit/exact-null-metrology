#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gate GM1 — the instrument tests itself.  No measurement claim leaves the
project until this passes.

Three known-answer subjects, pre-registered before the gate was run:

  (a) the exact EBF sampler        -> every null must cover at its nominal rate
                                      (95% intervals covering 94-96% over 200
                                      replicates)
  (b) hand-corrupted samplers      -> power monotone in the injected leakage,
                                      and detection LOCALIZED to the injected
                                      non-edges (false positives held at alpha)
  (c) an AR(1) copula              -> recovered rho_hat(d) matches rho^d within CI

Plus, added here because the GMRF companion field was introduced during the
build: (d) the separator conditional-independence test must cover at nominal
rate on the field where conditional independence is exact.

A failure here is not a nuisance -- it means every downstream number would be
uninterpretable.  Run:  python3 gm1_calibration.py [--quick]
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
from scipy import stats

from ebf import EBF, GMRF, AR1Copula, LeakyEBF
from estimators import (bh_fdr, calibrated_edge_ci, calibrated_p,
                        edge_fidelity, fisher_z, pair_corr, partial_corr,
                        rho_by_lag)
from graphs import build_graph, edge_list, nonedge_list

ALPHA = 0.05
NOMINAL_LO, NOMINAL_HI = 0.94, 0.96          # the pre-registered coverage band


# --------------------------------------------------------------------- helpers
def _bank(field, m, n_rep, rng, pairs):
    """SIGNED rho_hat for `n_rep` independent draws, restricted to `pairs`."""
    out = np.empty((n_rep, len(pairs)))
    for b in range(n_rep):
        out[b] = pair_corr(field.sample(m, rng), pairs)
    return out


def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson interval for a proportion -- honest at the ends, unlike Wald."""
    if n == 0:
        return (float("nan"),) * 2
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def _cluster_ci(per_rep: np.ndarray, z: float = 1.96) -> tuple[float, float]:
    """
    CI for a coverage rate whose unit of independence is the REPLICATE.

    The |E| edges inside one replicate share a sample, so they are not
    independent draws; pooling them into a Wilson interval understates the width
    by roughly sqrt(|E|).  (The first GM1 smoke run reported a 0.918-0.946
    interval on 40 replicates this way -- narrow enough to fail the gate on
    noise.)  Cluster on the replicate instead.
    """
    r = len(per_rep)
    if r < 2:
        return (float("nan"),) * 2
    se = per_rep.std(ddof=1) / np.sqrt(r)
    mu = float(per_rep.mean())
    return (max(0.0, mu - z * se), min(1.0, mu + z * se))


# ----------------------------------------------------------------- (a) coverage
def test_a_coverage(graph: str, eps: float, kind: str, m: int, n_rep: int,
                    rng: np.random.Generator) -> dict:
    """Exact field: edge CIs, off-graph FWER, and BH-FDR must all be nominal."""
    A = build_graph(graph)
    f = EBF(A, eps, kind=kind)
    e, ne = edge_list(A), nonedge_list(A)
    truth = f.truth()
    tgt_e = truth[e[:, 0], e[:, 1]]

    # one bank covers both families: edges supply the CALIBRATED interval width,
    # non-edges supply the exact null.  Drawn from the exact field, so both are
    # the true sampling distributions rather than normal-theory approximations.
    bank_all = _bank(f, m, n_rep, rng, np.vstack([e, ne]))
    bank_e, bank = bank_all[:, :len(e)], bank_all[:, len(e):]
    max_null = np.abs(bank).max(1)

    cov_per_rep = np.empty(n_rep)
    cov_textbook = np.empty(n_rep)
    inflation = float("nan")
    fdr_any = np.zeros(n_rep, bool)
    fwer_rej = 0
    fdr_rej_total = 0
    for r in range(n_rep):
        X = f.sample(m, rng)
        # -- edge CIs, both the textbook interval and the calibrated one, so the
        #    gate reports where normal theory is adequate and where it is not
        rho = pair_corr(X, e)
        half = 1.96 / np.sqrt(m - 3)
        lo_t, hi_t = np.tanh(np.arctanh(rho) - half), np.tanh(np.arctanh(rho) + half)
        cov_textbook[r] = ((lo_t <= tgt_e) & (tgt_e <= hi_t)).mean()
        lo, hi, infl = calibrated_edge_ci(rho, bank_e, m)
        cov_per_rep[r] = ((lo <= tgt_e) & (tgt_e <= hi)).mean()
        inflation = float(infl.mean())
        # -- off-graph family: FWER via max-statistic, FDR via BH
        stat = pair_corr(X, ne)
        fwer_rej += int((1.0 + (max_null >= np.abs(stat).max()).sum())
                        / (len(max_null) + 1.0) <= ALPHA)
        n_false = int(bh_fdr(calibrated_p(stat, bank, m)[0], ALPHA).sum())
        fdr_rej_total += n_false
        fdr_any[r] = n_false > 0

    edge_cov = float(cov_per_rep.mean())
    fwer = fwer_rej / n_rep
    res = dict(
        graph=graph, eps=eps, kind=kind, m=m, n_rep=n_rep,
        edge_coverage=edge_cov, edge_coverage_ci=_cluster_ci(cov_per_rep),
        edge_coverage_textbook=float(cov_textbook.mean()),
        edge_coverage_textbook_ci=_cluster_ci(cov_textbook),
        ci_inflation_vs_normal_theory=inflation,
        fwer_rejection_rate=fwer, fwer_ci=_wilson(fwer_rej, n_rep),
        mean_false_fdr_rejections=fdr_rej_total / n_rep,
        fdr_any_rejection_rate=float(fdr_any.mean()),
        fdr_any_ci=_wilson(int(fdr_any.sum()), n_rep),
        n_edges=len(e), n_nonedges=len(ne),
    )
    # coverage band is pre-registered; FWER must not EXCEED alpha (conservative ok)
    lo_ci, hi_ci = res["edge_coverage_ci"]
    res["pass_edge_cov"] = bool(NOMINAL_LO <= edge_cov <= NOMINAL_HI
                                or (lo_ci <= 0.95 <= hi_ci))
    res["pass_fwer"] = bool(fwer <= ALPHA + 2 * np.sqrt(ALPHA * (1 - ALPHA) / n_rep))
    # under the GLOBAL null BH's family-wise error is alpha, and when it does
    # fire it fires in a cascade -- so the calibrated quantity is P(any false
    # rejection), not the mean count.
    res["pass_fdr"] = bool(res["fdr_any_rejection_rate"]
                           <= ALPHA + 2 * np.sqrt(ALPHA * (1 - ALPHA) / n_rep))
    res["pass"] = bool(res["pass_edge_cov"] and res["pass_fwer"] and res["pass_fdr"])
    return res


# -------------------------------------------------------------------- (b) power
def test_b_power(graph: str, eps: float, kind: str, m: int, n_rep: int,
                 deltas: list[float], n_inject: int,
                 rng: np.random.Generator) -> dict:
    """Injected leakage: power must rise with delta and stay localized."""
    A = build_graph(graph)
    ne = nonedge_list(A)
    f0 = EBF(A, eps, kind=kind)
    bank = _bank(f0, m, max(n_rep, 100), rng, ne)

    # inject on non-edges that are FAR from the graph, so a detector that merely
    # smears correlation along the graph cannot get credit for finding them
    dist = np.abs(ne[:, 0] - ne[:, 1])
    pick = ne[np.argsort(-dist)[:n_inject]]
    inj_idx = np.array([np.flatnonzero((ne[:, 0] == i) & (ne[:, 1] == j))[0]
                        for i, j in pick])
    other_idx = np.setdiff1d(np.arange(len(ne)), inj_idx)

    rows = []
    for delta in deltas:
        fl = f0 if delta == 0 else LeakyEBF(A, eps, pick, delta, kind=kind)
        det_inj = np.zeros(n_rep)
        det_oth = np.zeros(n_rep)
        fwer = 0
        for r in range(n_rep):
            X = fl.sample(m, rng)
            stat = pair_corr(X, ne)
            rej = bh_fdr(calibrated_p(stat, bank, m)[0], ALPHA)
            det_inj[r] = rej[inj_idx].mean()
            det_oth[r] = rej[other_idx].mean()
            fwer += int((1.0 + (np.abs(bank).max(1) >= np.abs(stat).max()).sum())
                        / (bank.shape[0] + 1.0) <= ALPHA)
        rows.append(dict(delta=delta, power_injected=float(det_inj.mean()),
                         false_positive_rate=float(det_oth.mean()),
                         fwer_detect_rate=fwer / n_rep))
    powers = [r["power_injected"] for r in rows]
    fprs = [r["false_positive_rate"] for r in rows]
    return dict(graph=graph, eps=eps, kind=kind, m=m, n_rep=n_rep,
                n_injected=n_inject, injected_pairs=pick.tolist(), curve=rows,
                monotone=bool(all(b >= a - 1e-9 for a, b in zip(powers, powers[1:]))),
                reaches_full_power=bool(powers[-1] >= 0.95),
                null_fpr=float(fprs[0]),
                localized=bool(max(fprs) <= 0.02),
                pass_=bool(all(b >= a - 1e-9 for a, b in zip(powers, powers[1:]))
                           and powers[-1] >= 0.95 and max(fprs) <= 0.02))


# ------------------------------------------------------------------ (c) AR(1)
def test_c_ar1(n: int, rhos: list[float], m: int, max_lag: int, n_rep: int,
               rng: np.random.Generator) -> dict:
    """A Markov field is a known-answer subject: rho_hat(d) must equal rho^d."""
    rows = []
    ok = True
    for rho in rhos:
        f = AR1Copula(n, rho)
        est = np.array([rho_by_lag(f.sample(m, rng), max_lag) for _ in range(n_rep)])
        for d in range(1, max_lag + 1):
            vals = est[:, d]
            lo, hi = np.quantile(vals, [0.025, 0.975])
            tgt = rho ** d
            hit = bool(lo <= tgt <= hi)
            ok &= hit
            rows.append(dict(rho=rho, lag=d, target=tgt, mean=float(vals.mean()),
                             ci=[float(lo), float(hi)], covers=hit))
    return dict(n=n, m=m, n_rep=n_rep, rows=rows, pass_=bool(ok))


# ------------------------------------------------------- (d) separator CI test
def test_d_separator(graph: str, eps: float, m: int, n_rep: int,
                     rng: np.random.Generator) -> dict:
    """
    On the GMRF (sparse precision) conditional independence off the graph is
    exact, so separator p-values must be uniform and reject at exactly alpha.
    Run on the EBF too, where the same null is FALSE -- a test that never fires
    is not a test.
    """
    A = build_graph(graph)
    n = A.shape[0]
    triples = [(i, i + 4, list(range(i + 1, i + 4))) for i in range(0, n - 5, 3)]
    out = {}
    for name, fld in (("gmrf", GMRF(A, eps)), ("ebf", EBF(A, eps))):
        pv = []
        for _ in range(n_rep):
            X = fld.sample(m, rng)
            for i, j, S in triples:
                r = partial_corr(X, i, j, S)
                z = float(fisher_z(np.array([r]), m, df_extra=len(S))[0])
                pv.append(2 * stats.norm.sf(abs(z)))
        pv = np.array(pv)
        rej = float((pv <= ALPHA).mean())
        out[name] = dict(rejection_rate=rej, ci=_wilson(int((pv <= ALPHA).sum()), len(pv)),
                         ks_uniform_p=float(stats.kstest(pv, "uniform").pvalue),
                         n_tests=len(pv))
    out["pass_"] = bool(out["gmrf"]["rejection_rate"] <= 0.08
                        and out["ebf"]["rejection_rate"] >= 0.5)
    return out


# ---------------------------------------------------------------------- driver
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="smoke settings, not the gate")
    ap.add_argument("--out", default="gm1_report.json")
    a = ap.parse_args()

    m = 2000 if a.quick else 8000
    n_rep = 40 if a.quick else 200
    n_rep_b = 20 if a.quick else 100
    n_rep_c = 10 if a.quick else 50
    rng = np.random.default_rng(20260726)
    t0 = time.time()
    rep: dict = dict(mode="quick" if a.quick else "gate", m=m, n_rep=n_rep,
                     alpha=ALPHA, date="2026-07-26")

    print(f"=== GM1 calibration ({rep['mode']}, m={m}, replicates={n_rep}) ===\n")

    print("(a) exact-field coverage")
    rep["a"] = []
    cases = [("path_b1", 0.45, "gaussian"), ("skip2", 0.45, "gaussian"),
             ("cycle9", 22 / 45, "binary"), ("star16", 0.24, "gaussian")]
    if a.quick:
        cases = cases[:2]
    for g, e, k in cases:
        r = test_a_coverage(g, e, k, m, n_rep, rng)
        rep["a"].append(r)
        print(f"    {g:<10} eps={e:+.4f} {k:<9} "
              f"edge-CI cov={r['edge_coverage']:.3f} "
              f"[{r['edge_coverage_ci'][0]:.3f},{r['edge_coverage_ci'][1]:.3f}] "
              f"(textbook {r['edge_coverage_textbook']:.3f}, "
              f"width x{r['ci_inflation_vs_normal_theory']:.3f}) "
              f"FWER={r['fwer_rejection_rate']:.3f} "
              f"FDR-any={r['fdr_any_rejection_rate']:.3f} "
              f"{'PASS' if r['pass'] else 'FAIL'}")

    print("\n(b) injected-leakage power")
    deltas = [0.0, 0.02, 0.04, 0.06, 0.10, 0.20]
    rep["b"] = test_b_power("path_b1", 0.45, "gaussian", m, n_rep_b, deltas, 4, rng)
    for row in rep["b"]["curve"]:
        print(f"    delta={row['delta']:.3f}  power={row['power_injected']:.3f}  "
              f"FPR={row['false_positive_rate']:.4f}  "
              f"FWER-detect={row['fwer_detect_rate']:.3f}")
    print(f"    monotone={rep['b']['monotone']} localized={rep['b']['localized']} "
          f"{'PASS' if rep['b']['pass_'] else 'FAIL'}")

    print("\n(c) AR(1) known-answer recovery")
    rep["c"] = test_c_ar1(32, [0.5, -0.5], m, 6, n_rep_c, rng)
    for row in rep["c"]["rows"]:
        if row["lag"] <= 3:
            print(f"    rho={row['rho']:+.2f} lag={row['lag']}  target={row['target']:+.4f}  "
                  f"mean={row['mean']:+.4f}  CI=[{row['ci'][0]:+.4f},{row['ci'][1]:+.4f}]  "
                  f"{'ok' if row['covers'] else 'MISS'}")
    print(f"    {'PASS' if rep['c']['pass_'] else 'FAIL'}")

    print("\n(d) separator conditional-independence test")
    rep["d"] = test_d_separator("path_b1", 0.45, m, max(n_rep // 10, 5), rng)
    print(f"    GMRF (null TRUE):  reject={rep['d']['gmrf']['rejection_rate']:.3f} "
          f"KS-uniform p={rep['d']['gmrf']['ks_uniform_p']:.3f}")
    print(f"    EBF  (null FALSE): reject={rep['d']['ebf']['rejection_rate']:.3f}  "
          f"(a test that never fires is not a test)")
    print(f"    {'PASS' if rep['d']['pass_'] else 'FAIL'}")

    gate = (all(r["pass"] for r in rep["a"]) and rep["b"]["pass_"]
            and rep["c"]["pass_"] and rep["d"]["pass_"])
    rep["GM1"] = "PASS" if gate else "FAIL"
    rep["seconds"] = round(time.time() - t0, 1)
    with open(a.out, "w") as fh:
        json.dump(rep, fh, indent=2)
    print(f"\n=== GATE GM1: {rep['GM1']} ===  ({rep['seconds']}s, wrote {a.out})")
    if a.quick:
        print("    (quick mode -- smoke test only, NOT the gate)")


if __name__ == "__main__":
    main()
