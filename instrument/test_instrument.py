#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Contract tests for the instrument.  Standing rule: every new sampler/estimator
gets bit-identity nulls, marginal exactness and monotonicity checks before any
number it produces is trusted.

These are cheap and run in seconds -- they are the thing you run after every
edit, whereas gm1_calibration.py is the gate you run before claiming anything.

Run:  python3 test_instrument.py
"""
from __future__ import annotations

import sys

import numpy as np
from scipy import stats

from bound import (independent_set_schedule, oneshot_schedule, random_schedule,
                   sequential_schedule, thm1_floor, total_correlation_gaussian)
from ebf import (AR1Copula, EBF, GMRF, LeakyEBF, binary_corr_from_latent,
                 qary_corr_from_latent, target_corr)
from estimators import (bh_fdr, calibrated_edge_ci, edge_kl, leakage_test,
                        pair_corr, partial_corr, rho_by_lag)
from graphs import (build_graph, edge_list, eps_window, GRAPH_ZOO,
                    mve_binary_extreme, nonedge_list, spectral_ceiling)

PASS, FAIL = "  ok  ", " FAIL "
_failures: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"[{PASS if cond else FAIL}] {name}" + (f"   {detail}" if detail else ""))
    if not cond:
        _failures.append(name)


# --------------------------------------------------------------- 1. exact nulls
def test_eps0_bit_identity() -> None:
    """eps=0 must reproduce the iid field BIT-identically, not approximately."""
    A = build_graph("path_b1")
    s1 = np.random.default_rng(11).standard_normal((512, A.shape[0]))
    s2 = EBF(A, 0.0).sample(512, np.random.default_rng(11))
    check("eps=0 is bit-identical to iid", np.array_equal(s1, s2),
          f"max|diff|={np.abs(s1 - s2).max():.3e}")


def test_target_is_zero_off_graph() -> None:
    """The ground truth must be EXACTLY zero off the graph in all three regimes."""
    A = build_graph("skip2")
    ne = nonedge_list(A)
    for kind in ("gaussian", "binary", "qary"):
        T = target_corr(A, 0.45, kind)
        check(f"target off-graph == 0 exactly [{kind}]",
              np.all(T[ne[:, 0], ne[:, 1]] == 0.0))


def test_gmrf_partial_zero_off_graph() -> None:
    """The GMRF's exact partial correlations must be identically zero off-graph."""
    A = build_graph("path_b1")
    P = GMRF(A, 0.45).partial_truth()
    ne = nonedge_list(A)
    check("GMRF partial corr off-graph == 0 exactly",
          float(np.abs(P[ne[:, 0], ne[:, 1]]).max()) < 1e-12)


# ------------------------------------------------------ 2. marginal exactness
def test_marginals() -> None:
    rng = np.random.default_rng(3)
    A = build_graph("path_b1")
    X = EBF(A, 0.45, kind="gaussian").sample(20000, rng)
    ks = max(stats.kstest(X[:, i], "norm").statistic for i in range(A.shape[0]))
    check("gaussian marginals exact (KS)", ks < 0.02, f"max KS={ks:.4f}")

    for p in (0.3, 0.5):
        Xb = EBF(A, 0.45, kind="binary", p=p).sample(40000, rng)
        err = float(np.abs(Xb.mean(0) - p).max())
        check(f"binary marginal rate exact (p={p})", err < 0.01, f"max err={err:.4f}")

    q = 4
    Xq = EBF(A, 0.45, kind="qary", q=q).sample(40000, rng)
    freq = np.array([[np.mean(Xq[:, i] == v) for v in range(q)]
                     for i in range(A.shape[0])])
    err = float(np.abs(freq - 1.0 / q).max())
    check("q-ary bin frequencies exact", err < 0.012, f"max err={err:.4f}")


def test_edge_targets_recovered() -> None:
    """Measured edge correlation must match the closed-form target in each regime."""
    rng = np.random.default_rng(5)
    A = build_graph("path_b1")
    e = edge_list(A)
    for kind, tol in (("gaussian", 0.01), ("binary", 0.01), ("qary", 0.01)):
        f = EBF(A, 0.45, kind=kind)
        rho = float(pair_corr(f.sample(200000, rng), e).mean())
        tgt = float(f.truth()[e[0, 0], e[0, 1]])
        check(f"edge rho matches closed form [{kind}]", abs(rho - tgt) < tol,
              f"{rho:+.4f} vs {tgt:+.4f}")


# ---------------------------------------------------------- 3. monotonicity
def test_monotone_in_eps() -> None:
    rng = np.random.default_rng(7)
    A = build_graph("path_b1")
    e = edge_list(A)
    lo, hi = eps_window(A)
    eps_grid = np.linspace(0.95 * lo, 0.95 * hi, 9)
    rhos = [float(pair_corr(EBF(A, float(ep)).sample(20000, rng), e).mean())
            for ep in eps_grid]
    check("edge rho monotone across the full signed eps window",
          all(b > a for a, b in zip(rhos, rhos[1:])),
          f"{rhos[0]:+.3f} -> {rhos[-1]:+.3f}")


def test_leakage_power_monotone() -> None:
    rng = np.random.default_rng(9)
    A = build_graph("path_b1")
    ne = nonedge_list(A)
    pick = ne[np.argsort(-np.abs(ne[:, 0] - ne[:, 1]))[:3]]
    f0 = EBF(A, 0.45)
    stats_ = []
    for delta in (0.0, 0.05, 0.15):
        f = f0 if delta == 0 else LeakyEBF(A, 0.45, pick, delta)
        r = leakage_test(f.sample(8000, rng), A, field=f0, n_null=60, rng=rng)
        stats_.append(r.max_stat)
    check("leakage statistic monotone in injected delta",
          all(b > a for a, b in zip(stats_, stats_[1:])),
          " -> ".join(f"{s:.4f}" for s in stats_))


# ------------------------------------------------------------- 4. estimators
def test_exact_field_not_flagged() -> None:
    """The exact field must not be accused of leaking."""
    rng = np.random.default_rng(13)
    A = build_graph("path_b1")
    f = EBF(A, 0.45)
    r = leakage_test(f.sample(8000, rng), A, field=f, n_null=100, rng=rng)
    check("exact field: no FDR rejections", int(r.rejected_fdr.sum()) == 0,
          r.summary())
    check("exact field: null scale ~ 1", abs(r.null_scale - 1.0) < 0.1,
          f"scale={r.null_scale:.4f}")


def test_leaky_field_localized() -> None:
    """
    Detection must find the injected pairs and stay close to them.

    Note what is asserted and what is not.  BH controls the EXPECTED false
    discovery proportion, not the per-run count, so "zero extra detections in
    this run" is not a property the procedure has -- asserting it (as the first
    version of this test did) produces a flaky test that fails on a correct
    implementation.  Averaging FDP over replicates is the assertion that matches
    the guarantee.
    """
    rng = np.random.default_rng(17)
    A = build_graph("path_b1")
    ne = nonedge_list(A)
    pick = ne[np.argsort(-np.abs(ne[:, 0] - ne[:, 1]))[:3]]
    want = {tuple(x) for x in pick}
    f0 = EBF(A, 0.45)
    f = LeakyEBF(A, 0.45, pick, 0.15)
    fdp, power = [], []
    for _ in range(12):
        det = {tuple(x) for x in
               leakage_test(f.sample(8000, rng), A, field=f0, n_null=100,
                            rng=rng).detected}
        fdp.append(len(det - want) / max(len(det), 1))
        power.append(len(det & want) / len(want))
    check("injected leakage detected", float(np.mean(power)) > 0.95,
          f"power={np.mean(power):.3f}")
    check("false discoveries stay near the BH level", float(np.mean(fdp)) <= 0.15,
          f"mean FDP={np.mean(fdp):.4f} over {len(fdp)} replicates")


def test_injection_preserves_exact_zeros() -> None:
    """
    Injecting leakage must not disturb the rest of the null.

    If a nearest-PSD repair ever fired it would rewrite every entry of the
    covariance, silently destroying the exact off-graph zeros while the leakage
    test kept reporting p-values against a null that no longer held.  The code
    refuses instead of repairing; this pins that down.
    """
    A = build_graph("path_b1")
    ne = nonedge_list(A)
    pick = ne[np.argsort(-np.abs(ne[:, 0] - ne[:, 1]))[:3]]
    f = LeakyEBF(A, 0.45, pick, 0.3)
    mask = (A == 0) & (f.B == 0) & ~np.eye(A.shape[0], dtype=bool)
    check("injection leaves every other non-edge exactly zero",
          float(np.abs(f.Sigma[mask]).max()) == 0.0,
          f"max|off (A∪B)|={np.abs(f.Sigma[mask]).max():.3e}")


def test_infeasible_eps_raises() -> None:
    A = build_graph("path_b1")
    hi = spectral_ceiling(A)
    raised = False
    try:
        EBF(A, hi * 1.2)
    except ValueError:
        raised = True
    check("eps outside the spectral window raises rather than being repaired",
          raised, f"ceiling={hi:.4f}")


def test_bh_fdr() -> None:
    # BH is a step-UP procedure: find the LARGEST k with p_(k) <= alpha*k/m and
    # reject all k hypotheses below it.  Here m=5, alpha=0.05, thresholds
    # 0.01/0.02/0.03/0.04/0.05, and p_(3)=0.02 <= 0.03 -- so three rejections,
    # not two.  (The first version of this test asserted two and was itself the
    # thing that was wrong; kept as a comment because the step-up/step-down
    # confusion is the standard way to get BH subtly wrong.)
    p = np.array([0.001, 0.008, 0.02, 0.2, 0.5])
    got = bh_fdr(p, 0.05)
    check("BH-FDR matches the hand-computed answer",
          np.array_equal(got, np.array([True, True, True, False, False])),
          str(got.astype(int)))
    check("BH-FDR rejects nothing when all p are large",
          not bh_fdr(np.linspace(0.3, 0.9, 50), 0.05).any())


def test_ar1_recovery() -> None:
    rng = np.random.default_rng(19)
    f = AR1Copula(32, 0.5)
    r = rho_by_lag(f.sample(50000, rng), 4)
    err = max(abs(r[d] - 0.5 ** d) for d in range(1, 5))
    check("AR(1) rho^d recovered at every lag", err < 0.01, f"max err={err:.4f}")


def test_separator_test_fires_and_holds() -> None:
    rng = np.random.default_rng(23)
    A = build_graph("path_b1")
    i, j, S = 5, 9, [6, 7, 8]
    pc_gmrf = abs(partial_corr(GMRF(A, 0.45).sample(50000, rng), i, j, S))
    pc_ebf = abs(partial_corr(EBF(A, 0.45).sample(50000, rng), i, j, S))
    check("GMRF: conditional independence holds across a separator", pc_gmrf < 0.02,
          f"|pcorr|={pc_gmrf:.4f}")
    check("EBF: the same test correctly FIRES (it is not inert)", pc_ebf > 0.05,
          f"|pcorr|={pc_ebf:.4f}")


def test_calibrated_ci_widens_where_needed() -> None:
    """Binary regime must need a wider interval than normal theory gives."""
    rng = np.random.default_rng(29)
    A = build_graph("cycle9")
    e = edge_list(A)
    m = 8000
    f = EBF(A, 22 / 45, kind="binary")
    bank = np.array([pair_corr(f.sample(m, rng), e) for _ in range(120)])
    _, _, infl = calibrated_edge_ci(pair_corr(f.sample(m, rng), e), bank, m)
    check("binary C9 needs wider-than-normal-theory intervals",
          float(infl.mean()) > 1.02, f"inflation x{float(infl.mean()):.3f}")

    fg = EBF(build_graph("path_b1"), 0.45)
    eg = edge_list(build_graph("path_b1"))
    bankg = np.array([pair_corr(fg.sample(m, rng), eg) for _ in range(120)])
    _, _, inflg = calibrated_edge_ci(pair_corr(fg.sample(m, rng), eg), bankg, m)
    check("gaussian regime needs no inflation", abs(float(inflg.mean()) - 1) < 0.05,
          f"inflation x{float(inflg.mean()):.3f}")


# ----------------------------------------------------------------- 5. the bound
def test_floor_endpoints() -> None:
    A = build_graph("path_b1")
    n = A.shape[0]
    Sigma = np.eye(n) + 0.45 * A
    seq = thm1_floor(Sigma, sequential_schedule(n))["floor_nats"]
    one = thm1_floor(Sigma, oneshot_schedule(n))["floor_nats"]
    check("sequential schedule has floor exactly 0", abs(seq) < 1e-12, f"{seq:.3e}")
    check("one-shot floor == total correlation of the field",
          abs(one - total_correlation_gaussian(Sigma)) < 1e-9, f"{one:.4f}")


def test_floor_monotone_in_T() -> None:
    """ParallelBench Thm 2: the floor decreases as the number of steps grows."""
    rng = np.random.default_rng(31)
    A = build_graph("path_b1")
    n = A.shape[0]
    Sigma = np.eye(n) + 0.45 * A
    vals = [thm1_floor(Sigma, random_schedule(n, k, rng))["floor_nats"]
            for k in (16, 8, 4, 2, 1)]
    check("floor decreases with more steps (Thm 2)",
          all(b <= a + 1e-9 for a, b in zip(vals, vals[1:])),
          " >= ".join(f"{v:.3f}" for v in vals))


def test_floor_nonnegative_everywhere() -> None:
    rng = np.random.default_rng(37)
    bad = []
    for name in GRAPH_ZOO:
        A = build_graph(name)
        n = A.shape[0]
        eps = 0.9 * spectral_ceiling(A)
        Sigma = np.eye(n) + eps * A
        for sched in (random_schedule(n, 4, rng), independent_set_schedule(A, 4)):
            v = thm1_floor(Sigma, sched)["floor_nats"]
            if v < -1e-9 or not np.isfinite(v):
                bad.append((name, v))
    check("floor is finite and non-negative across the whole zoo", not bad, str(bad))


# -------------------------------------------------------------- 6. theory box
def test_theory_values() -> None:
    check("MVE(C9) = 22/45 exactly",
          abs(mve_binary_extreme("odd_cycle", n=9) - 22 / 45) < 1e-15)
    check("C9 binary extreme is strictly below its spectral ceiling",
          mve_binary_extreme("odd_cycle", n=9) < spectral_ceiling(build_graph("cycle9")),
          f"{22/45:.4f} < {spectral_ceiling(build_graph('cycle9')):.4f}")
    check("star K_{1,16} ceiling = 1/sqrt(16) = 0.25",
          abs(mve_binary_extreme("star", m=16) - 0.25) < 1e-12)
    A = build_graph("path_b1")
    check("eps window is symmetric for a bipartite band graph",
          abs(sum(eps_window(A))) < 1e-9, str(eps_window(A)))
    check("shuffled control preserves the spectrum exactly",
          np.allclose(np.linalg.eigvalsh(build_graph("path_b1")),
                      np.linalg.eigvalsh(build_graph("path_b1_shuf"))))


# ------------------------------------- reverse KL sees what forward KL cannot
def test_rev_kl_zero_on_exact_sampler() -> None:
    """Both directions must vanish on the exact field, in all three regimes."""
    A = build_graph("path_b1")
    rng = np.random.default_rng(5)
    for kind in ("gaussian", "binary", "qary"):
        f = EBF(A, 0.45, kind=kind)
        r = edge_kl(f.sample(200_000, rng), A, f)
        check(f"{kind}: exact sampler -> both KL directions ~ 0",
              r["edge_kl_rev"] < 2e-4 and r["nonedge_kl_rev"] < 2e-4
              and r["edge_kl_fwd"] < 2e-4,
              f"edge_rev={r['edge_kl_rev']:.2e} nonedge_rev={r['nonedge_kl_rev']:.2e}")


def test_fwd_kl_is_blind_to_missing_support() -> None:
    """
    The demonstration the reverse-KL statistic exists for (cf. arXiv:2602.00286).

    A rigid sampler collapses onto a SUBSET of the true support.  Forward
    KL(sampler||true) only sums where the sampler has mass, so it stays small
    and finite -- it literally cannot see the missing configurations.  Reverse
    KL(true||sampler) puts true mass on them and diverges.  If this test ever
    passes with a finite reverse KL, the estimator has been silently smoothed
    and every rigid-instance claim built on it is unsupported.
    """
    A = build_graph("path_b1")
    f = EBF(A, 0.45, kind="binary")
    X = f.sample(200_000, np.random.default_rng(7))
    # rigid collapse: force every edge pair to agree (drop the (0,1)/(1,0) cells)
    Xr = X.copy()
    Xr[:, 1::2] = Xr[:, 0::2][:, :Xr[:, 1::2].shape[1]]
    r = edge_kl(Xr, A, f)
    fwd_finite_small = np.isfinite(r["edge_kl_fwd"]) and r["edge_kl_fwd"] < 1.0
    rev_diverges = not np.isfinite(r["edge_kl_rev"])
    check("rigid collapse: forward KL stays finite and small (it is BLIND)",
          fwd_finite_small, f"edge_kl_fwd={r['edge_kl_fwd']:.4f}")
    check("rigid collapse: reverse KL diverges (it SEES the missing support)",
          rev_diverges, f"edges with missing support={r['n_edges_missing_support']}"
                        f"/{r['n_edges']}")


def test_rev_kl_monotone_in_corruption() -> None:
    """Reverse KL must grow monotonically as the sampler is pushed off target."""
    A = build_graph("path_b1")
    f = EBF(A, 0.45, kind="gaussian")
    rng = np.random.default_rng(9)
    prev, ok = -1.0, True
    vals = []
    for w in (0.0, 0.1, 0.25, 0.5, 1.0):
        # blend the exact field with iid noise: w=0 exact, w=1 fully independent
        X = (1 - w) * f.sample(120_000, np.random.default_rng(3)) + \
            w * EBF(A, 0.0).sample(120_000, np.random.default_rng(4))
        v = edge_kl(X, A, f)["edge_kl_rev"]
        vals.append(v)
        ok &= v >= prev - 1e-9
        prev = v
    check("reverse KL is monotone in corruption strength", ok,
          " -> ".join(f"{v:.4f}" for v in vals))


def test_rev_kl_nonedge_is_the_exact_null() -> None:
    """Off-graph KL must be ~0 under the exact field and fire under leakage."""
    A = build_graph("path_b1")
    f = EBF(A, 0.30, kind="gaussian")
    rng = np.random.default_rng(13)
    clean = edge_kl(f.sample(150_000, rng), A, f)["nonedge_kl_rev"]
    inject = np.array([[0, 20], [1, 21], [2, 22], [3, 23]])
    lk = LeakyEBF(A, 0.30, inject=inject, delta=0.15)
    dirty = edge_kl(lk.sample(150_000, rng), A, f)["nonedge_kl_rev"]
    check("off-graph reverse KL ~ 0 on the exact field, fires under leakage",
          clean < 1e-3 and dirty > 10 * max(clean, 1e-6),
          f"clean={clean:.2e} leaky={dirty:.2e}")


def main() -> int:
    print("=== instrument contract tests ===\n")
    for fn in (test_eps0_bit_identity, test_target_is_zero_off_graph,
               test_gmrf_partial_zero_off_graph, test_marginals,
               test_edge_targets_recovered, test_monotone_in_eps,
               test_leakage_power_monotone, test_exact_field_not_flagged,
               test_leaky_field_localized, test_injection_preserves_exact_zeros,
               test_infeasible_eps_raises, test_bh_fdr, test_ar1_recovery,
               test_separator_test_fires_and_holds,
               test_calibrated_ci_widens_where_needed, test_floor_endpoints,
               test_floor_monotone_in_T, test_floor_nonnegative_everywhere,
               test_theory_values,
               test_rev_kl_zero_on_exact_sampler,
               test_fwd_kl_is_blind_to_missing_support,
               test_rev_kl_monotone_in_corruption,
               test_rev_kl_nonedge_is_the_exact_null):
        fn()
    print(f"\n{'ALL CONTRACT TESTS PASS' if not _failures else 'FAILURES: ' + ', '.join(_failures)}")
    return 1 if _failures else 0


if __name__ == "__main__":
    sys.exit(main())
