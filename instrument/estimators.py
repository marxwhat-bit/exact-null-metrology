#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The estimators -- the instrument proper.

Four measurements, all resolved per edge, all with a null that is true by
construction rather than by assumption:

  edge fidelity     err_E = mean_e |rho_hat_e - rho_target_e|   (how well the
                    sampler reproduces the dependence that IS there)
  off-graph leakage  the map Lambda over non-edges, where the truth is exactly
                    zero, so any detected dependence was manufactured by the
                    sampler under test -- tested with FWER (max-statistic) and
                    FDR (Benjamini-Hochberg) control
  separator tests   partial correlation given a graph separator: conditional
                    independence, the property DAPD's Limitations section names
                    as the thing attention proxies cannot test
  co-commit deficit  per-edge correlation loss conditioned on whether the two
                    endpoints were committed in the SAME sampler step -- the
                    mechanism variable that explains where the loss comes from

Why two nulls, and why we report both
-------------------------------------
`exact_null` simulates the *known* ground-truth field: it is the principled
null, valid because we built the truth.  `perm_null` destroys all dependence by
permuting each coordinate independently: assumption-light, but it also destroys
the on-graph dependence, so its calibration is not identical.  Reporting both
makes the difference auditable instead of a modelling choice hidden in a
p-value.  GM1(a) measures whether each one actually covers at its nominal rate.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from graphs import edge_list, nonedge_list

__all__ = ["pair_kl_categorical", "pair_kl_gaussian", "edge_kl",
           "conditional_null_map",
           "fisher_z", "pair_corr", "edge_fidelity", "leakage_map",
           "exact_null_stats", "perm_null_stats", "calibrated_p",
           "calibrated_edge_ci", "bh_fdr",
           "LeakageReport", "leakage_test", "partial_corr", "separator_test",
           "rho_by_lag", "co_commit_deficit"]


# ------------------------------------------------------------------ basic stats
def pair_corr(X: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    """Pearson correlation for a list of (i,j) pairs, computed once from X."""
    Xc = X - X.mean(0, keepdims=True)
    sd = Xc.std(0)
    sd = np.where(sd < 1e-12, np.inf, sd)          # constant coordinate -> corr 0
    Xs = Xc / sd
    i, j = pairs[:, 0], pairs[:, 1]
    return (Xs[:, i] * Xs[:, j]).mean(0)


def fisher_z(rho: np.ndarray, m: int, df_extra: int = 0) -> np.ndarray:
    """Variance-stabilised statistic; ~N(0,1) under independence for Gaussian X."""
    r = np.clip(rho, -0.999999, 0.999999)
    return np.arctanh(r) * np.sqrt(max(m - 3 - df_extra, 1))


# ---------------------------------------------------------------- edge fidelity
def edge_fidelity(X: np.ndarray, A: np.ndarray, truth: np.ndarray,
                  n_boot: int = 200, rng: np.random.Generator | None = None) -> dict:
    """
    err_E = mean_e |rho_hat_e - rho_target_e|, plus the signed bias and a
    bootstrap CI.  Signed bias matters on its own: a parallel sampler that
    commits dependent tokens concurrently should *under*-produce edge
    correlation, so the sign is a mechanism check, not decoration.
    """
    e = edge_list(A)
    rho = pair_corr(X, e)
    tgt = truth[e[:, 0], e[:, 1]]
    err = np.abs(rho - tgt)
    out = dict(err_E=float(err.mean()), bias=float((rho - tgt).mean()),
               rho_mean=float(rho.mean()), target_mean=float(tgt.mean()),
               n_edges=len(e))
    if n_boot:
        rng = rng or np.random.default_rng(0)
        m = X.shape[0]
        boots = np.empty(n_boot)
        for b in range(n_boot):
            idx = rng.integers(0, m, m)
            boots[b] = np.abs(pair_corr(X[idx], e) - tgt).mean()
        out["err_E_ci"] = (float(np.quantile(boots, 0.025)),
                           float(np.quantile(boots, 0.975)))
    return out


# --------------------------------------------------------------- leakage & nulls
def leakage_map(X: np.ndarray, A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Non-edge pairs and their observed (signed) rho_hat.  Truth is exactly 0."""
    ne = nonedge_list(A)
    return ne, pair_corr(X, ne)


_NULL_BANK: dict = {}


def exact_null_stats(field, m: int, n_null: int, rng: np.random.Generator,
                     pairs: np.ndarray, cache: bool = True) -> np.ndarray:
    """
    (n_null, n_pairs) SIGNED rho_hat under the EXACT ground-truth field.

    Signed, not absolute: the null scale is estimated from these values, and a
    scale read off folded (absolute) statistics is wrong by a constant factor.

    Simulating the null is legitimate here precisely because the construction
    makes non-edges exactly independent: this is the null, not a model of it.

    CACHED (2026-07-27).  The null distribution is a property of the FIELD and
    the sample size -- it does not depend on the sampler under test.  A sweep
    that re-simulates it for all 25 (scheduler x K) cells of a unit is doing the
    identical computation 25 times: profiling put `leakage_test` at 24.9 s of a
    ~40 s cell, i.e. ~63% of the whole trained-model sweep, essentially all of it
    here.  Caching is a correction to duplicated work, NOT an approximation --
    the returned bank is bit-identical to the first call's.  Key includes the
    field's tag, its graph, m, n_null and the pair set, so two different fields
    can never share a bank.  Pass cache=False to force a fresh draw.
    """
    key = (getattr(field, "tag", repr(field)),
           hash(np.asarray(field.A).tobytes()), m, n_null, hash(pairs.tobytes()))
    if cache and key in _NULL_BANK:
        return _NULL_BANK[key]
    out = np.empty((n_null, len(pairs)))
    for b in range(n_null):
        out[b] = pair_corr(field.sample(m, rng), pairs)
    if cache:
        _NULL_BANK[key] = out
    return out


def perm_null_stats(X: np.ndarray, n_null: int, rng: np.random.Generator,
                    pairs: np.ndarray) -> np.ndarray:
    """
    Assumption-light null: permute every coordinate's sample order independently.

    This destroys ALL dependence (edges included), so it is a null for complete
    independence, not for "independence off the graph".  Kept because a user of
    the instrument who cannot simulate the truth still needs a null -- and
    because the gap between the two is worth publishing rather than hiding.
    """
    m, n = X.shape
    out = np.empty((n_null, len(pairs)))
    for b in range(n_null):
        Xp = np.take_along_axis(X, rng.argsort(rng.random((m, n)), axis=0), axis=0)
        out[b] = pair_corr(Xp, pairs)
    return out


def calibrated_p(rho_obs: np.ndarray, rho_bank: np.ndarray, m: int
                 ) -> tuple[np.ndarray, float]:
    """
    Per-non-edge p-values for a family large enough that BH needs a deep tail.

    Two failure modes were hit building this, both recorded because they will
    bite anyone reusing the instrument:

    1. *Pure Monte-Carlo p-values have no resolution.*  p >= 1/(B+1), so with
       B=200 the smallest attainable p is ~0.005 while BH's threshold for the
       most significant of P non-edges is alpha/P (1e-4 at P=500).  BH can then
       never reject at any leakage strength -- the test silently has zero power.
    2. *Pooling the bank to buy resolution moves the noise, it does not remove
       it.*  Pooling B x P null values gives resolution 1/(BP+1), but the 1e-4
       quantile is then estimated from the handful of most extreme pooled draws;
       that estimate is unstable, and when it lands low BH rejects a cascade.
       Measured on the exact field: ~2 false rejections per replicate.

    So: estimate a SCALE nonparametrically (stable -- it uses the whole bank)
    and take the TAIL analytically.  Under the exact null, Fisher-z of a
    non-edge correlation is N(0, 1) after the sqrt(m-3) scaling for Gaussian
    data; for binary/q-ary subjects the scale departs from 1, and the bank
    measures by how much.  Returns (p-values, fitted null scale) -- the scale is
    reported, not hidden, because a scale far from 1 says the subject's regime
    is not the one the analytic tail assumes.
    """
    from scipy.stats import norm as _norm
    z_obs = fisher_z(np.asarray(rho_obs), m)
    z_bank = fisher_z(np.asarray(rho_bank).ravel(), m)
    s = float(z_bank.std())
    s = s if s > 1e-12 else 1.0
    return 2.0 * _norm.sf(np.abs(z_obs) / s), s


def calibrated_edge_ci(rho_obs: np.ndarray, bank_edge: np.ndarray,
                       m: int, alpha: float = 0.05
                       ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Confidence intervals for edge correlations, calibrated against the exact field.

    The textbook Fisher-z interval assumes bivariate NORMALITY: sd(atanh(rho_hat))
    = 1/sqrt(m-3).  That assumption is false for the binary and q-ary regimes,
    and GM1 caught it -- binary C_9 at the extremal eps = 22/45 covered 0.931
    [0.920, 0.942] over 200 replicates, i.e. genuinely below nominal, while the
    Gaussian cases sat at 0.950.  An instrument whose intervals are 2 points
    too narrow in one of its three regimes cannot be used to declare a sampler
    unbiased in that regime.

    The fix uses the instrument's own defining property: the truth is known, so
    the sampling distribution of rho_hat UNDER that truth can be simulated
    exactly rather than assumed.  `bank_edge` is (B, |E|) signed rho_hat from the
    exact field; the per-edge sd of atanh(rho_hat) replaces 1/sqrt(m-3).

    Returns (lo, hi, inflation) where inflation = calibrated sd relative to the
    normal-theory sd -- 1.0 means the textbook interval was right, and how far
    it departs is itself a reportable property of the regime.
    """
    from scipy.stats import norm as _norm
    zc = float(_norm.ppf(1 - alpha / 2))
    zeta = np.arctanh(np.clip(rho_obs, -0.999999, 0.999999))
    sd = np.arctanh(np.clip(bank_edge, -0.999999, 0.999999)).std(0, ddof=1)
    return (np.tanh(zeta - zc * sd), np.tanh(zeta + zc * sd),
            sd * np.sqrt(m - 3))


def bh_fdr(pvals: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Benjamini-Hochberg step-up; returns the boolean rejection vector."""
    m = len(pvals)
    order = np.argsort(pvals)
    thresh = alpha * np.arange(1, m + 1) / m
    passed = pvals[order] <= thresh
    rej = np.zeros(m, bool)
    if passed.any():
        rej[order[:np.flatnonzero(passed)[-1] + 1]] = True
    return rej


@dataclass
class LeakageReport:
    pairs: np.ndarray
    stat: np.ndarray                 # observed |rho_hat| per non-edge
    pval: np.ndarray                 # per-non-edge p, calibrated on the null
    rejected_fdr: np.ndarray         # BH-FDR rejections
    max_stat: float
    max_pval: float                  # FWER p-value (max-statistic null)
    null_kind: str
    n_null: int
    alpha: float
    null_scale: float = 1.0
    detected: np.ndarray = field(default_factory=lambda: np.empty((0, 2), int))

    def summary(self) -> str:
        return (f"leakage[{self.null_kind}] max|rho|={self.max_stat:.4f} "
                f"(null scale {self.null_scale:.3f}) "
                f"p_FWER={self.max_pval:.4f}  "
                f"BH-FDR rejects {int(self.rejected_fdr.sum())}/{len(self.pairs)} "
                f"non-edges at alpha={self.alpha}")


def leakage_test(X: np.ndarray, A: np.ndarray, *, null: str = "exact",
                 field=None, n_null: int = 200, alpha: float = 0.05,
                 rng: np.random.Generator | None = None) -> LeakageReport:
    """
    The off-graph dependence test.  H0: every non-edge pair is independent --
    true by construction for the ground-truth field, so any rejection is
    dependence the sampler under test created.

    null='exact' requires `field` (the exact EBF with the same graph and eps);
    null='perm' needs nothing but X.
    """
    rng = rng or np.random.default_rng(0)
    ne, stat = leakage_map(X, A)
    m = X.shape[0]
    if null == "exact":
        if field is None:
            raise ValueError("null='exact' needs the ground-truth field")
        null_stats = exact_null_stats(field, m, n_null, rng, ne)
    elif null == "perm":
        null_stats = perm_null_stats(X, n_null, rng, ne)
    else:
        raise ValueError(null)

    pval, scale = calibrated_p(stat, null_stats, m)
    max_null = np.abs(null_stats).max(1)
    max_stat = float(np.abs(stat).max())
    max_p = float((1.0 + (max_null >= max_stat).sum()) / (n_null + 1.0))
    rej = bh_fdr(pval, alpha)
    return LeakageReport(pairs=ne, stat=np.abs(stat), pval=pval, rejected_fdr=rej,
                         max_stat=max_stat, max_pval=max_p, null_kind=null,
                         n_null=n_null, alpha=alpha, detected=ne[rej],
                         null_scale=scale)


# ------------------------------------------------- conditional independence
def partial_corr(X: np.ndarray, i: int, j: int, S: list[int]) -> float:
    """corr(X_i, X_j | X_S) by residualising both on X_S."""
    if not S:
        return float(pair_corr(X, np.array([[i, j]]))[0])
    Z = np.column_stack([np.ones(len(X)), X[:, S]])
    beta, *_ = np.linalg.lstsq(Z, X[:, [i, j]], rcond=None)
    R = X[:, [i, j]] - Z @ beta
    sd = R.std(0)
    if np.any(sd < 1e-12):
        return 0.0
    return float((R[:, 0] * R[:, 1]).mean() / (sd[0] * sd[1]))


def separator_test(X: np.ndarray, i: int, j: int, S: list[int]) -> dict:
    """
    Conditional-independence test across a graph separator.

    In a Markov random field with graph G, if S separates i from j then
    X_i ⟂ X_j | X_S *exactly*.  For the Gaussian EBF this is exact at every eps,
    which turns the field's textbook property into a calibrated test -- the
    thing an attention-derived proxy graph can only approximate.
    """
    r = partial_corr(X, i, j, S)
    z = float(fisher_z(np.array([r]), X.shape[0], df_extra=len(S))[0])
    from scipy.stats import norm as _norm
    return dict(pcorr=r, z=z, p=float(2 * _norm.sf(abs(z))), sep_size=len(S))


def rho_by_lag(X: np.ndarray, max_lag: int) -> np.ndarray:
    """
    Mean correlation at each lag 1..max_lag (translation-averaged).

    The AR(1) known-answer check reads this: a Markov field must show rho^d at
    every lag, an EBF exactly zero off its graph.  Same estimator, two verdicts.
    """
    n = X.shape[1]
    out = np.zeros(max_lag + 1)
    for d in range(1, max_lag + 1):
        pairs = np.array([[i, i + d] for i in range(n - d)])
        out[d] = pair_corr(X, pairs).mean() if len(pairs) else np.nan
    return out


# ------------------------------------------------------------ co-commit deficit
def co_commit_deficit(X: np.ndarray, step: np.ndarray, A: np.ndarray,
                      truth: np.ndarray) -> dict:
    """
    Per-edge correlation deficit split by whether the endpoints were committed
    in the same sampler step.

    `step[b, i]` = the decoding step at which coordinate i of sample b was
    committed.  The law-of-total-covariance prediction is that the deficit
    concentrates on same-step edges.  Rigid content is the pre-registered case
    where it should NOT, because there the error is driven by
    distance-from-evidence instead -- we have observed that reversal on a 2-D
    shift-structured corpus.  Both outcomes are reportable; that is the point of
    measuring rather than assuming.
    """
    e = edge_list(A)
    same = step[:, e[:, 0]] == step[:, e[:, 1]]          # (m, |E|)
    rho_all = pair_corr(X, e)
    tgt = truth[e[:, 0], e[:, 1]]
    out = dict(deficit_all=float((tgt - rho_all).mean()),
               p_cocommit=float(same.mean()))
    for name, sel in (("same", same), ("diff", ~same)):
        rows = sel.any(1)
        if rows.sum() < 32:
            out[f"deficit_{name}"] = float("nan")
            continue
        # per-edge, restricted to the samples where that edge was (not) co-committed
        defs = []
        for k in range(len(e)):
            idx = np.flatnonzero(sel[:, k])
            if len(idx) >= 32:
                defs.append(tgt[k] - pair_corr(X[idx], e[k:k + 1])[0])
        out[f"deficit_{name}"] = float(np.mean(defs)) if defs else float("nan")
        out[f"n_edges_{name}"] = len(defs)
    return out


# ------------------------------------------- reverse-KL / incoherence statistic
def _pair_table(X: np.ndarray, i: int, j: int, q: int) -> np.ndarray:
    """Empirical q x q joint of two categorical coordinates."""
    idx = (X[:, i].astype(int) * q + X[:, j].astype(int))
    return np.bincount(idx, minlength=q * q).reshape(q, q) / len(X)


def pair_kl_categorical(X: np.ndarray, i: int, j: int, P_true: np.ndarray,
                        smooth: float = 0.0) -> dict:
    """
    Both KL directions between the sampler's edge joint and the exact one.

    Why both (cf. arXiv:2602.00286).  Forward KL(sampler || true) is
    *blind* to configurations the sampler never produces: those terms carry
    weight P_hat = 0 and drop out.  Reverse KL(true || sampler) puts the true
    mass on them and diverges.  Rigid instances are exactly the near-zero-
    support case, so a table reporting only forward KL cannot see the failure
    mode this program is about.

    `smooth` adds a pseudo-count *only* to the reported `kl_rev_smoothed`; the
    unsmoothed `kl_rev` is left infinite when support is genuinely missing,
    because silently regularising an infinity into a finite number is how a
    blind statistic gets shipped.
    """
    q = P_true.shape[0]
    P_hat = _pair_table(X, i, j, q)
    tt, hh = P_true.ravel(), P_hat.ravel()

    fwd_mask = hh > 0
    kl_fwd = float(np.sum(hh[fwd_mask] * np.log(hh[fwd_mask] / np.clip(tt[fwd_mask], 1e-300, None))))

    missing = (tt > 0) & (hh <= 0)
    if missing.any():
        kl_rev = float("inf")
    else:
        rev_mask = tt > 0
        kl_rev = float(np.sum(tt[rev_mask] * np.log(tt[rev_mask] / hh[rev_mask])))

    Ps = (P_hat.ravel() + smooth) / (1.0 + smooth * q * q) if smooth > 0 else hh
    rev_mask = tt > 0
    kl_rev_s = float(np.sum(tt[rev_mask] * np.log(tt[rev_mask] / np.clip(Ps[rev_mask], 1e-300, None))))

    return dict(kl_fwd=kl_fwd, kl_rev=kl_rev, kl_rev_smoothed=kl_rev_s,
                missing_support=int(missing.sum()),
                missing_true_mass=float(tt[missing].sum()))


def pair_kl_gaussian(rho_hat: float, rho_true: float) -> dict:
    """Closed-form bivariate-normal KL, unit marginals, both directions."""
    def _kl(a: float, b: float) -> float:
        da, db = 1.0 - a * a, 1.0 - b * b
        if da <= 0 or db <= 0:
            return float("inf")
        return float(0.5 * (np.log(db / da) - 2.0 + (2.0 - 2.0 * a * b) / db))
    return dict(kl_fwd=_kl(rho_true, rho_hat), kl_rev=_kl(rho_hat, rho_true))


def edge_kl(X: np.ndarray, A: np.ndarray, field, smooth: float = 0.0) -> dict:
    """
    Per-edge KL in both directions, aggregated -- the headline reporting statistic
    for how far a sampler's edge law sits from the exact one.

    Categorical fields use the exact `qary_joint_table` at the field's own eps
    as P_true (binary = the q=2 case), so the reference is exact rather than
    estimated.  Gaussian fields use the closed form on correlations.  Non-edges
    are reported separately: their P_true is the exact product, so any KL there
    is manufactured by the sampler -- the exact-null statement in KL units.
    """
    from ebf import qary_joint_table

    n = A.shape[0]
    edges = [(i, j) for i in range(n) for j in range(i + 1, n) if A[i, j]]
    nonedges = [(i, j) for i in range(n) for j in range(i + 1, n) if not A[i, j]]
    kind = getattr(field, "kind", "gaussian")

    def _collect(pairs, on_edge):
        out = []
        for (i, j) in pairs:
            if kind == "gaussian":
                r_hat = float(np.corrcoef(X[:, i], X[:, j])[0, 1])
                r_true = float(field.Sigma[i, j]) if on_edge else 0.0
                out.append(pair_kl_gaussian(r_hat, r_true))
            else:
                q = 2 if kind == "binary" else field.q
                P_true = (qary_joint_table(field.eps if on_edge else 0.0, q)
                          if kind == "qary" else _binary_table(field, on_edge))
                out.append(pair_kl_categorical(X, i, j, P_true, smooth))
        return out

    def _binary_table(fld, on_edge):
        p, r = fld.p, (fld.eps if on_edge else 0.0)
        if abs(r) < 1e-12:
            return np.array([[(1 - p) ** 2, (1 - p) * p], [p * (1 - p), p * p]])[::-1, ::-1]
        from ebf import bvn_cdf
        from scipy.stats import norm as _nrm
        t = float(_nrm.ppf(p))
        p11 = bvn_cdf(t, t, r)
        return np.array([[1 - 2 * p + p11, p - p11], [p - p11, p11]])

    E, N = _collect(edges, True), _collect(nonedges, False)
    agg = lambda rows, k: float(np.mean([r[k] for r in rows])) if rows else float("nan")
    finite = lambda rows, k: float(np.mean([r[k] for r in rows if np.isfinite(r[k])])) if rows else float("nan")
    return dict(
        n_edges=len(edges), n_nonedges=len(nonedges),
        edge_kl_fwd=agg(E, "kl_fwd"), edge_kl_rev=agg(E, "kl_rev"),
        edge_kl_rev_finite=finite(E, "kl_rev"),
        nonedge_kl_fwd=agg(N, "kl_fwd"), nonedge_kl_rev=agg(N, "kl_rev"),
        nonedge_kl_rev_finite=finite(N, "kl_rev"),
        n_edges_missing_support=sum(1 for r in E if not np.isfinite(r["kl_rev"])),
        per_edge=E)


def conditional_null_map(X: np.ndarray, A: np.ndarray, max_pairs: int = 400,
                         rng: np.random.Generator | None = None) -> dict:
    """
    The CONDITIONAL null: |partial corr(i, j | N(i))| over non-adjacent pairs.

    Uses the local Markov property -- for any j not in N(i) u {i}, the
    neighbourhood N(i) separates i from j -- so this is exactly zero on a
    genuine MRF (ebf.IsingField) for ANY graph, with no need to search for a
    minimal cut.  It is NOT zero on an EBF, which is covariance-sparse rather
    than precision-sparse; that asymmetry is the point (see IsingField).

    Pair with `leakage_test` (the marginal null).  Each corpus makes exactly one
    of the two exactly zero, so reporting both says *which kind* of dependence a
    schedule destroyed rather than only *how much*.
    """
    rng = rng or np.random.default_rng(0)
    n = A.shape[0]
    cand = [(i, j) for i in range(n) for j in range(n)
            if i != j and A[i, j] == 0 and (A[i] > 0).sum() >= 1]
    if len(cand) > max_pairs:
        cand = [cand[t] for t in rng.choice(len(cand), max_pairs, replace=False)]
    vals = []
    for i, j in cand:
        S = [int(v) for v in np.where(A[i] > 0)[0] if v != j]
        if not S:
            continue
        pc = partial_corr(X, i, j, S)
        if np.isfinite(pc):
            vals.append(abs(pc))
    v = np.asarray(vals)
    return dict(cond_mean_abs=float(v.mean()) if len(v) else float("nan"),
                cond_max_abs=float(v.max()) if len(v) else float("nan"),
                cond_q95=float(np.quantile(v, 0.95)) if len(v) else float("nan"),
                cond_n_pairs=int(len(v)))
