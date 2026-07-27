#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The subject: an ideal parallel decoder on a Gaussian field.

An instrument with no subject is untested plumbing.  This is the smallest honest
subject -- and it happens to be exactly the object ParallelBench's Thm 1 bounds,
which is what makes realized-vs-floor computable here and nowhere else.

The pathology, stated precisely.  A masked-diffusion decoder committing a block
S_t at step t draws each position from its own conditional marginal given
everything already revealed, and *independently of the other positions in the
block*.  It is not an approximation of the joint; it is the product of the
correct conditional marginals.  On a Gaussian field both objects are closed
form, so:

  mu   = Sigma_SR Sigma_RR^-1 x_R          (correct conditional mean)
  C    = Sigma_SS - Sigma_SR Sigma_RR^-1 Sigma_RS
  ideal joint : x_S ~ N(mu, C)
  parallel    : x_S ~ N(mu, diag(C))       <-- the only difference

Everything the scheduling literature argues about lives in that one `diag`.  With
T = n the diag is 1x1 and the sampler is exact; with T = 1 it is the whole
covariance and every edge collapses.  In between, the instrument measures what
is actually lost, and `bound.thm1_floor` says what must be lost -- so
"realized vs floor" is a subtraction rather than a research programme.

`step_of` returns the per-coordinate commit step, which is what
`estimators.co_commit_deficit` needs to split edges by co-commitment.
"""
from __future__ import annotations

import numpy as np

__all__ = ["parallel_sample", "step_of", "exact_sample", "realized_cov",
           "realized_kl"]


def step_of(blocks: list[np.ndarray], n: int) -> np.ndarray:
    """Per-coordinate commit step, for the co-commit split."""
    s = np.full(n, -1, dtype=int)
    for t, b in enumerate(blocks):
        s[np.asarray(b, int)] = t
    if (s < 0).any():
        raise ValueError("schedule does not cover every coordinate")
    return s


def exact_sample(Sigma: np.ndarray, m: int, rng: np.random.Generator) -> np.ndarray:
    """The reference: draws from the true joint, whatever the schedule."""
    return rng.standard_normal((m, Sigma.shape[0])) @ np.linalg.cholesky(Sigma).T


def parallel_sample(Sigma: np.ndarray, blocks: list[np.ndarray], m: int,
                    rng: np.random.Generator) -> np.ndarray:
    """
    Draw m samples from the ideal parallel decoder following `blocks`.

    Exact in the conditional MEANS, independent within each committed block --
    the joint-vs-marginal mismatch and nothing else.  With the sequential
    schedule this reproduces the true field exactly (contract-tested).
    """
    n = Sigma.shape[0]
    X = np.zeros((m, n))
    revealed: list[int] = []
    for S in blocks:
        S = np.asarray(S, int)
        if len(revealed) == 0:
            mu = np.zeros((m, len(S)))
            C = Sigma[np.ix_(S, S)]
        else:
            R = np.asarray(revealed, int)
            W = np.linalg.solve(Sigma[np.ix_(R, R)], Sigma[np.ix_(R, S)])   # (|R|,|S|)
            mu = X[:, R] @ W
            C = Sigma[np.ix_(S, S)] - Sigma[np.ix_(S, R)] @ W
        sd = np.sqrt(np.clip(np.diag(C), 0.0, None))
        X[:, S] = mu + rng.standard_normal((m, len(S))) * sd
        revealed.extend(S.tolist())
    return X


def realized_cov(Sigma: np.ndarray, blocks: list[np.ndarray]) -> np.ndarray:
    """
    EXACT covariance of the ideal parallel decoder's output -- no sampling.

    The decoder is a linear-Gaussian recursion, x_S = W' x_R + diag(sd) z_S with
    one fresh standard normal per coordinate, so x = L z for a lower-triangular-
    in-commit-order L that we can accumulate blockwise.  Sigma_real = L L'.

    This is what makes realized-vs-floor a subtraction rather than a Monte-Carlo
    study: the
    realized error of a schedule is computable to machine precision, so the
    eps x scheduler x K grid costs nothing and carries no sampling noise.
    Contract: sequential -> Sigma exactly; one-shot -> diag(Sigma).
    """
    n = Sigma.shape[0]
    L = np.zeros((n, n))
    revealed: list[int] = []
    for S in blocks:
        S = np.asarray(S, int)
        if revealed:
            R = np.asarray(revealed, int)
            W = np.linalg.solve(Sigma[np.ix_(R, R)], Sigma[np.ix_(R, S)])
            C = Sigma[np.ix_(S, S)] - Sigma[np.ix_(S, R)] @ W
            L[S, :] = W.T @ L[R, :]
        else:
            C = Sigma[np.ix_(S, S)]
        L[S, S] += np.sqrt(np.clip(np.diag(C), 0.0, None))
        revealed.extend(S.tolist())
    return L @ L.T


def realized_kl(Sigma: np.ndarray, blocks: list[np.ndarray]) -> dict:
    """
    KL between the decoder's output law and the true field, both directions.

    ParallelBench Thm 1 lower-bounds the *achievable* KL of a T-step schedule by
    Sum_t C(S_t | X, S_<t).  `kl_fwd` = KL(parallel || true) is the quantity that
    bound speaks about; `kl_rev` is reported too because the two differ
    substantially once a schedule destroys variance, and picking whichever is
    smaller after the fact would be cherry-picking.
    """
    P = realized_cov(Sigma, blocks)
    n = Sigma.shape[0]
    sP, ldP = np.linalg.slogdet(P)
    sQ, ldQ = np.linalg.slogdet(Sigma)
    if sP <= 0 or sQ <= 0:
        return dict(kl_fwd=float("inf"), kl_rev=float("inf"), cov=P)
    kl_fwd = 0.5 * (np.trace(np.linalg.solve(Sigma, P)) - n + ldQ - ldP)
    kl_rev = 0.5 * (np.trace(np.linalg.solve(P, Sigma)) - n + ldP - ldQ)
    return dict(kl_fwd=float(kl_fwd), kl_rev=float(kl_rev), cov=P)


if __name__ == "__main__":
    from bound import (blocks_from_steps, dilated_schedule,
                       independent_set_schedule, oneshot_schedule,
                       random_schedule, sequential_schedule, thm1_floor)
    from estimators import co_commit_deficit, edge_fidelity, leakage_test
    from ebf import EBF
    from graphs import build_graph

    rng = np.random.default_rng(2026)
    m = 20000
    for gname, eps in (("path_b1", 0.45), ("skip2", 0.45)):
        A = build_graph(gname)
        n = A.shape[0]
        Sigma = np.eye(n) + eps * A
        field = EBF(A, eps)
        truth = field.truth()
        print(f"\n=== {gname}  n={n}  eps={eps}  (m={m}) ===")
        print(f"{'schedule':<22}{'K':>4}{'floor':>9}{'deficit':>10}"
              f"{'same-step':>11}{'diff-step':>11}{'leak p_FWER':>13}")
        scheds = [("sequential", sequential_schedule(n)),
                  ("one-shot", oneshot_schedule(n))]
        for k in (2, 4, 8, 16):
            scheds += [(f"random k={k}", random_schedule(n, k, rng)),
                       (f"dilated k={k}", dilated_schedule(n, k)),
                       (f"indep-set k={k}", independent_set_schedule(A, k))]
        for name, blocks in scheds:
            X = parallel_sample(Sigma, blocks, m, rng)
            fl = thm1_floor(Sigma, blocks)["floor_nats"]
            ef = edge_fidelity(X, A, truth, n_boot=0)
            cc = co_commit_deficit(X, np.tile(step_of(blocks, n), (m, 1)), A, truth)
            lk = leakage_test(X, A, field=field, n_null=60, rng=rng)
            k = max(len(b) for b in blocks)
            print(f"{name:<22}{k:>4}{fl:>9.4f}{-ef['bias']:>10.4f}"
                  f"{cc.get('deficit_same', float('nan')):>11.4f}"
                  f"{cc.get('deficit_diff', float('nan')):>11.4f}"
                  f"{lk.max_pval:>13.4f}")
