#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The bound comparator: what a realized error is measured *against*.

ParallelBench (ICLR'26) Thm 1 lower-bounds the KL any T-step parallel schedule
must pay:

    min KL  >=  sum_t  E[ C(S_t | X, S_<t) ]

where C is the conditional TOTAL CORRELATION of the block S_t revealed at step
t, given everything already revealed.  Thm 2 makes it monotone in T.  The bound
is real theory and it is not ours -- what is missing from the literature is any
*measurement* of where an actual sampler sits relative to it, because C is
intractable on real text (ParallelBench says so explicitly).

On an EBF it is not intractable.  For the Gaussian regime it is a closed form:
the total correlation of a Gaussian block is

    C(S) = 1/2 * ( sum_i log Sigma_ii  -  log det Sigma )

and conditioning is a Schur complement, so the whole floor is a few Choleskys.
That turns "how close to the information-theoretic floor does this scheduler
operate?" from an unanswerable question into a number.

Everything here is schedule-side only: no model, no samples.  The floor depends
on the field and the commit schedule, nothing else -- which is exactly why it
can be computed before a single token is generated.
"""
from __future__ import annotations

import numpy as np

__all__ = ["total_correlation_gaussian", "conditional_total_correlation",
           "thm1_floor", "sequential_schedule", "oneshot_schedule",
           "random_schedule", "dilated_schedule", "independent_set_schedule",
           "blocks_from_steps"]


# ------------------------------------------------------------------ the floor
def total_correlation_gaussian(Sigma: np.ndarray) -> float:
    """C(S) = 1/2 (sum_i log Sigma_ii - logdet Sigma), in nats.  Zero iff diagonal."""
    if Sigma.shape[0] <= 1:
        return 0.0
    sign, logdet = np.linalg.slogdet(Sigma)
    if sign <= 0:
        raise ValueError("covariance is not positive definite")
    return 0.5 * (float(np.log(np.diag(Sigma)).sum()) - float(logdet))


def conditional_total_correlation(Sigma: np.ndarray, S: np.ndarray,
                                  R: np.ndarray) -> float:
    """C(S | R) for a Gaussian field: total correlation of the Schur complement."""
    S = np.asarray(S, dtype=int)
    if len(S) <= 1:
        return 0.0
    R = np.asarray(R, dtype=int)
    if len(R) == 0:
        return total_correlation_gaussian(Sigma[np.ix_(S, S)])
    SS = Sigma[np.ix_(S, S)]
    SR = Sigma[np.ix_(S, R)]
    RR = Sigma[np.ix_(R, R)]
    cond = SS - SR @ np.linalg.solve(RR, SR.T)
    cond = 0.5 * (cond + cond.T)                     # symmetrise round-off
    return total_correlation_gaussian(cond)


def thm1_floor(Sigma: np.ndarray, blocks: list[np.ndarray]) -> dict:
    """
    Sum_t C(S_t | S_<t) for a commit schedule given as an ordered partition.

    Returns the total and the per-step breakdown -- the per-step curve is the
    interesting object, because it says *when* in decoding a schedule pays.
    """
    per_step, revealed = [], []
    for S in blocks:
        per_step.append(conditional_total_correlation(Sigma, S, np.array(revealed, int)))
        revealed.extend(list(np.asarray(S, int)))
    n_committed = sum(len(b) for b in blocks)
    if n_committed != Sigma.shape[0]:
        raise ValueError(f"schedule commits {n_committed} of {Sigma.shape[0]} coordinates")
    return dict(floor_nats=float(sum(per_step)), per_step=[float(x) for x in per_step],
                n_steps=len(blocks))


# --------------------------------------------------------------- schedule zoo
def sequential_schedule(n: int) -> list[np.ndarray]:
    """T = n.  The floor is exactly 0: nothing is ever committed jointly."""
    return [np.array([i]) for i in range(n)]


def oneshot_schedule(n: int) -> list[np.ndarray]:
    """T = 1.  The floor is the total correlation of the whole field."""
    return [np.arange(n)]


def random_schedule(n: int, k: int, rng: np.random.Generator) -> list[np.ndarray]:
    """T = ceil(n/k) blocks of k, positions in random order."""
    perm = rng.permutation(n)
    return [perm[i:i + k] for i in range(0, n, k)]


def dilated_schedule(n: int, k: int) -> list[np.ndarray]:
    """
    DUS-style: each step takes positions spread by a fixed stride.

    Data-independent repulsion -- the geometric-schedule family's rule.  On a band
    graph it
    keeps co-committed positions far apart; on a skip-d graph with stride
    matching d it does the opposite -- it co-commits every edge, a resonance the
    coupling sweep is designed to expose.
    """
    T = int(np.ceil(n / k))
    return [np.arange(t, n, T)[:k] if len(np.arange(t, n, T)) else np.array([], int)
            for t in range(T)]


def independent_set_schedule(A: np.ndarray, k: int,
                             order: np.ndarray | None = None) -> list[np.ndarray]:
    """
    Conflict-free selection (DAPD-style, on a fixed graph): greedily commit up to
    k positions per step
    subject to no two being graph neighbours; top up when no conflict-free set
    of size k exists.  The top-up is not cosmetic: at small k it is where the
    schedule stops being an independent set at all, and in our own sweeps it is
    what produced a regression at K=2.

    `order` is the tie-break rule, and it is a real hyperparameter rather than a
    convenience argument -- position order is an *extremal* choice, not a
    neutral one, and the sign of its extremality flips with k.  Fix it and
    report it, or scheduler comparisons are confounded.
    """
    n = A.shape[0]
    order = np.arange(n) if order is None else np.asarray(order)
    remaining = list(order)
    blocks = []
    while remaining:
        chosen: list[int] = []
        for v in list(remaining):
            if len(chosen) >= k:
                break
            if all(A[v, u] == 0 for u in chosen):
                chosen.append(v)
        if len(chosen) < k:                       # top up, breaking conflict-freeness
            for v in list(remaining):
                if len(chosen) >= k:
                    break
                if v not in chosen:
                    chosen.append(v)
        blocks.append(np.array(chosen))
        remaining = [v for v in remaining if v not in chosen]
    return blocks


def blocks_from_steps(step: np.ndarray) -> list[np.ndarray]:
    """Turn a per-coordinate commit-step vector into an ordered partition."""
    return [np.flatnonzero(step == t) for t in np.unique(step)]


if __name__ == "__main__":
    from graphs import build_graph

    rng = np.random.default_rng(0)
    for name, eps in (("path_b1", 0.45), ("skip2", 0.45), ("grid8x8", 0.25)):
        A = build_graph(name)
        n = A.shape[0]
        Sigma = np.eye(n) + eps * A
        print(f"\n{name}  n={n}  eps={eps}")
        print(f"  sequential (T=n)   floor = {thm1_floor(Sigma, sequential_schedule(n))['floor_nats']:.6f} nats")
        print(f"  one-shot   (T=1)   floor = {thm1_floor(Sigma, oneshot_schedule(n))['floor_nats']:.4f} nats")
        for k in (2, 4, 8, 16):
            r = thm1_floor(Sigma, random_schedule(n, k, rng))["floor_nats"]
            d = thm1_floor(Sigma, dilated_schedule(n, k))["floor_nats"]
            s = thm1_floor(Sigma, independent_set_schedule(A, k))["floor_nats"]
            print(f"  k={k:<3} random {r:7.4f}   dilated {d:7.4f}   independent-set {s:7.4f}")
