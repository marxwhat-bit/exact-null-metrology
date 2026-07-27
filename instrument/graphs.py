#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Graph zoo for the exact-null dependence instrument.

Every graph here defines an epsilon-biased field (EBF) with covariance I + eps*A:
edges carry correlation eps, **non-edges carry exactly zero**.  That last clause
is the whole instrument -- it is what makes "dependence found off the graph" a
hypothesis test with a null that is true by construction rather than by
approximation.

The zoo spans the axes a scheduler could plausibly be sensitive to:

  path/band      translation-invariant, Markov-realizable (AR(1) can express it)
  skip-d         dependence at lag d ONLY -- no Markov chain can express this
                 (corr(d) = rho^d is never zero for rho != 0)
  grid           the naturally occurring 2-D graph; raster lags 1 and w
  star K_{1,m}   hub topology; the theory's 1/sqrt(m) ceiling corollary
  odd cycle C_n  the extremal instances: MVE(C_9) = 22/45 < 1/2 needs
                 NON-hypermetric cut-polytope facets (theory/certify_c9.py)
  random d-reg   geometry-free control: same degree, no spatial structure
  shuffled(A)    identical spectrum + degree sequence, destroyed alignment

`eps_window(A)` returns [-1/lambda_max, +1/|lambda_min|], the exact feasible
interval for the Gaussian field; `mve_binary_extreme` returns the *binary*
extreme where it differs from the spectral one (odd cycles), taken from the
certified exact rationals in theory/.
"""
from __future__ import annotations

from fractions import Fraction

import numpy as np

__all__ = [
    "path_band", "skip", "grid", "star", "odd_cycle", "random_regular",
    "shuffled", "eps_window", "spectral_ceiling", "spectral_floor",
    "mve_binary_extreme", "edge_list", "nonedge_list", "GRAPH_ZOO", "build_graph",
]


# --------------------------------------------------------------------- builders
def path_band(n: int, band: int = 1) -> np.ndarray:
    """Band graph: edges between positions at distance 1..band."""
    d = np.abs(np.arange(n)[:, None] - np.arange(n)[None, :])
    return ((d >= 1) & (d <= band)).astype(np.float64)


def skip(n: int, dist: int = 2) -> np.ndarray:
    """Edges ONLY at exactly `dist`.  The minimal non-Markov instance."""
    d = np.abs(np.arange(n)[:, None] - np.arange(n)[None, :])
    return (d == dist).astype(np.float64)


def grid(h: int, w: int, metric: str = "manhattan", radius: int = 1) -> np.ndarray:
    """Row-major flattened lattice.  metric='manhattan' -> 4-neighbourhood."""
    coords = np.array([(i, j) for i in range(h) for j in range(w)])
    di = np.abs(coords[:, None, 0] - coords[None, :, 0])
    dj = np.abs(coords[:, None, 1] - coords[None, :, 1])
    d = (di + dj) if metric == "manhattan" else np.maximum(di, dj)
    return ((d <= radius) & (d >= 1)).astype(np.float64)


def star(m: int) -> np.ndarray:
    """K_{1,m}: hub 0 joined to m leaves.  n = m + 1."""
    A = np.zeros((m + 1, m + 1))
    A[0, 1:] = A[1:, 0] = 1.0
    return A


def odd_cycle(n: int) -> np.ndarray:
    """C_n.  For odd n >= 9 the *binary* extreme sits strictly below 1/2."""
    A = np.zeros((n, n))
    for i in range(n):
        A[i, (i + 1) % n] = A[(i + 1) % n, i] = 1.0
    return A


def random_regular(n: int, deg: int, seed: int = 0) -> np.ndarray:
    """
    Geometry-free control: a random deg-regular graph on n vertices.

    Same local density as the structured graphs, no spatial or sequential
    meaning at all -- so any estimator behaviour that depends on *geometry*
    rather than on *the graph* shows up as a difference against this arm.
    """
    rng = np.random.default_rng(seed)
    for _ in range(200):                       # configuration model with retries
        stubs = np.repeat(np.arange(n), deg)
        rng.shuffle(stubs)
        A = np.zeros((n, n))
        ok = True
        for a, b in stubs.reshape(-1, 2):
            if a == b or A[a, b]:
                ok = False
                break
            A[a, b] = A[b, a] = 1.0
        if ok and np.all(A.sum(1) == deg):
            return A
    raise RuntimeError(f"could not build a {deg}-regular graph on {n} vertices")


def shuffled(A: np.ndarray, seed: int = 0) -> np.ndarray:
    """Vertex-permuted copy: identical spectrum and degree sequence."""
    p = np.random.default_rng(seed).permutation(A.shape[0])
    return A[np.ix_(p, p)]


# ------------------------------------------------------------------- eps window
def spectral_ceiling(A: np.ndarray) -> float:
    """eps_max = 1/|lambda_min(A)| -- the Gaussian (elliptope) bound."""
    lam = float(np.linalg.eigvalsh(A).min())
    return float("inf") if lam >= 0 else -1.0 / lam


def spectral_floor(A: np.ndarray) -> float:
    """eps_min = -1/lambda_max(A) -- the anti-correlated branch."""
    lam = float(np.linalg.eigvalsh(A).max())
    return float("-inf") if lam <= 0 else -1.0 / lam


def eps_window(A: np.ndarray) -> tuple[float, float]:
    return spectral_floor(A), spectral_ceiling(A)


# certified exact binary extremes for odd cycles (theory/exact_cycle.py,
# theory/certify_c9.py -- primal+dual certificates).  C_5, C_7 reach 1/2;
# from n=9 the true value is strictly below and needs non-hypermetric facets.
_MVE_ODD_CYCLE = {
    5: Fraction(1, 2), 7: Fraction(1, 2), 9: Fraction(22, 45),
    11: Fraction(59, 121), 13: Fraction(121, 247),
}


def mve_binary_extreme(kind: str, **kw) -> float | None:
    """
    The exact BINARY maximum edge correlation, where it is known and differs
    from the spectral ceiling.  Returns None when the spectral bound is the
    operative one.  These are the hardest feasible instances in the zoo --
    the one place the exact combinatorics is load-bearing rather than decorative.
    """
    if kind == "odd_cycle":
        return float(_MVE_ODD_CYCLE.get(int(kw["n"]), 0)) or None
    if kind == "star":
        # closed form: the binary ceiling is 1/sqrt(m), ATTAINED iff m is a
        # perfect square (theory/mve.py, mve_star_orbit).  The zoo uses m=16 so
        # the instrument's hardest star instance is an exactly attainable one.
        return 1.0 / np.sqrt(float(kw["m"]))
    return None


# ------------------------------------------------------------------ edge lists
def edge_list(A: np.ndarray) -> np.ndarray:
    """(E,2) upper-triangular indices where A > 0."""
    return np.array(np.triu(A, 1).nonzero()).T


def nonedge_list(A: np.ndarray) -> np.ndarray:
    """(M,2) upper-triangular indices where A == 0 -- the exact-null family."""
    n = A.shape[0]
    iu = np.triu_indices(n, 1)
    keep = A[iu] == 0
    return np.array([iu[0][keep], iu[1][keep]]).T


# ------------------------------------------------------------------------- zoo
GRAPH_ZOO: dict[str, dict] = {
    # name              kind           kwargs                     note
    "path_b1":   dict(kind="path_band",     n=32, band=1),
    "path_b2":   dict(kind="path_band",     n=32, band=2),
    "skip2":     dict(kind="skip",          n=32, dist=2),
    "skip4":     dict(kind="skip",          n=32, dist=4),
    "grid8x8":   dict(kind="grid",          h=8, w=8),
    "star16":    dict(kind="star",          m=16),
    "cycle9":    dict(kind="odd_cycle",     n=9),
    "cycle13":   dict(kind="odd_cycle",     n=13),
    "reg3_32":   dict(kind="random_regular", n=32, deg=3, seed=0),
    "path_b1_shuf": dict(kind="shuffled", base="path_b1", seed=0),
}


def build_graph(name: str) -> np.ndarray:
    """Instantiate a zoo entry by name."""
    spec = dict(GRAPH_ZOO[name])
    kind = spec.pop("kind")
    if kind == "shuffled":
        return shuffled(build_graph(spec["base"]), seed=spec.get("seed", 0))
    return {
        "path_band": lambda: path_band(spec["n"], spec["band"]),
        "skip": lambda: skip(spec["n"], spec["dist"]),
        "grid": lambda: grid(spec["h"], spec["w"]),
        "star": lambda: star(spec["m"]),
        "odd_cycle": lambda: odd_cycle(spec["n"]),
        "random_regular": lambda: random_regular(spec["n"], spec["deg"], spec.get("seed", 0)),
    }[kind]()


if __name__ == "__main__":
    print(f"{'graph':<14}{'n':>4}{'|E|':>6}{'deg':>7}"
          f"{'eps_min':>10}{'eps_max':>10}{'binary extreme':>18}")
    for name in GRAPH_ZOO:
        A = build_graph(name)
        lo, hi = eps_window(A)
        spec = GRAPH_ZOO[name]
        ext = mve_binary_extreme(spec.get("kind", ""), **{k: v for k, v in spec.items()
                                                          if k in ("n", "m")})
        print(f"{name:<14}{A.shape[0]:>4}{int(A.sum() // 2):>6}{A.sum(1).mean():>7.2f}"
              f"{lo:>10.4f}{hi:>10.4f}"
              f"{('-' if ext is None else f'{ext:.4f}'):>18}")
