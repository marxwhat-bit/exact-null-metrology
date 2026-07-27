#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The epsilon-biased field and its exact ground truth.

Three data regimes, one construction.  A latent Gaussian Z ~ N(0, I + eps*A) is
pushed through a per-coordinate monotone map, so

  * marginals are exact at every eps (each Z_i is exactly N(0,1)),
  * edges carry a **known closed-form** dependence,
  * non-edges are **exactly independent** -- independence of the latent pair
    survives any per-coordinate transform, so this is not an approximation that
    degrades with eps, q, or the threshold.

That asymmetry is the instrument: `target_corr` says what a perfect sampler must
produce on edges, and says *exactly zero* everywhere else.

Regimes
-------
gaussian : X = Z.                       target corr on edges = eps
binary   : X = 1[Z < Phi^-1(p)].        target = (Phi_2(t,t;eps) - p^2)/(p(1-p))
qary     : X = quantile bin of Z (q equiprobable bins).
           target = the exact q x q joint cell table (bivariate normal
           rectangle probabilities); mutual information is the natural
           statistic and is exactly 0 off the graph.

Subjects under test (for GM1 calibration)
-----------------------------------------
EBF          the exact sampler -- nulls must cover at their nominal rate
LeakyEBF     exact field PLUS injected correlation `delta` on chosen NON-edges;
             the instrument must detect those edges and only those, with power
             monotone in delta
AR1Copula    a genuine Markov field, corr(d) = rho^d; leaks at every lag, so
             recovered rho_hat(d) must match rho^d within CI
"""
from __future__ import annotations

import numpy as np
from scipy.stats import multivariate_normal, norm

__all__ = ["EBF", "GMRF", "IsingField", "LeakyEBF", "AR1Copula", "target_corr",
           "assert_feasible", "bvn_cdf",
           "binary_corr_from_latent", "qary_corr_from_latent",
           "qary_mutual_information", "qary_joint_table", "psd_project"]


# ------------------------------------------------------------------ primitives
def bvn_cdf(a: float, b: float, rho: float) -> float:
    """P(Z1 < a, Z2 < b) for a standard bivariate normal with correlation rho."""
    if abs(rho) < 1e-12:
        return float(norm.cdf(a) * norm.cdf(b))
    cov = np.array([[1.0, rho], [rho, 1.0]])
    return float(multivariate_normal(mean=[0.0, 0.0], cov=cov).cdf([a, b]))


def binary_corr_from_latent(rho: float, p: float = 0.5) -> float:
    """
    Exact Bernoulli(p) correlation induced by a latent Gaussian correlation rho
    under the threshold X = 1[Z < Phi^-1(p)].

    rho = 0  ->  exactly 0 (the exact null, for any p).
    p = 1/2  ->  the classical (2/pi) arcsin(rho).
    """
    if abs(rho) < 1e-12:
        return 0.0
    t = float(norm.ppf(p))
    return (bvn_cdf(t, t, rho) - p * p) / (p * (1.0 - p))


def qary_joint_table(rho: float, q: int) -> np.ndarray:
    """
    Exact q x q joint distribution of two equiprobable-quantile-binned
    coordinates with latent correlation rho.  Off the graph (rho=0) this is the
    outer product 1/q^2 -- exact independence, hence exactly zero mutual
    information.
    """
    edges = np.concatenate([[-np.inf], norm.ppf(np.arange(1, q) / q), [np.inf]])
    if abs(rho) < 1e-12:
        return np.full((q, q), 1.0 / (q * q))
    F = np.empty((q + 1, q + 1))
    for i, a in enumerate(edges):
        for j, b in enumerate(edges):
            if not np.isfinite(a) or not np.isfinite(b):
                F[i, j] = 0.0 if (a == -np.inf or b == -np.inf) else (
                    norm.cdf(b) if a == np.inf else norm.cdf(a))
                if a == np.inf and b == np.inf:
                    F[i, j] = 1.0
            else:
                F[i, j] = bvn_cdf(a, b, rho)
    P = F[1:, 1:] - F[:-1, 1:] - F[1:, :-1] + F[:-1, :-1]
    return np.clip(P, 0.0, None) / P.sum()


def qary_corr_from_latent(rho: float, q: int) -> float:
    """
    Exact Pearson correlation of the BIN INDICES (0..q-1) induced by latent rho.

    Computed from the exact joint table rather than from a Spearman-style
    approximation -- an instrument may not carry a target that is only nearly
    right, or every fidelity number it reports inherits that error.
    """
    if abs(rho) < 1e-12:
        return 0.0
    P = qary_joint_table(rho, q)
    k = np.arange(q, dtype=float)
    pi = P.sum(1)
    mu = float(k @ pi)
    var = float(((k - mu) ** 2) @ pi)
    cov = float((k - mu) @ P @ (k - mu))
    return cov / var


def qary_mutual_information(rho: float, q: int) -> float:
    """Exact MI (nats) of a binned pair at latent correlation rho; 0 iff rho=0."""
    P = qary_joint_table(rho, q)
    pi, pj = P.sum(1, keepdims=True), P.sum(0, keepdims=True)
    nz = P > 0
    return float(np.sum(P[nz] * np.log(P[nz] / (pi @ pj)[nz])))


def assert_feasible(C: np.ndarray, what: str, floor: float = 1e-9) -> np.ndarray:
    """
    Refuse to silently repair an infeasible covariance.

    A nearest-PSD nudge rewrites EVERY entry, which would destroy the exact
    off-graph zeros the whole instrument rests on -- and it would do it quietly,
    so the leakage test would keep reporting p-values against a null that is no
    longer true.  An infeasible (graph, eps) is a user error with a computable
    answer (the spectral window), so say so instead.
    """
    w = float(np.linalg.eigvalsh(C).min())
    if w <= floor:
        raise ValueError(
            f"{what}: covariance is not positive definite (min eig {w:.3e}). "
            f"eps is outside the feasible window -- see graphs.eps_window(A).")
    return C


def psd_project(C: np.ndarray, floor: float = 1e-9) -> np.ndarray:
    """
    Nearest-PSD nudge with unit diagonal restored.

    Only used where the matrix is dense anyway (the GMRF's normalised
    covariance).  Never on a covariance whose zero pattern is load-bearing --
    use `assert_feasible` there.
    """
    w, V = np.linalg.eigh(C)
    if w.min() >= floor:
        return C
    C = V @ np.diag(np.clip(w, floor, None)) @ V.T
    d = np.sqrt(np.diag(C))
    return C / np.outer(d, d)


def target_corr(A: np.ndarray, eps: float, kind: str = "gaussian",
                p: float = 0.5, q: int = 4) -> np.ndarray:
    """
    The exact ground-truth correlation matrix a perfect sampler must reproduce.

    Off-graph entries are exactly 0 in every regime -- that is the null the whole
    instrument tests against.
    """
    n = A.shape[0]
    if kind == "gaussian":
        R = np.eye(n) + eps * A
    elif kind == "binary":
        R = np.eye(n) + binary_corr_from_latent(eps, p) * A
    elif kind == "qary":
        # exact bin-index correlation; the companion statistic is mutual
        # information, whose target is qary_joint_table(eps, q) on edges and
        # exactly 1/q^2 (i.e. MI = 0) off them.
        R = np.eye(n) + qary_corr_from_latent(eps, q) * A
    else:
        raise ValueError(kind)
    return R


# -------------------------------------------------------------------- samplers
class EBF:
    """The exact epsilon-biased field.  This is the instrument's calibration standard."""

    kind_default = "gaussian"

    def __init__(self, A: np.ndarray, eps: float, kind: str = "gaussian",
                 p: float = 0.5, q: int = 4):
        self.A, self.eps, self.kind, self.p, self.q = A, float(eps), kind, p, int(q)
        self.n = A.shape[0]
        self.Sigma = assert_feasible(np.eye(self.n) + self.eps * A,
                                     f"EBF(eps={eps})")
        self.L = np.linalg.cholesky(self.Sigma)

    # -- latent -------------------------------------------------------------
    def latent(self, m: int, rng: np.random.Generator) -> np.ndarray:
        return rng.standard_normal((m, self.n)) @ self.L.T

    # -- observed -----------------------------------------------------------
    def sample(self, m: int, rng: np.random.Generator) -> np.ndarray:
        Z = self.latent(m, rng)
        if self.kind == "gaussian":
            return Z
        if self.kind == "binary":
            return (Z < norm.ppf(self.p)).astype(np.float64)
        if self.kind == "qary":
            cuts = norm.ppf(np.arange(1, self.q) / self.q)
            return np.searchsorted(cuts, Z).astype(np.float64)
        raise ValueError(self.kind)

    def truth(self) -> np.ndarray:
        return target_corr(self.A, self.eps, self.kind, self.p, self.q)

    @property
    def tag(self) -> str:
        return f"ebf[{self.kind}] eps={self.eps:+.4f}"


class LeakyEBF(EBF):
    """
    The exact field plus deliberately injected dependence on chosen NON-edges.

    This is the instrument's *power* standard: we know precisely which non-edges
    were corrupted and by how much, so detection must be localized to them and
    must strengthen monotonically with `delta`.  A detector that lights up
    elsewhere is producing false positives; one that stays dark is blind.
    """

    def __init__(self, A: np.ndarray, eps: float, inject: np.ndarray, delta: float,
                 kind: str = "gaussian", p: float = 0.5, q: int = 4):
        self.inject = np.asarray(inject, dtype=int).reshape(-1, 2)
        self.delta = float(delta)
        B = np.zeros_like(A)
        for i, j in self.inject:
            if A[i, j] != 0:
                raise ValueError(f"injection pair ({i},{j}) is an EDGE of A, not a non-edge")
            B[i, j] = B[j, i] = 1.0
        self.B = B
        super().__init__(A, eps, kind, p, q)
        self.Sigma = assert_feasible(
            np.eye(A.shape[0]) + eps * A + self.delta * B,
            f"LeakyEBF(eps={eps}, delta={delta})")
        self.L = np.linalg.cholesky(self.Sigma)

    @property
    def tag(self) -> str:
        return f"leaky[{self.kind}] eps={self.eps:+.3f} delta={self.delta:+.3f} x{len(self.inject)}"


class GMRF(EBF):
    """
    The companion field: sparse PRECISION, Sigma = (I - eps*A)^-1, unit-scaled.

    Why this exists.  The EBF is sparse in *covariance*, so its exact null is
    MARGINAL independence off the graph.  A Gaussian MRF is sparse in
    *precision*, so its exact null is CONDITIONAL independence off the graph:
    X_i ⟂ X_j | X_S for any separator S, exactly, at every eps.

    That distinction is load-bearing for line M.  The heuristic the field runs on
    -- "attention ≈ dependence" (DAPD §3.2) -- is a claim about *conditional*
    dependence, and DAPD's own Limitations section names "exact
    conditional-independence tests" as the missing tool.  A covariance-sparse
    field alone cannot supply that test.  Holding both fields means the
    instrument can say which notion of independence a sampler actually breaks,
    instead of conflating them.

    Feasible window: I - eps*A ≻ 0  <=>  eps in (-1/|lambda_min|, 1/lambda_max)
    -- the mirror image of the EBF's [-1/lambda_max, +1/|lambda_min|].
    """

    def __init__(self, A: np.ndarray, eps: float, kind: str = "gaussian",
                 p: float = 0.5, q: int = 4):
        self.A, self.eps, self.kind, self.p, self.q = A, float(eps), kind, p, int(q)
        self.n = A.shape[0]
        K = np.eye(self.n) - self.eps * A
        w = np.linalg.eigvalsh(K).min()
        if w <= 1e-9:
            raise ValueError(f"eps={eps} outside the GMRF window; min eig(I-epsA)={w:.3e}")
        self.K = K
        S = np.linalg.inv(K)
        d = np.sqrt(np.diag(S))
        self.Sigma = S / np.outer(d, d)
        self.L = np.linalg.cholesky(psd_project(self.Sigma))

    def truth(self) -> np.ndarray:
        """Marginal correlations are dense here; the exact zeros live in K."""
        if self.kind == "gaussian":
            return self.Sigma
        R = np.vectorize(lambda r: binary_corr_from_latent(r, self.p)
                         if self.kind == "binary" else
                         qary_corr_from_latent(r, self.q))(self.Sigma)
        np.fill_diagonal(R, 1.0)
        return R

    def partial_truth(self) -> np.ndarray:
        """Exact partial correlations: -K_ij/sqrt(K_ii K_jj); ZERO off the graph."""
        d = np.sqrt(np.diag(self.K))
        P = -self.K / np.outer(d, d)
        np.fill_diagonal(P, 1.0)
        return P

    @staticmethod
    def eps_window(A: np.ndarray) -> tuple[float, float]:
        w = np.linalg.eigvalsh(A)
        lo = -1.0 / abs(float(w.min())) if w.min() < 0 else -np.inf
        hi = 1.0 / float(w.max()) if w.max() > 0 else np.inf
        return lo, hi

    @property
    def tag(self) -> str:
        return f"gmrf[{self.kind}] eps={self.eps:+.4f}"


class AR1Copula(EBF):
    """
    Genuine Markov field: latent corr(i,j) = rho^|i-j|.

    It is the honest strong baseline for "just use a correlated field", and for
    the instrument it is a known-answer subject: recovered rho_hat(d) must track
    rho^d at every lag, including the lags where an EBF would read exactly zero.
    """

    def __init__(self, n: int, rho: float, kind: str = "gaussian",
                 p: float = 0.5, q: int = 4):
        self.rho = float(rho)
        d = np.abs(np.arange(n)[:, None] - np.arange(n)[None, :])
        A = np.zeros((n, n))
        self.A, self.eps, self.kind, self.p, self.q, self.n = A, 0.0, kind, p, int(q), n
        self.Sigma = assert_feasible(self.rho ** d, f"AR1Copula(rho={rho})")
        self.L = np.linalg.cholesky(self.Sigma)

    def truth(self) -> np.ndarray:
        d = np.abs(np.arange(self.n)[:, None] - np.arange(self.n)[None, :])
        R = self.rho ** d
        if self.kind == "binary":
            R = np.vectorize(lambda r: binary_corr_from_latent(r, self.p))(R)
            np.fill_diagonal(R, 1.0)
        return R

    @property
    def tag(self) -> str:
        return f"ar1[{self.kind}] rho={self.rho:+.4f}"


if __name__ == "__main__":
    from graphs import build_graph, nonedge_list

    rng = np.random.default_rng(0)
    A = build_graph("path_b1")
    for kind in ("gaussian", "binary", "qary"):
        f = EBF(A, 0.45, kind=kind)
        X = f.sample(200_000, rng)
        R = np.corrcoef(X.T)
        ne = nonedge_list(A)
        e = np.array(np.triu(A, 1).nonzero()).T
        tgt = f.truth()
        print(f"{f.tag:<28} edge rho_hat={R[e[:,0],e[:,1]].mean():+.4f} "
              f"(target {tgt[e[0,0],e[0,1]]:+.4f})   "
              f"max|non-edge rho_hat|={np.abs(R[ne[:,0],ne[:,1]]).max():.4f}")


class IsingField:
    """
    A genuinely discrete MRF on G -- the companion the token track needs.

    Why this class exists.  `GMRF` is conditionally independent off the graph in
    the LATENT Gaussian, but thresholding destroys that: conditioning on the
    coarse token X_S is not conditioning on Z_S.  Measured on a band-1 path at
    eps=0.45, |pcorr(0,2 | 1)| is 0.0011 for the Gaussian GMRF and **0.0696**
    once binarised -- 23% of the edge signal, i.e. no usable null.  An Ising
    model is Markov w.r.t. G *in token space by construction*, so the separator
    test has an exactly-zero null on the corpus a masked-diffusion model is
    actually trained on.

    The two corpora are complementary, and P3 needs both:
        EBF   -- marginal   null exact off-graph, conditional null NOT exact
        Ising -- conditional null exact across separators, marginal null NOT
    A scheduler claim that holds under one and not the other is a finding about
    which kind of dependence the schedule destroys, not a measurement artefact.

    Coupling dial: beta (inverse temperature) plays eps's role.  `edge_corr()`
    reports the realised edge correlation so the two tracks can be compared at
    matched dependence strength rather than at matched nominal parameter.
    """

    def __init__(self, A: np.ndarray, beta: float, q: int = 2):
        if int(q) != 2:
            raise NotImplementedError("Ising track is binary (q=2) for now")
        self.A, self.beta, self.q, self.kind = np.asarray(A, float), float(beta), 2, "binary"
        self.n = self.A.shape[0]
        self.eps = float(beta)          # the dial, for interface compatibility

    # -- exact sampling on paths via forward simulation ---------------------
    def _path_components(self) -> list[list[int]] | None:
        """
        Decompose A into connected components and return each as a path order,
        or None if any component is not a simple path.

        Why components matter: `skip2` is TWO disjoint 16-chains, and a
        single-path check would silently fall back to Gibbs -- approximate
        sampling inside an exact-null instrument.  Disjoint components are
        independent under the Gibbs measure, so sampling each path exactly
        samples the whole field exactly.
        """
        deg = (self.A > 0).sum(1)
        unseen = set(range(self.n))
        comps: list[list[int]] = []
        while unseen:
            start = next(iter(unseen))
            comp, stack = set(), [start]
            while stack:
                v = stack.pop()
                if v in comp:
                    continue
                comp.add(v)
                stack.extend(int(u) for u in np.where(self.A[v] > 0)[0])
            unseen -= comp
            if len(comp) == 1:
                comps.append([int(next(iter(comp)))]); continue
            ends = [v for v in comp if deg[v] == 1]
            if len(ends) != 2 or not all(deg[v] in (1, 2) for v in comp):
                return None                       # not a path -> caller uses Gibbs
            order, prev, cur = [int(ends[0])], -1, int(ends[0])
            while True:
                nb = [int(v) for v in np.where(self.A[cur] > 0)[0] if v != prev]
                if not nb:
                    break
                prev, cur = cur, nb[0]
                order.append(cur)
            if len(order) != len(comp):
                return None
            comps.append(order)
        return comps

    def _sample_paths(self, comps: list[list[int]], m: int, rng) -> np.ndarray:
        """Exact forward simulation along each path component; no burn-in."""
        b = self.beta
        # symmetric two-state chain: P(s_{i+1}=s_i) = e^b / (e^b + e^-b)
        p_same = np.exp(b) / (np.exp(b) + np.exp(-b))
        out = np.empty((m, self.n), dtype=np.int8)
        for order in comps:
            L = len(order)
            S = np.empty((m, L), dtype=np.int8)
            S[:, 0] = rng.integers(0, 2, m)
            for t in range(1, L):
                flip = rng.random(m) > p_same
                S[:, t] = S[:, t - 1] ^ flip
            out[:, np.asarray(order)] = S
        return out

    def _sample_gibbs(self, m: int, rng, sweeps: int = 200) -> np.ndarray:
        """Gibbs for non-chain graphs.  `sweeps` is reported, not hidden."""
        X = rng.integers(0, 2, (m, self.n)).astype(np.int8)
        Aint = (self.A > 0)
        for _ in range(sweeps):
            for i in range(self.n):
                nb = np.where(Aint[i])[0]
                s = (2 * X[:, nb].astype(np.int32) - 1).sum(1) if len(nb) else 0
                p1 = 1.0 / (1.0 + np.exp(-2.0 * self.beta * s))
                X[:, i] = (rng.random(m) < p1).astype(np.int8)
        return X

    def sample(self, m: int, rng) -> np.ndarray:
        comps = self._path_components()
        S = (self._sample_paths(comps, m, rng) if comps is not None
             else self._sample_gibbs(m, rng))
        return S.astype(np.float64)

    def edge_corr(self, m: int = 200_000, seed: int = 0) -> float:
        X = self.sample(m, np.random.default_rng(seed))
        e = np.argwhere(np.triu(self.A, 1) > 0)
        return float(np.mean([np.corrcoef(X[:, i], X[:, j])[0, 1] for i, j in e]))

    def truth(self, m: int = 400_000, seed: int = 0) -> np.ndarray:
        """
        Ising has no closed-form pairwise correlation on a general graph, and
        crucially NO exact off-graph zero -- so `truth` is estimated, and the
        off-graph entries are genuinely non-zero.  This field is used for the
        CONDITIONAL null; do not read its off-graph entries as an exact null.
        """
        X = self.sample(m, np.random.default_rng(seed))
        return np.corrcoef(X.T)

    @property
    def tag(self) -> str:
        return f"ising beta={self.beta:+.4f}"
