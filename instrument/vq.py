#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The nominal-token (VQ) adaptation of the instrument.

The defect this fixes
---------------------
`estimators.py::pair_corr` is Pearson on raw values.  On a VQ token field the
raw values are **codebook indices 0..8191**, which are *nominal*: index 4711 is
not "larger" than index 12, the codebook's order is an artefact of how k-means
happened to enumerate its centroids.  Pearson over them is meaningless -- and,
worse, it returns a number, so any image/video measurement would have produced
silently wrong statistics.  This module is the fix, and `gm1_vq.py` is the fresh
calibration pass that every new token space must survive before a claim built on
it can leave.

The fix, and why it is exact
----------------------------
Project each code through the model's own codebook embedding
(`vqvae.quantize.embedding.weight`, (8192, 64) for aMUSEd-256) and keep the
leading principal component(s):  code -> a real scalar (or k-vector).

Two facts make this legitimate rather than a heuristic:

1. **The exact null survives, with no argument needed.**  The projection is a
   deterministic *per-position* map.  Functions of independent random variables
   are independent.  So off-graph pairs remain **exactly** independent after
   projection, at every eps, every codebook, every k.  The null -- the thing the
   whole instrument rests on -- is untouched.

2. **The edge target does NOT survive; it must be recomputed.**  Projection is a
   nonlinear map, so the projected edge correlation is not eps.  It is computed
   here in closed form via the Hermite expansion of the quantizer:

       h(z) = level assigned to the cell containing z          (a step function)
       b_k  = <h, He_k>/sqrt(k!)                               (Hermite coeffs)
       corr(h(Z_i), h(Z_j)) = sum_{k>=1} w_k rho^k ,  w_k = b_k^2 / sum_j b_j^2

   Since the w_k are a probability distribution on k >= 1, the projected
   correlation is a convex combination of powers of rho, hence

       |corr_projected| <= |rho|,  with equality iff the quantizer is exact.

   **The VQ projection strictly attenuates the coupling dial, by a factor that
   is computable in advance.**  That is a reportable property of the token
   space, not an estimator error -- but an instrument that did not recompute the
   target would report the attenuation as sampler infidelity.

The synthetic token model
-------------------------
Real VQ assigns codes by nearest-neighbour in embedding space, so the *code
index* is nominal while the *embedding* varies continuously with content.  The
calibration field mirrors exactly that:

    latent  Z ~ N(0, I + eps A)      (the usual EBF/GMRF/AR1 machinery)
    encode  c(z) = argmin_c |p_c - z| where p = standardized PC scores of the
                   real aMUSEd codebook  ->  a nearest-neighbour quantizer
    observe the CODE INDEX c(z)      -- nominal, arbitrary integer
    project p_{c(z)}                 -- monotone step function of z

so `pair_corr` on the observed indices is broken (demonstrated in `main`), and
`pair_corr` on the projection is correct with a computable target.

Real image models have no ground truth, so they can only be measured with
permutation nulls and known-structure probes.  This synthetic field calibrates
the *estimator path* instead, which is what the gate asks of it.
"""
from __future__ import annotations

import os

import numpy as np
from scipy.stats import norm

__all__ = ["load_codebook", "codebook_pcs", "Quantizer", "VQField",
           "hermite_weights", "projected_corr", "projected_truth"]

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CODEBOOK = os.path.join(_HERE, "amused256_codebook.npy")


# --------------------------------------------------------------- the codebook
def load_codebook(path: str = DEFAULT_CODEBOOK) -> np.ndarray:
    """The model's own embedding table, (n_codes, d)."""
    E = np.load(path)
    if E.ndim != 2:
        raise ValueError(f"codebook must be 2-D, got {E.shape}")
    return np.asarray(E, dtype=np.float64)


def codebook_pcs(E: np.ndarray, k: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """
    PC scores of every code, and the explained-variance ratios.

    Returns (scores (n_codes, k) standardized to unit variance, evr (d,)).
    Standardizing is safe: correlation is scale-invariant, and it puts the
    quantizer levels on the same scale as the N(0,1) latent.
    """
    Ec = E - E.mean(0, keepdims=True)
    U, S, _ = np.linalg.svd(Ec, full_matrices=False)
    evr = (S ** 2) / (S ** 2).sum()
    sc = U[:, :k] * S[:k]
    return sc / sc.std(0, keepdims=True), evr


# ---------------------------------------------------------------- quantizer
class Quantizer:
    """
    Nearest-neighbour scalar quantizer whose levels are real codebook PC scores.

    `levels[c]` is the projected value of code `c`; codes are stored in their
    ORIGINAL codebook order, so the emitted index carries no ordinal
    information -- which is the whole point.
    """

    def __init__(self, levels: np.ndarray, code_ids: np.ndarray | None = None):
        levels = np.asarray(levels, dtype=np.float64).ravel()
        self.q = len(levels)
        order = np.argsort(levels)                  # rank -> position in `levels`
        self.order = order
        self.sorted_levels = levels[order]          # v_0 < v_1 < ... < v_{q-1}
        self.code_ids = (np.arange(self.q) if code_ids is None
                         else np.asarray(code_ids, dtype=np.int64))[order]
        self.cuts = 0.5 * (self.sorted_levels[1:] + self.sorted_levels[:-1])
        self._w = None

    # -- encode / project ---------------------------------------------------
    def rank(self, z: np.ndarray) -> np.ndarray:
        """Cell index (ordinal, internal)."""
        return np.searchsorted(self.cuts, z)

    def encode(self, z: np.ndarray) -> np.ndarray:
        """The observable: a NOMINAL codebook index."""
        return self.code_ids[self.rank(z)]

    def project(self, codes: np.ndarray) -> np.ndarray:
        """codes -> scalar.  The estimator path's entry point."""
        lut = np.empty(int(self.code_ids.max()) + 1)
        lut[self.code_ids] = self.sorted_levels
        return lut[np.asarray(codes, dtype=np.int64)]

    def h(self, z: np.ndarray) -> np.ndarray:
        """encode-then-project, without materialising the codes."""
        return self.sorted_levels[self.rank(z)]

    # -- exact spectrum of the quantizer ------------------------------------
    def weights(self, K: int = 64) -> np.ndarray:
        """Hermite weights w_k, k = 1..K (a probability vector)."""
        if self._w is None or len(self._w) != K:
            self._w = hermite_weights(self.sorted_levels, self.cuts, K)
        return self._w

    def attenuation(self, rho: float, K: int = 64) -> float:
        """corr_projected(rho) / rho -- always in (0, 1] for rho != 0."""
        return projected_corr(rho, self.weights(K)) / rho if rho else 1.0


def hermite_weights(levels: np.ndarray, cuts: np.ndarray, K: int = 64) -> np.ndarray:
    """
    Hermite spectrum of the step function h(z) = levels[cell(z)].

    Telescoping the cell integrals (boundary terms vanish at +-inf) gives

        a_k = sum_j (v_j - v_{j-1}) He_{k-1}(t_j) phi(t_j),     k >= 1

    accumulated as b_k = a_k/sqrt(k!) with the normalised recurrence
    u_k = He_k/sqrt(k!) so nothing overflows:

        u_{k+1} = t u_k / sqrt(k+1) - u_{k-1} sqrt(k/(k+1))

    Returns w_k = b_k^2 / Var(h),  k = 1..K.

    ⚠️ **The denominator must NOT be sum_k b_k^2.**  For a step function
    b_k^2 decays only like k^{-3/2}, so a truncated sum under-counts the
    variance by ~1-2% at K=200; that inflates every w_k, and since low k carry
    the largest rho^k the projected target comes out ~1.5% too high.  Caught by
    a 4M-draw Monte Carlo (q=4, rho=0.45: Hermite 0.37604 vs direct 0.37023 vs
    MC 0.36970) -- an error that would have been read as sampler infidelity.
    The variance is therefore taken in CLOSED FORM from the cell probabilities,
    where it is exact, and only the numerator is truncated.  That is safe: the
    dropped terms carry a factor rho^k, so at rho <= 0.5 and K = 64 the omitted
    mass is below 1e-19.

    `sum(w)` is thus <= 1 by construction, and `1 - sum(w)` is the reportable
    high-order residual.
    """
    v = np.asarray(levels, dtype=np.float64)
    t = np.asarray(cuts, dtype=np.float64)
    dv, phi = np.diff(v), norm.pdf(t)

    # exact variance from the cell table
    edges = np.concatenate([[-np.inf], t, [np.inf]])
    p = np.diff(norm.cdf(edges))
    mu = float(v @ p)
    var = float(((v - mu) ** 2) @ p)
    if var <= 0:
        raise ValueError("degenerate quantizer: zero variance after projection")

    b = np.empty(K)
    u_prev = np.zeros_like(t)
    u_cur = np.ones_like(t)                  # u_0 = 1
    for k in range(1, K + 1):
        b[k - 1] = float(np.sum(dv * u_cur * phi)) / np.sqrt(k)
        u_next = (t * u_cur / np.sqrt(k) - u_prev * np.sqrt((k - 1) / k)) if k > 1 \
            else t * u_cur / np.sqrt(k)
        u_prev, u_cur = u_cur, u_next
    return (b ** 2) / var


def projected_corr(rho, w: np.ndarray):
    """corr after projection = sum_k w_k rho^k.  Exactly 0 at rho = 0."""
    r = np.asarray(rho, dtype=np.float64)
    k = np.arange(1, len(w) + 1)
    return np.tensordot(r[..., None] ** k, w, axes=([-1], [0]))[()]


def projected_truth(R_latent: np.ndarray, w: np.ndarray) -> np.ndarray:
    """
    Exact projected correlation matrix from the latent one.

    Off-graph entries are 0 in R_latent, and sum_k w_k 0^k = 0 EXACTLY -- the
    null is preserved by construction, not by approximation.
    """
    R = np.asarray(R_latent, dtype=np.float64)
    out = np.zeros_like(R)
    P = np.ones_like(R)
    for k in range(1, len(w) + 1):
        P = P * R
        out += w[k - 1] * P
    np.fill_diagonal(out, 1.0)
    return out


# -------------------------------------------------------------------- field
class VQField:
    """
    Wraps ANY latent field from `ebf.py` (EBF / GMRF / LeakyEBF / AR1Copula) and
    emits nominal VQ codes, with `sample()` returning the PROJECTED scalars so
    every existing estimator applies unchanged.

    `truth()` returns the exact projected correlation matrix -- which is the
    thing this whole module exists to get right.
    """

    def __init__(self, latent_field, quant: Quantizer, K: int = 64):
        self.f, self.quant, self.K = latent_field, quant, K
        self.n = latent_field.n
        self.w = quant.weights(K)

    def codes(self, m: int, rng: np.random.Generator) -> np.ndarray:
        return self.quant.encode(self.f.latent(m, rng))

    def sample(self, m: int, rng: np.random.Generator) -> np.ndarray:
        """Projected scalars -- the estimator path."""
        return self.quant.h(self.f.latent(m, rng))

    def latent_truth(self) -> np.ndarray:
        f = self.f
        if hasattr(f, "Sigma") and f.Sigma is not None:
            return np.asarray(f.Sigma, dtype=np.float64)
        raise AttributeError("latent field exposes no Sigma")

    def truth(self) -> np.ndarray:
        return projected_truth(self.latent_truth(), self.w)

    @property
    def tag(self) -> str:
        return f"vq[q={self.quant.q}] <- {getattr(self.f, 'tag', type(self.f).__name__)}"


# --------------------------------------------------------------------- main
def _direct_corr(levels: np.ndarray, cuts: np.ndarray, rho: float) -> float:
    """Brute-force projected correlation from the exact cell table (small q)."""
    from ebf import bvn_cdf
    t = np.concatenate([[-np.inf], cuts, [np.inf]])
    q = len(levels)
    F = np.zeros((q + 1, q + 1))
    for i, a in enumerate(t):
        for j, b in enumerate(t):
            if a == -np.inf or b == -np.inf:
                F[i, j] = 0.0
            elif a == np.inf and b == np.inf:
                F[i, j] = 1.0
            elif a == np.inf:
                F[i, j] = float(norm.cdf(b))
            elif b == np.inf:
                F[i, j] = float(norm.cdf(a))
            else:
                F[i, j] = bvn_cdf(a, b, rho)
    P = F[1:, 1:] - F[:-1, 1:] - F[1:, :-1] + F[:-1, :-1]
    P = np.clip(P, 0, None); P /= P.sum()
    pi = P.sum(1)
    mu = float(levels @ pi)
    var = float(((levels - mu) ** 2) @ pi)
    cov = float((levels - mu) @ P @ (levels - mu))
    return cov / var


if __name__ == "__main__":
    import sys
    sys.path.insert(0, _HERE)
    from ebf import EBF
    from graphs import build_graph, edge_list, nonedge_list
    from estimators import pair_corr

    E = load_codebook()
    sc, evr = codebook_pcs(E, 3)
    print(f"codebook {E.shape}  PC1 {evr[0]:.4f}  PC1-3 {evr[:3].sum():.4f}")

    rng = np.random.default_rng(0)
    A = build_graph("path_b1")
    e, ne = edge_list(A), nonedge_list(A)

    for q, label in ((8192, "full codebook"), (256, "subsampled"), (16, "coarse")):
        ids = np.arange(len(sc)) if q == len(sc) else \
            rng.choice(len(sc), q, replace=False)
        Q = Quantizer(sc[ids, 0], ids)
        w = Q.weights(64)
        rho = 0.45
        tgt = projected_corr(rho, w)
        line = (f"q={q:<5} ({label:<14}) attenuation={tgt/rho:.4f}  "
                f"target={tgt:+.4f}  w1={w[0]:.4f} tail(k>4)={w[4:].sum():.2e}")
        if q <= 16:
            line += f"  direct={_direct_corr(Q.sorted_levels, Q.cuts, rho):+.4f}"
        print(line)

    # the defect and the fix, on the same draw
    ids = np.arange(len(sc))
    Q = Quantizer(sc[:, 0], ids)
    fld = VQField(EBF(A, 0.45), Q)
    codes = fld.codes(200_000, rng)
    proj = Q.project(codes)
    tr = fld.truth()
    print("\n           edge rho_hat   target      max|off-graph|")
    for nm, X in (("raw indices", codes.astype(float)), ("projected", proj)):
        R = pair_corr(X, e)
        print(f"{nm:<12}  {R.mean():+.4f}     "
              f"{tr[e[0,0],e[0,1]]:+.4f}      {np.abs(pair_corr(X, ne)).max():.4f}")
