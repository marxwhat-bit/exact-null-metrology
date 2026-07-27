# Changelog

## v1.0.0 — 2026-07-27

First public release: the calibrated instrument and the two gate reports it is
certified by.

**Gates archived**

- **GM1** (2026-07-26) — core instrument on Gaussian / binary / q-ary fields.
  PASS. Coverage 0.948–0.951 over four zoo instances (including the binary
  `cycle9` at its certified extremal ε = 22/45), FWER 0.025–0.060,
  injected-leakage power 0→1 monotone and localized at off-target FPR ≤ 8e-4,
  AR(1) `ρ^d` recovery 12/12, separator test nominal on GMRF (0.061) and firing
  on EBF (1.000). `instrument/gm1_report.json`.
- **GM1-VQ** (2026-07-27) — the same instrument in an image-VQ token space
  (aMUSEd-256, 8192 codes, 64-d). PASS. Coverage 0.948–0.951 over
  `path_b1`/`skip2`/`grid8x8`/`star16`, FWER 0.015–0.065, power 0→1 at
  FPR ≤ 7e-4, AR(1) 12/12 against the *projected* target, separator 0.050 (PC1)
  / 0.061 (PC1–3) with the EBF arm at 1.000. `instrument/gm1_vq_report.json`.

**Contents**

`graphs` · `ebf` · `estimators` · `bound` · `vq` · `parallel_sampler` ·
`gm1_calibration` · `gm1_vq` · `test_instrument` · `fetch_codebook`.
Pure numpy/scipy, CPU-only.

**Design decisions worth recording**

- **Two exact nulls, not one.** The covariance-sparse EBF gives exact *marginal*
  independence off-graph; the precision-sparse GMRF gives exact *conditional*
  independence across separators. An EBF is not a Markov field, so a scheduler
  can improve one null while leaving the other intact or worse — a single null
  would have hidden that.
- **No silent PSD repair.** `assert_feasible` refuses an infeasible ε rather
  than nudging the covariance to the nearest PSD matrix. The nudge was caught
  during GM1 development; it would have destroyed the exact off-graph zeros the
  whole instrument rests on, while still returning plausible numbers.
- **Calibrated p-values and calibrated intervals.** Plain Monte-Carlo p-values
  have too little resolution for BH-FDR to have power at these replicate counts,
  and Fisher-z is invalid in the binary regime (×1.076 error at `cycle9`'s
  ε = 22/45). Both were found by the gate, not by inspection.
- **Hermite variance taken in closed form, not from the truncated sum.**
  Normalising by the truncated `Σ b_k²` inflated the VQ edge target by ~1.5%:
  for a step function `b_k²` decays only like `k^{-3/2}`, so even K = 200
  under-counts, and the deficit lands on the low-`k` terms carrying the largest
  `ρ^k`. Caught by a 4M-draw Monte-Carlo cross-check (q = 4, ρ = 0.45: Hermite
  0.37604 vs direct table 0.37023 vs MC 0.36970); post-fix the Hermite target
  matches the direct table to five decimals at every cell.
- **The aMUSEd codebook is fetched, not redistributed** — it is a slice of
  openrail++ model weights, and shipping it would attach that licence's use
  restrictions to this archive.

**Known scope limits**

- Exact nulls apply to the controlled setting only; real models have no
  ground-truth graph and are not covered by these gates.
- Image VQ is cleared; **video VQ is not** and requires its own gate. Binarising
  destroys the conditional null (median |pcorr| 0.0696) — a coarse video
  tokenizer may not have a usable conditional null at all.
- Bit-identical report reproduction is claimed for the reference environment
  (python 3.13.9, numpy 2.4.2, scipy 1.17.0, scipy-openblas 0.3.31). Elsewhere,
  the gate thresholds are the criterion.
