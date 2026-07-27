# Exact-null dependence metrology for parallel decoders

**A calibrated measurement instrument, not a method.** It measures how much
statistical dependence a parallel/masked decoder destroys or manufactures, on
fields whose dependence structure is known **exactly** — every non-edge is
independent by construction, not approximately, so any dependence found off the
graph was created by the system under test.

Two calibration gates, both passed and both archived here with their numbers:

| gate | subject | date | verdict |
|---|---|---|---|
| **GM1** | the core instrument, on Gaussian / binary / q-ary fields | 2026-07-26 | ✅ PASS |
| **GM1-VQ** | the same instrument re-calibrated for an **image-VQ token space** (aMUSEd-256, 8192 codes) | 2026-07-27 | ✅ PASS |

Both gates are seeded. On the reference environment they reproduce
`gm1_report.json` and `gm1_vq_report.json` **bit-identically** — every field but
the wall-clock. `./reproduce.sh` checks exactly that and fails if anything moved.

---

## Why an exact null

The theory of parallel-decoding error exists (ParallelBench's Thm-1 floor and
its matching upper bounds) and the methods exist (confidence, threshold, dilated,
independent-set, low-discrepancy schedules). What did not exist is the
**measurement**: a null against which realized per-edge dependence error can be
tested with false-positive control.

Heuristic nulls cannot do this job. A permutation or shuffled-graph null is
approximately zero and its residual is confounded with the effect. A field whose
off-graph independence is **exact by construction** has no heuristic substitute:
the null is zero, so a rejection is attributable, and a power curve is meaningful.

Two exact nulls ship here, because "dependence" is two different claims:

- **EBF** — covariance-sparse ε-biased field. Non-edges are exactly **marginally**
  independent. Answers *"did the decoder invent correlation where there was none?"*
- **GMRF** — precision-sparse field, Σ = (I − εA)⁻¹. Non-edges are exactly
  **conditionally** independent across any separator. Answers the conditional
  claim — the one an "attention ≈ dependence" argument is actually making.

An EBF is *not* a Markov field, which is the point: the two nulls fail in
different places, and a scheduler can improve one while leaving the other intact
or worse.

The **ε dial** is the second half. Every published ground-truth construction we
know of is a deterministic constraint — coupling is always maximal — so nobody
can ask at *what coupling strength* a scheduler breaks. Here coupling is a
continuous parameter with an exact feasible window
`ε ∈ [−1/λ_max(A), +1/|λ_min(A)|]`, both signs valid, with certified extremal
values in the binary regime (odd cycles `MVE(C₉) = 22/45`, stars `≤ 1/√m`).

---

## The certified numbers

### GATE GM1 — `instrument/gm1_report.json` (m = 8000, 200 replicates, α = 0.05)

**(a) Nominal coverage and family-wise error, on the exact field**

| graph | regime | ε | edge-CI coverage | FWER |
|---|---|---|---|---|
| `path_b1` | gaussian | 0.45 | 0.951 | 0.045 |
| `skip2` | gaussian | 0.45 | 0.949 | 0.030 |
| `cycle9` | **binary** | 22/45 = 0.4889 | 0.950 | 0.060 |
| `star16` | gaussian | 0.24 | 0.948 | 0.025 |

**(b) Injected-leakage power** (`path_b1`, ε = 0.45, 4 injected non-edges)

| injected δ | 0.00 | 0.02 | 0.04 | 0.06 | 0.10 | 0.20 |
|---|---|---|---|---|---|---|
| power | 0.000 | 0.022 | 0.477 | 0.983 | 1.000 | 1.000 |
| off-target FPR | 6.5e-5 | 1.3e-4 | 3.3e-4 | 4.6e-4 | 8.0e-4 | 7.2e-4 |

Monotone, reaches full power, and **localized** — the rejections land on the
injected pairs, not scattered.

**(c) AR(1) known-answer recovery** — 12/12 cells cover `ρ^d` (ρ = ±0.5, lags 1–6).

**(d) Separator conditional-independence test** — nominal on GMRF where the null
is TRUE (rejection 0.061, KS-uniform p = 0.52) and fires at 1.000 on EBF where
the null is FALSE. A test that never fires is not a test.

### GATE GM1-VQ — `instrument/gm1_vq_report.json` (same design, image-VQ token space)

VQ codes are **nominal**: Pearson on codebook indices 0–8191 is not noisy, it is
blind — and it silently returns a number. Measured on the same 200k draws,
`path_b1`, ε = 0.45, true edge correlation +0.4497:

| estimator input | edge ρ̂ | max off-graph \|ρ̂\| |
|---|---|---|
| raw nominal code indices | **+0.0010** | 0.0075 |
| PC1 projection of the same codes | **+0.4501** | 0.0080 |

The fix projects each position through the model's own embedding table
(PC1 = 62.0% of variance, PC1–3 = 89.6%). After it:

| graph | ε | projected target | coverage | FWER |
|---|---|---|---|---|
| `path_b1` | 0.45 | +0.4497 | 0.951 | 0.015 |
| `skip2` | 0.45 | +0.4497 | 0.948 | 0.030 |
| `grid8x8` | 0.24 | +0.2398 | 0.949 | 0.065 |
| `star16` | 0.24 | +0.2398 | 0.950 | 0.040 |

Power 0 → 0.052 → 0.512 → 0.968 → 1.000 → 1.000 at FPR ≤ 6.9e-4; AR(1) 12/12
against the *projected* target; separator nominal at 0.050 (PC1) / 0.061 (PC1–3)
with the EBF arm firing at 1.000.

**Two facts that transfer to any future token space.**

1. **The exact null survives the fix by construction.** The projection is a
   deterministic *per-position* map, and functions of independent variables are
   independent — so off-graph pairs remain exactly independent at every ε, every
   codebook, every k. No "approximately independent" argument is needed anywhere.

2. **The edge target does not survive, and has a closed form.** With the
   quantizer's Hermite spectrum `w_k`, `corr_proj(ρ) = Σ_{k≥1} w_k ρ^k`. Since
   `w` is a probability vector, **|corr_proj| ≤ |ρ|: a token space strictly
   attenuates the ε dial**, by a factor computable in advance:

   | codebook size | 8192 | 1024 | 256 | 64 | 16 |
   |---|---|---|---|---|---|
   | attenuation @ ε = 0.45 | ×0.9994 | ×0.9990 | ×0.9972 | ×0.9859 | ×0.9698 |

   Negligible on the real codebook, but an instrument that skipped this would
   have reported quantizer attenuation as sampler infidelity.

**Bonus, and it scopes every conditional claim on discrete tokens.** Binarising
(q = 2) destroys the GMRF conditional null — median |partial corr| **0.0696**,
about 23% of the edge signal, no usable null. At real codebook resolution it is
**rescued**: **0.0081**. Conditional-null validity is a function of codebook
resolution, measured rather than assumed.

---

## What this does *not* claim

Stated up front, because the scope is the contribution.

- **The exact nulls live in the controlled setting only.** Real models — real
  dLLMs, real image generators — have no ground-truth dependence graph. Against
  them this instrument offers permutation nulls, separator tests, matched-pairs
  co-commit and known-structure probes; those are ordinary statistics, not exact
  nulls, and are not what these two gates certify.
- **Every new token space needs its own gate.** Image VQ is cleared here.
  **Video VQ is not** — a coarse video tokenizer may not have a usable
  conditional null at all (see the q = 2 result above).
- **The mathematics is not the novelty.** The feasible-ε framework is the cut
  polytope / elliptope (Huber–Marić; Deza–Laurent). It is the design principle,
  not a claim.
- **In continuous space a band-limited Gaussian field is just an MA filter**
  (verified: MA(1) max lag-1 correlation 0.5000 = the ceiling). What is not
  reproducible by a filter, and what this archive is about, is the **token-space**
  exact null.
- **Bit-identical reproduction is claimed for the reference environment**
  (python 3.13.9 / numpy 2.4.2 / scipy 1.17.0, scipy-openblas 0.3.31). Elsewhere
  the gate *thresholds* are what must hold; small last-digit drift from a
  different BLAS is expected and is not a failure.

---

## Reproducing

```bash
pip install -r requirements.txt
./reproduce.sh              # contract tests + GM1 + GM1-VQ, ~6 min, CPU only
./reproduce.sh --no-vq      # skip GM1-VQ (no network needed)
```

`reproduce.sh` runs each gate into a temp file and diffs it field-by-field
against the archived report. It exits non-zero if any gate fails **or** if any
archived number fails to reproduce.

Or run the pieces directly:

```bash
cd instrument
python3 test_instrument.py      # contract tests: exact nulls, marginals, monotonicity
python3 gm1_calibration.py      # GATE GM1     -> gm1_report.json   (~70 s)
python3 fetch_codebook.py       # get the aMUSEd-256 codebook (see below)
python3 gm1_vq.py               # GATE GM1-VQ  -> gm1_vq_report.json (~180 s)
```

Several modules also print a summary table when run directly — `graphs.py` (the
zoo and its ε windows), `bound.py` (Thm-1 floors per graph × K),
`parallel_sampler.py` (floor / deficit / co-commit split per schedule × K),
`vq.py` (the nominal-index failure demonstrated).

### The aMUSEd codebook

GATE GM1-VQ calibrates against the **real** aMUSEd-256 VQ embedding table
(8192 × 64). That table is **not redistributed here**: it is a slice of model
weights under the CreativeML Open RAIL++-M licence, and shipping it would attach
that licence's use restrictions to this whole archive.

`instrument/fetch_codebook.py` downloads it from the Hugging Face Hub under the
model's own terms and verifies the checksum. Verified 2026-07-27: this path
reproduces the exact table used for the published run, byte for byte —
SHA256 `a3d4b5157cc81b8a557e947b17d87451a61726625ef386a2ff667d69b1042f04`.

---

## What is in here

| file | what it does |
|---|---|
| `graphs.py` | the graph zoo (`path_b1/b2`, `skip2/4`, `grid8x8`, `star16`, `cycle9/13`, `reg3_32`, `path_b1_shuf`), `spectral_ceiling` = 1/\|λ_min\|, `eps_window` = the exact signed feasible window, certified binary extremals |
| `ebf.py` | the fields and their exact ground truth. `EBF` (covariance-sparse ⇒ exact marginal independence off-graph), `GMRF` (precision-sparse ⇒ exact conditional independence across separators), `LeakyEBF` (injected off-graph leakage — the power standard), `AR1Copula` (the matched-Markov contrast). `assert_feasible` **refuses** an infeasible ε instead of repairing it: a nearest-PSD nudge would silently destroy the exact zeros everything else rests on |
| `estimators.py` | the instrument proper: `edge_fidelity` (ρ̂ vs ε, signed bias, bootstrap CI), `leakage_test` (max-statistic FWER + BH-FDR over non-edges, exact-null and permutation variants), `calibrated_p` (plain MC p-values have too little resolution for BH), `calibrated_edge_ci` (normal theory undercovers in the binary regime), `separator_test`, `rho_by_lag`, `co_commit_deficit` |
| `bound.py` | the comparator: ParallelBench Thm-1 floor `Σ_t C(S_t \| S_<t)` in closed form on Gaussian fields, plus the schedule zoo (`sequential`/`oneshot`/`random`/`dilated`/`independent_set`) |
| `vq.py` | the nominal-token adaptation: codebook PCA projection, the Hermite-spectrum edge target `corr_proj`, and a runnable demonstration that Pearson-on-indices is blind |
| `parallel_sampler.py` | the subject: the ideal parallel decoder — exact conditional means, `diag` of the conditional covariance. The joint-vs-marginal mismatch isolated and nothing else |
| `gm1_calibration.py` | GATE GM1 → `gm1_report.json` |
| `gm1_vq.py` | GATE GM1-VQ → `gm1_vq_report.json` |
| `test_instrument.py` | contract tests: ε = 0 bit-identical to iid, exact off-graph zeros in all three regimes, marginal exactness (KS), monotonicity in ε and in injected δ, BH-FDR against a hand-computed answer, floor endpoints, the certified theory values |
| `fetch_codebook.py` | fetches + checksums the aMUSEd-256 codebook (not redistributed) |

**One identity worth knowing before using this.** For the ideal parallel decoder,
`KL(true ‖ parallel)` **equals** the Thm-1 floor exactly (the textbook
`KL(p ‖ Πp_i) = TC(p)`, chained over steps; verified here to 1.8e-15). The bound
is therefore *attained*: there is no realized-vs-floor gap on the ideal decoder.
That is the point — bound slack is exactly zero, so any gap a **real** model
shows is entirely the model's own departure, by construction.

### Deliberately not in this release

This is the calibrated instrument and its two gate reports. Work in flight that
uses it — trained-model sweeps, scheduler comparisons, real-model deficit
maps — is not here and will be released with the paper it belongs to.

---

## Citing

Cite the archived record; `CITATION.cff` carries the machine-readable form and the DOI.

This instrument underpins a research programme on parallel-decoding error whose
paper is in preparation. The archive exists so that the instrument and its
calibration are dated, fixed and citable independently of that paper's timeline —
the measurement device is a contribution in its own right, and it should be
possible to cite it without waiting on, or being scoped by, the findings it
produces.

## Licence

MIT for the software (`instrument/*.py`), CC BY 4.0 for the measurement results
(`instrument/gm1_*.json`, `gm1_vq_run.log`). See `LICENSE` and
`LICENSE-CC-BY-4.0.txt`. The aMUSEd-256 codebook is not redistributed and
remains under its own licence.
