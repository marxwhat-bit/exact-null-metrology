#!/usr/bin/env python3
"""
Fetch the aMUSEd-256 VQ codebook that GATE GM1-VQ calibrates against.

The table itself is NOT redistributed here. It is a slice of the aMUSEd-256
model weights, which are released under the CreativeML Open RAIL++-M licence;
shipping it would make this archive a Derivative of the Model and push that
licence's use restrictions onto everyone who downloads this record. So we ship
the fetch path and a checksum instead, and this archive stays plain MIT.

What it does: downloads `vqvae/diffusion_pytorch_model.safetensors` from the
`amused/amused-256` repo on the Hugging Face Hub, extracts the single tensor
`quantize.embedding.weight` (8192 codes x 64 dims), and writes it as
`amused256_codebook.npy` in float64 -- the exact file `vq.load_codebook` expects.

Verified 2026-07-27: this path reproduces the table used for the published
GM1-VQ run **byte for byte**, SHA256 below. Only the model weights are
downloaded; nothing is executed from the Hub and torch/diffusers are not needed.

    python3 fetch_codebook.py            # fetch, write, verify the checksum
    python3 fetch_codebook.py --verify   # only check an existing file

Requires: huggingface_hub, safetensors, numpy (see ../requirements.txt).
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys

import numpy as np

REPO = "amused/amused-256"
WEIGHTS = "vqvae/diffusion_pytorch_model.safetensors"
TENSOR = "quantize.embedding.weight"
SHAPE = (8192, 64)

# SHA256 of the .npy this script writes, as used for the published GM1-VQ report.
EXPECTED_SHA256 = "a3d4b5157cc81b8a557e947b17d87451a61726625ef386a2ff667d69b1042f04"

_HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(_HERE, "amused256_codebook.npy")


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(path: str) -> int:
    if not os.path.exists(path):
        print(f"missing: {path}  (run without --verify to fetch it)")
        return 1
    got = sha256(path)
    ok = got == EXPECTED_SHA256
    print(f"{'OK  ' if ok else 'FAIL'} {os.path.basename(path)}  sha256={got}")
    if not ok:
        print(f"     expected {EXPECTED_SHA256}")
        print("     GM1-VQ numbers will not match the published report.")
    return 0 if ok else 1


def fetch() -> int:
    try:
        from huggingface_hub import hf_hub_download
        from safetensors import safe_open
    except ImportError as e:
        print(f"need huggingface_hub and safetensors: {e}")
        print("  pip install huggingface_hub safetensors")
        return 2

    print(f"downloading {REPO}/{WEIGHTS} ...")
    path = hf_hub_download(REPO, WEIGHTS)
    with safe_open(path, framework="numpy") as f:
        if TENSOR not in f.keys():
            print(f"tensor {TENSOR!r} not found in {path}")
            return 2
        E = f.get_tensor(TENSOR)

    if tuple(E.shape) != SHAPE:
        print(f"unexpected shape {E.shape}, expected {SHAPE}")
        return 2

    # float64 is what load_codebook() uses; the cast from the stored float32 is
    # exact, so the checksum is stable.
    np.save(OUT, E.astype(np.float64))
    print(f"wrote {OUT}  {SHAPE[0]}x{SHAPE[1]} float64")
    return verify(OUT)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true",
                    help="check an existing codebook instead of fetching")
    a = ap.parse_args()
    return verify(OUT) if a.verify else fetch()


if __name__ == "__main__":
    sys.exit(main())
