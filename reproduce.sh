#!/usr/bin/env bash
# Reproduce both gates from scratch and diff the results against the archived
# reports. Both gates are seeded, so a match is expected to be exact.
#
#   ./reproduce.sh            # contract tests + GM1 + GM1-VQ   (~6 min, CPU)
#   ./reproduce.sh --no-vq    # skip GM1-VQ (no network / no codebook)
#
# Exit 0 iff every gate PASSes and every archived number is reproduced.
set -uo pipefail
cd "$(dirname "$0")/instrument"

DO_VQ=1
[ "${1:-}" = "--no-vq" ] && DO_VQ=0
OUT=$(mktemp -d)
FAIL=0

hdr() { printf '\n\033[1m== %s\033[0m\n' "$1"; }
note() { printf '   %s\n' "$1"; }

# Compare two gate reports, ignoring only the wall-clock field.
cmp_report() {
  python3 - "$1" "$2" <<'PY'
import json, sys
a, b = (json.load(open(p)) for p in sys.argv[1:3])
a.pop("seconds", None); b.pop("seconds", None)
if json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True):
    print("   reproduced bit-identically (every field except wall-clock)"); sys.exit(0)
print("   DIFFERS from the archived report:")
for k in sorted(set(a) | set(b)):
    if json.dumps(a.get(k), sort_keys=True) != json.dumps(b.get(k), sort_keys=True):
        print(f"     field: {k}")
sys.exit(1)
PY
}

hdr "environment"
python3 -c "import sys,numpy,scipy;print(f'   python {sys.version.split()[0]}  numpy {numpy.__version__}  scipy {scipy.__version__}')"
note "published run: python 3.13.9  numpy 2.4.2  scipy 1.17.0  (scipy-openblas 0.3.31)"

hdr "contract tests (rule 7: exact nulls, marginal exactness, monotonicity)"
python3 test_instrument.py || FAIL=1

hdr "GATE GM1 -- exact-null metrology core (~70 s)"
python3 gm1_calibration.py --out "$OUT/gm1_report.json" || FAIL=1
cmp_report gm1_report.json "$OUT/gm1_report.json" || FAIL=1

if [ "$DO_VQ" = 1 ]; then
  hdr "GATE GM1-VQ -- image-VQ token space (~180 s)"
  if [ ! -f amused256_codebook.npy ]; then
    note "codebook absent, fetching from the Hugging Face Hub ..."
    python3 fetch_codebook.py || { note "fetch failed; rerun with --no-vq to skip"; FAIL=1; }
  else
    python3 fetch_codebook.py --verify || FAIL=1
  fi
  if [ -f amused256_codebook.npy ]; then
    python3 gm1_vq.py --out "$OUT/gm1_vq_report.json" || FAIL=1
    cmp_report gm1_vq_report.json "$OUT/gm1_vq_report.json" || FAIL=1
  fi
else
  hdr "GATE GM1-VQ -- SKIPPED (--no-vq)"
fi

hdr "verdict"
if [ "$FAIL" = 0 ]; then
  echo "   ALL GATES REPRODUCED"
else
  echo "   REPRODUCTION FAILED -- see above"
fi
rm -rf "$OUT"
exit $FAIL
