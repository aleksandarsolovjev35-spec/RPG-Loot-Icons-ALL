#!/usr/bin/env bash
# End-to-end build of the icon sets. Run from the repo root (or anywhere - the
# script cds to its own parent). Every step is the exact command that produced
# the sets currently in the repo; see docs/PIPELINE.md for the reasoning and the
# expected numbers of each step.
#
#   tools/build_set.sh cut            # 82 sheets -> cut_icons/      (4100 palette PNGs)
#   tools/build_set.sh base           # -> icons-256-base/          (clean 256 px WebP, the style source)
#   tools/build_set.sh actual         # base + quiet+vign -> icons-256/   (the actual set)
#   tools/build_set.sh 512            # -> icons-512/               (previous full-growth set)
#   tools/build_set.sh verify         # qa + manifest/md5 verification of both 256 sets
#   tools/build_set.sh audit          # consistency audit of the actual set (spread + outliers)
#   tools/build_set.sh all            # cut -> base -> actual -> verify -> audit
#
# Options:
#   JOBS=2                 parallel encoder processes for repack/stylize
#   PY=/path/to/python     interpreter (default .venv/bin/python)
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-.venv/bin/python}
JOBS=${JOBS:-0}
STYLE=${STYLE:-quiet+vign}

if [ ! -x "$PY" ]; then
  echo "no interpreter at $PY - create one first:" >&2
  echo "  python3 -m venv .venv && .venv/bin/pip install -r tools/requirements.txt" >&2
  exit 2
fi

step_cut() {
  echo "== cut: 82 sheets -> cut_icons/ =="
  "$PY" tools/cut_icons.py
}

step_base() {
  echo "== base: sheets -> icons-256-base/ (clean 256 px WebP q92 + enhance + Laplacian) =="
  "$PY" tools/repack_icons.py --size 256 --quality 92 \
      --denoise 0.014 --steer 1.4 --sharpen 0.28 --sharpen-sigma 1.0 --laplacian 0.70 \
      --out icons-256-base ${JOBS:+--jobs "$JOBS"}
}

step_actual() {
  echo "== actual: icons-256-base/ + $STYLE -> icons-256/ =="
  "$PY" tools/stylize.py --set icons-256-base --apply "$STYLE" --out-dir icons-256 \
      ${JOBS:+--jobs "$JOBS"}
}

step_512() {
  echo "== 512: sheets -> icons-512/ (previous set, q92 + enhance) =="
  "$PY" tools/repack_icons.py --out icons-512 ${JOBS:+--jobs "$JOBS"}
}

step_verify() {
  echo "== verify: qa + manifest contract =="
  "$PY" tools/qa_icons.py icons-256-base webp
  "$PY" tools/qa_icons.py icons-256 webp
  "$PY" tools/verify_set.py icons-256-base
  "$PY" tools/verify_set.py icons-256
  echo "== verify: the actual set really carries the style =="
  "$PY" - <<'EOF'
import json, os, numpy as np
from PIL import Image
base, actual = 'icons-256-base', 'icons-256'
man = json.load(open(os.path.join(actual, 'manifest.json')))
print('  style: %s (from %s)' % ('+'.join(man['style']['presets']), man['derived_from']))
d = []
for e in man['icons'][::200]:
    a = np.asarray(Image.open(os.path.join(base, e['file'])).convert('RGB'), np.int16)
    b = np.asarray(Image.open(os.path.join(actual, e['file'])).convert('RGB'), np.int16)
    d.append(float(np.abs(a - b).mean()))
print('  mean |base - actual| over %d icons: %.1f levels (min %.1f, max %.1f)'
      % (len(d), np.mean(d), min(d), max(d)))
assert np.mean(d) > 3.0, 'the actual set looks identical to the base - style not applied?'
print('  ok')
EOF
}

step_audit() {
  echo "== audit: how consistent the actual set is (see docs/quality-lab/) =="
  "$PY" tools/audit_set.py --set icons-256 --dupes
}

case "${1:-all}" in
  cut)    step_cut ;;
  base)   step_base ;;
  actual) step_actual ;;
  512)    step_512 ;;
  verify) step_verify ;;
  audit)  step_audit ;;
  all)    step_cut; step_base; step_actual; step_verify; step_audit ;;
  *)      sed -n '2,20p' "$0"; exit 2 ;;
esac
echo "done: $1"
