#!/usr/bin/env python3
"""Rebuild the 256 v2 candidate from the original full-colour sheets.

The normal v2 set was encoded after the first enhancement pass.  This builder
keeps the v2 style recipe but starts at the JPG sheet for every icon, applies the
source-first enhancement used by `redraw_v2_demo.py`, and encodes only once to
WebP.  It writes a complete candidate directory with a reproducible manifest.

It is intentionally separate from the canonical `icons-256/` set.  Use a temporary
output, verify it, then merge it into `icons-256-v2/` when the comparison is good.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools import cut_icons as C
from tools import imgproc as I
from tools import stylize as S

STYLE_RECIPE = "quiet+flat+vign+fit+punch"
QUALITY = 92
SOURCE_ENHANCE = dict(
    size=256,
    denoise=0.009,
    denoise_r=1,
    black=0.010,
    kernel="lanczos3",
    clamp=True,
    stretch=False,
    steer=1.2,
    steer_taps=5,
    steer_rho=1.8,
    steer_range=0.026,
    sharpen=0.40,
    sharpen_sigma=0.82,
    sharpen_thr=0.008,
    sharpen_knee=0.032,
    sharpen_clamp=0.038,
    laplacian=0.80,
    post_denoise=0.0,
    native_u8=True,
)
DETAIL = dict(amount=0.52, fine=0.06, coarse=0.06, s1=0.65, s2=1.7, s3=4.5)
FINAL_SHARPEN = dict(sigma=0.72, amount=0.45, thr=0.006, knee=0.028, clamp=0.035)

_WORK_FN = None


def enhance_crop(crop: np.ndarray) -> np.ndarray:
    """Exact source-first path used by the six-icon demo."""
    base = I.to_f(I.enhance(crop, SOURCE_ENHANCE))
    y = _WORK_FN(base, {})
    y = I.laplacian_detail(y, **DETAIL)
    y = I.masked_unsharp(y, **FINAL_SHARPEN)
    y = I.keep_black(y, base, knee=0.025)
    return I.to_u8(np.clip(y, 0.0, 1.0))


def encode_webp(rgb: np.ndarray, quality: int = QUALITY) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(rgb, "RGB").save(buf, "WEBP", quality=quality, method=6)
    return buf.getvalue()


def init_worker():
    global _WORK_FN
    _WORK_FN = S.build_chain(STYLE_RECIPE)[1]


def process_sheet(task):
    """Process one original sheet so the JPG is decoded only once per worker task."""
    source, entries = task
    by_index = {int(e["index"]): e for e in entries}
    got = []
    for index, xy, crop in C.iter_cell_crops(source, None):
        entry = by_index.get(int(index))
        if entry is None:
            continue
        data = encode_webp(enhance_crop(crop))
        got.append((entry["file"], data, hashlib.md5(data).hexdigest(), len(data)))
    expected = {e["index"] for e in entries}
    seen = {next((int(e["index"]) for e in entries if e["file"] == rel), -1) for rel, *_ in got}
    missing = sorted(expected - seen)
    if missing:
        raise RuntimeError(f"{source}: missing icon indices {missing}")
    return got


def build_manifest(old: dict, entries: list[dict], now: str) -> dict:
    man = copy.deepcopy(old)
    man["role"] = "candidate"
    man["note"] = (
        "source-first enhanced candidate: full-colour JPG crop -> anti-ringing "
        "256 resample -> v2 style -> mid-band detail -> WebP q92"
    )
    man["generated"] = now
    man["quality"] = QUALITY
    man["lossless"] = False
    man["size"] = 256
    man["resample"] = "lanczos3 anti-ringing"
    man["enhance"] = {
        "enabled": True,
        "pipeline": "tools/rebuild_v2_source_first.py:source-first",
        "config": SOURCE_ENHANCE,
    }
    man["post_process"] = {
        "style_recipe": STYLE_RECIPE,
        "detail": DETAIL,
        "final_sharpen": FINAL_SHARPEN,
        "background": "keep_black(knee=0.025)",
    }
    style = man.get("style", {})
    style["presets"] = ["quiet", "flat", "vign", "fit", "punch"]
    style["source_set"] = "original JPG sheets"
    style["generated"] = now
    man["style"] = style
    man["derived_from"] = "original JPG sheets"
    man["derived_from_manifest"] = None
    man["icons"] = entries
    man["count"] = len(entries)
    man["total_bytes"] = sum(e["bytes"] for e in entries)
    return man


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default=".cache/icons-256-v2-source-first",
                   help="temporary output directory")
    p.add_argument("--manifest", default="icons-256-v2/manifest.json",
                   help="manifest defining the crop order and metadata")
    p.add_argument("--jobs", type=int, default=2, help="worker processes")
    p.add_argument("--quality", type=int, default=QUALITY, help="WebP quality")
    p.add_argument("--limit-sheets", type=int, default=0,
                   help="smoke test: process only the first N sheets")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    global QUALITY
    QUALITY = args.quality
    old_path = ROOT / args.manifest
    old = json.loads(old_path.read_text(encoding="utf-8"))
    entries = old["icons"]
    by_source = {}
    for e in entries:
        by_source.setdefault(e["source"], []).append(e)
    sources = sorted(by_source)
    if args.limit_sheets:
        sources = sources[: args.limit_sheets]
    if len(sources) != len(by_source):
        selected = {e["file"] for source in sources for e in by_source[source]}
        entries = [e for e in entries if e["file"] in selected]

    out = ROOT / args.out
    if out.exists():
        # Avoid stale files silently entering the manifest.
        for p in sorted(out.rglob("*"), reverse=True):
            if p.is_file() or p.is_symlink():
                p.unlink()
            elif p.is_dir():
                p.rmdir()
    out.mkdir(parents=True, exist_ok=True)
    for e in entries:
        (out / e["file"]).parent.mkdir(parents=True, exist_ok=True)

    tasks = [(source, by_source[source]) for source in sources]
    t0 = time.time()
    got = {}
    jobs = max(1, int(args.jobs))
    print(f"sheets={len(tasks)} icons={len(entries)} jobs={jobs} out={out.relative_to(ROOT)}")
    with ProcessPoolExecutor(max_workers=jobs, initializer=init_worker) as pool:
        futures = [pool.submit(process_sheet, task) for task in tasks]
        for n, future in enumerate(as_completed(futures), 1):
            for rel, data, digest, size in future.result():
                (out / rel).write_bytes(data)
                got[rel] = (digest, size)
            print(f"  {n}/{len(futures)} sheets  {len(got)}/{len(entries)} icons  {time.time() - t0:.0f}s", flush=True)

    if set(got) != {e["file"] for e in entries}:
        missing = sorted({e["file"] for e in entries} - set(got))
        raise RuntimeError(f"missing outputs: {missing[:10]}")

    result_entries = []
    for e in entries:
        rec = dict(e)
        rec["bytes"] = got[e["file"]][1]
        rec["md5"] = got[e["file"]][0]
        result_entries.append(rec)
    now = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    manifest = build_manifest(old, result_entries, now)
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {len(result_entries)} icons, {manifest['total_bytes'] / 2**20:.1f} MB in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
