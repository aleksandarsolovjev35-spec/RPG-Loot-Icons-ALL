#!/usr/bin/env python3
"""Build a source-first crispness demo for selected icons-256-v2 icons.

The earlier pass sharpened the already encoded WebP.  This version goes one step
back: it reads the matching full-colour crop from the original sheet, enhances and
resamples that crop once, then applies the same v2 consistency recipe and restores
mid-band detail.  This avoids sharpening WebP blocks and gives a cleaner result.

It is still deliberately non-generative: no pixels are moved, no new object is
invented, and the black backdrop is protected.  The output is a demonstration; the
canonical icons-256-v2 set is never overwritten.

Examples:
    .venv/bin/python tools/redraw_v2_demo.py
    .venv/bin/python tools/redraw_v2_demo.py --out /tmp/v2-demo
    .venv/bin/python tools/redraw_v2_demo.py \
        --icon "RPG Loot Icons 01/part1/icon_001.webp"
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools import cut_icons as C
from tools import imgproc as I
from tools import stylize as S

# A deliberately varied, deterministic sample: thin curves, bottles, scrolls,
# a bow, a hard-edged chest, and organic texture.
DEFAULTS = [
    ("rope", "RPG Loot Icons 20/part2/icon_036.webp"),
    ("bottle", "RPG Loot Icons 26/part1/icon_048.webp"),
    ("scroll", "RPG Loot Icons 12/part2/icon_019.webp"),
    ("bow", "RPG Loot Icons 36/part2/icon_021.webp"),
    ("chest", "RPG Loot Icons 24/part1/icon_011.webp"),
    ("cacao", "RPG Loot Icons 30/part2/icon_028.webp"),
]

# One clean source -> 256 px pass.  Less denoise than the old WebP pass keeps
# brush texture; the anti-ringing clamp and guided contour work stay enabled.
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

# A second, small mid-band pass restores facets and fine painted edges without
# turning the silhouette into a bright pencil outline.
DETAIL = dict(amount=0.52, fine=0.06, coarse=0.06, s1=0.65, s2=1.7, s3=4.5)
FINAL_SHARPEN = dict(sigma=0.72, amount=0.45, thr=0.006, knee=0.028, clamp=0.035)
STYLE_RECIPE = "quiet+flat+vign+fit+punch"


def normalize_rel(rel: str, set_dir: str) -> str:
    """Accept either `pack/...` or `icons-256-v2/pack/...` from the CLI."""
    parts = Path(rel).parts
    prefix = Path(set_dir).parts
    if parts[: len(prefix)] == prefix:
        parts = parts[len(prefix) :]
    return Path(*parts).as_posix()


def load_manifest(set_dir: str):
    path = ROOT / set_dir / "manifest.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {entry["file"]: entry for entry in data.get("icons", [])}


def native_crop(entry):
    """Read the full-colour crop from the original JPG referenced by the manifest."""
    source = entry["source"]
    for index, _xy, crop in C.iter_cell_crops(source, None):
        if index == entry["index"]:
            return crop
    raise RuntimeError(f"icon {entry['file']} was not found in {source}")


def crisp_from_source(entry, fallback: np.ndarray) -> np.ndarray:
    """Rebuild one icon from its source crop, then apply the v2 detail recipe."""
    try:
        crop = native_crop(entry)
    except (KeyError, RuntimeError, TypeError):
        # A custom set without a manifest still gets a useful, local-only pass.
        x = I.to_f(fallback)
        y = I.laplacian_detail(x, **DETAIL)
        y = I.masked_unsharp(y, **FINAL_SHARPEN)
        return I.to_u8(I.keep_black(y, x, knee=0.025))

    base = I.to_f(I.enhance(crop, SOURCE_ENHANCE))
    style = S.build_chain(STYLE_RECIPE)[1]
    y = style(base, {})
    y = I.laplacian_detail(y, **DETAIL)
    y = I.masked_unsharp(y, **FINAL_SHARPEN)
    # Keep the source-derived black tile dead black; no new halo is allowed.
    y = I.keep_black(y, base, knee=0.025)
    return I.to_u8(np.clip(y, 0.0, 1.0))


def font(path: str, size: int):
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        return ImageFont.load_default()


def make_sheet(rows, out: Path) -> None:
    """Render v2, source-first crisp version, and amplified changed pixels."""
    regular = font("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    bold = font("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    title = font("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 26)

    tile = 256
    label_w = 220
    gap = 16
    top = 96
    row_h = tile + 48
    width = label_w + 3 * tile + 4 * gap
    height = top + len(rows) * row_h + gap
    sheet = Image.new("RGB", (width, height), (20, 23, 32))
    draw = ImageDraw.Draw(sheet)
    draw.text((gap, 15), "256 v2 — восстановление деталей", font=title, fill=(242, 244, 250))
    draw.text(
        (gap, 53),
        "v2  |  source-first enhancement  |  усиленная разница ×6",
        font=regular,
        fill=(157, 164, 180),
    )
    headers = ["icons-256-v2", "улучшенная версия", "что изменилось"]
    for col, text in enumerate(headers):
        x = label_w + gap + col * (tile + gap)
        draw.text((x, top - 28), text, font=regular, fill=(181, 188, 205))

    for idx, row in enumerate(rows):
        y = top + idx * row_h
        original = Image.fromarray(row["original"])
        crisp = Image.fromarray(row["crisp"])
        diff = np.abs(row["crisp"].astype(np.int16) - row["original"].astype(np.int16))
        diff_luma = np.clip(diff.mean(axis=2) * 6.0, 0, 255).astype(np.uint8)
        difference = Image.fromarray(np.repeat(diff_luma[..., None], 3, axis=2))

        draw.rectangle((gap, y, gap + 5, y + tile - 1), fill=(92, 110, 148))
        draw.text((gap + 16, y + 8), row["name"], font=bold, fill=(237, 240, 247))
        draw.text((gap + 16, y + 37), row["kind"], font=regular, fill=(157, 164, 180))
        draw.text((gap + 16, y + 68), f"Δ {row['mean_delta']:.2f}/255", font=regular, fill=(137, 185, 166))
        draw.text((gap + 16, y + 91), "силуэт сохранён", font=regular, fill=(137, 185, 166))

        for col, picture in enumerate((original, crisp, difference)):
            x = label_w + gap + col * (tile + gap)
            sheet.paste(picture, (x, y))
            draw.rectangle((x, y, x + tile - 1, y + tile - 1), outline=(75, 82, 100), width=1)

    sheet.save(out, "PNG", optimize=True)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="docs/quality-lab/256-v2-sharp-demo",
                   help="directory for the demo images")
    p.add_argument("--set", default="icons-256-v2", dest="set_dir",
                   help="source icon set")
    p.add_argument("--icon", action="append", default=None,
                   help="relative icon path inside --set; repeat for a custom selection")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    source_root = ROOT / args.set_dir
    manifest = load_manifest(args.set_dir)
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    specs = DEFAULTS
    if args.icon:
        specs = []
        for rel in args.icon:
            rel_norm = normalize_rel(rel, args.set_dir)
            rel_path = Path(rel_norm)
            label = f"{rel_path.parent.parent.name}-{rel_path.stem}".replace(" ", "_")
            specs.append((label, rel_norm))

    rows = []
    for name, rel in specs:
        rel = normalize_rel(rel, args.set_dir)
        src_path = source_root / rel
        if not src_path.exists():
            raise FileNotFoundError(src_path)
        original = np.asarray(Image.open(src_path).convert("RGB"), dtype=np.uint8)
        entry = manifest.get(rel)
        crisp = crisp_from_source(entry, original) if entry else crisp_from_source({}, original)
        out_name = f"{name}-sharp.png"
        Image.fromarray(crisp).save(out / out_name, "PNG", optimize=True)
        delta = np.abs(crisp.astype(np.int16) - original.astype(np.int16))
        rows.append({
            "name": name,
            "kind": {
                "rope": "тонкие линии",
                "bottle": "мягкие градиенты",
                "scroll": "грани и орнамент",
                "bow": "тонкие линии",
                "chest": "крупные края",
                "cacao": "мелкая фактура",
            }.get(name, "выбранная иконка"),
            "source": str(src_path.relative_to(ROOT)),
            "output": str((out / out_name).relative_to(ROOT)),
            "original": original,
            "crisp": crisp,
            "mean_delta": float(delta.mean()),
            "max_delta": int(delta.max()),
            # Ignore one-level codec/rounding noise in this summary.
            "changed_px_percent": float((delta.max(axis=2) > 2).mean() * 100.0),
        })

    make_sheet(rows, out / "comparison.png")
    metadata = {
        "source_set": args.set_dir,
        "method": "source-first full-colour crop -> anti-ringing 256 resample -> v2 style -> mid-band detail; no geometry or generative redraw",
        "style_recipe": STYLE_RECIPE,
        "source_enhance": SOURCE_ENHANCE,
        "detail": DETAIL,
        "final_sharpen": FINAL_SHARPEN,
        "icons": [
            {k: v for k, v in row.items() if k not in {"original", "crisp"}}
            for row in rows
        ],
    }
    (out / "README.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {out / 'comparison.png'}")
    for row in rows:
        print(f"{row['name']:7s} Δ={row['mean_delta']:.2f}/255, changed={row['changed_px_percent']:.1f}% -> {row['output']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
