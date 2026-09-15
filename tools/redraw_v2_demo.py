#!/usr/bin/env python3
"""Make a small, conservative crispness demo from icons-256-v2.

This is intentionally *not* a generative redraw: the pixels are kept in place and
only local high-frequency detail is restored.  A soft threshold ignores JPEG/WebP
noise, the addition is clamped to avoid halos, and the black backdrop is pinned to
its source.  The result is useful when we want a sharper icon without changing its
silhouette, colours, or composition.

Example:
    .venv/bin/python tools/redraw_v2_demo.py
    .venv/bin/python tools/redraw_v2_demo.py --out /tmp/v2-demo
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools import imgproc as I

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

# Conservative because v2 already includes the set-level `punch` pass.
SHARPEN = dict(sigma=0.72, amount=0.48, thr=0.006, knee=0.028, clamp=0.026)


def crisp_v2(img: Image.Image) -> np.ndarray:
    """Sharpen only existing detail; preserve layout and colour structure."""
    src = np.asarray(img.convert("RGB"), dtype=np.uint8)
    x = I.to_f(src)
    y = I.masked_unsharp(x, **SHARPEN)
    # Do not let a sharpen operation create a new halo in the tile background.
    y = I.keep_black(y, x, knee=0.025)
    return I.to_u8(np.clip(y, 0.0, 1.0))


def font(path: str, size: int):
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        return ImageFont.load_default()


def make_sheet(rows, out: Path) -> None:
    """Render original, crisp version, and amplified changed pixels."""
    regular = font("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    bold = font("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    title = font("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 26)

    tile = 256
    label_w = 220
    gap = 16
    top = 96
    row_h = tile + 48
    W = label_w + 3 * tile + 4 * gap
    H = top + len(rows) * row_h + gap
    sheet = Image.new("RGB", (W, H), (20, 23, 32))
    draw = ImageDraw.Draw(sheet)
    draw.text((gap, 15), "256 v2 — чётче без перерисовки", font=title, fill=(242, 244, 250))
    draw.text(
        (gap, 53),
        "Исходный рисунок  |  edge-safe crisp  |  усиленная разница ×6",
        font=regular,
        fill=(157, 164, 180),
    )
    headers = ["icons-256-v2", "точечная дорисовка", "что изменилось"]
    for col, text in enumerate(headers):
        x = label_w + gap + col * (tile + gap)
        draw.text((x, top - 28), text, font=regular, fill=(181, 188, 205))

    for idx, row in enumerate(rows):
        y = top + idx * row_h
        original = Image.fromarray(row["original"])
        crisp = Image.fromarray(row["crisp"])
        diff = np.abs(row["crisp"].astype(np.int16) - row["original"].astype(np.int16))
        # Neutral grayscale difference makes it clear that the silhouette did not move.
        diff_luma = np.clip(diff.mean(axis=2) * 6.0, 0, 255).astype(np.uint8)
        diff_rgb = np.repeat(diff_luma[..., None], 3, axis=2)
        difference = Image.fromarray(diff_rgb)

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
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    specs = DEFAULTS
    if args.icon:
        specs = []
        for rel in args.icon:
            rel_path = Path(rel)
            # Include the pack in the output name so icon_001 from two packs
            # cannot overwrite its neighbour.
            label = f"{rel_path.parent.parent.name}-{rel_path.stem}".replace(" ", "_")
            specs.append((label, rel))
    for name, rel in specs:
        src_path = source_root / rel
        if not src_path.exists():
            raise FileNotFoundError(src_path)
        original = np.asarray(Image.open(src_path).convert("RGB"), dtype=np.uint8)
        crisp = crisp_v2(Image.fromarray(original))
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
            "changed_px_percent": float((delta.max(axis=2) > 0).mean() * 100.0),
        })

    make_sheet(rows, out / "comparison.png")
    metadata = {
        "source_set": args.set_dir,
        "method": "edge-safe masked unsharp; no resampling, geometry, colour grading, or generative redraw",
        "parameters": SHARPEN,
        "icons": [
            {k: v for k, v in row.items() if k not in {"original", "crisp"}}
            for row in rows
        ],
    }
    (out / "README.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out / 'comparison.png'}")
    for row in rows:
        print(f"{row['name']:7s} Δ={row['mean_delta']:.2f}/255, changed={row['changed_px_percent']:.1f}% -> {row['output']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
