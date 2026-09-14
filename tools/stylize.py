"""Artistic re-styling presets for the icons - the "style lab".

`tools/imgproc.py` only *cleans* the artwork (denoise, anti-ringing resample,
staircase removal, masked unsharp, Laplacian detail); it does not change how an
icon reads.  This module is the other half: filters that change the style of the
picture, built on the same numpy primitives so the toolchain still needs nothing
but `pip install pillow numpy`.

Sets in this repo: `icons-256-base/` is the clean enhanced set (repack output,
the source for styling), and `icons-256/` is the *actual* set - that same base
with `quiet+vign` baked in.  So the canonical run is:

    python3 tools/stylize.py --set icons-256-base --apply quiet+vign --out-dir icons-256

and the lab always measures against the clean base:

    python3 tools/stylize.py                       # lab sheet + metrics table
    python3 tools/stylize.py --presets grade,ink   # only some styles
    python3 tools/stylize.py --apply paint --out-dir /tmp/probe --limit 200

Presets (see PRESETS).  Two of them fix set-level inconsistency rather than
adding "art":

    grade    Кисть / градиент - S-curve contrast, vibrance, split tone; the
             look modern RPG art has: same shapes, deliberate light.
    cel      Cel-shading - pre-smooth + banded luminance (5 soft tones) +
             saturation lift; the flat-colour "drawn game art" look.
    ink      Тушь - darkens the contours themselves (coherence-gated gradient).
    rim      Контровой свет - a lit edge on the top-left of every silhouette.
    glow     Свечение - bloom on the bright parts only: potions, gems, magic.
    paint    Живопись - quadrant Kuwahara: facets instead of noise.
    quiet    Тихий фон - pulls the ambient glow *behind* each item towards one
             flat dark tone: at the moment every sheet draws its own coloured
             smoke/aura at its own brightness, which is what makes a set of 4100
             icons look like a pile of separate images in a loot grid.
    vign     Виньетка - fixed corner falloff, the frame reads the same on all.
    warm     Тёплый свет (torchlight) and cold - Ночной холод: two fixed grades.
    palette  Единая палитра - the whole set k-means'd to K colours.
    pixel    Пиксель-арт 64 - 64 px + 32-colour palette, nearest x4 (a different
             art direction, shown for scale).
    loot     Loot-стиль - grade + rim + glow on top of each other.

All numbers in the lab output are measured on the canon `icons-256` set, so the
comparison is against the size the icons are actually shown at.
"""
import argparse
import json
import os
import sys
import time
from collections import OrderedDict

import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    import imgproc as I
    import measure_quality as M
except ImportError:
    from tools import imgproc as I
    from tools import measure_quality as M

ROOT = M.ROOT
SHEET_FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
SHEET_FONT_BOLD = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'

TONE_QUIET = (0.010, 0.013, 0.022)      # flat dark backdrop the styles pull to


# --------------------------------------------------------------------------
# small shared helpers
# --------------------------------------------------------------------------

def rescale_luma(x, L_new, L):
    """Repaint `x` with a new luminance field, keeping its chroma.

    The multiplier is clamped (0.5x .. 2x) and faded out under luma 0.06, so the
    flat black background never turns into coloured blotches.
    """
    r = np.clip(L_new / np.maximum(L, 1e-3), 0.5, 2.0)
    w = I.soft(L, 0.0, 0.06)
    return x * (w * r + (1.0 - w))[..., None]


def saturate(x, k):
    """Scale chroma around the pixel's own luma (k = 1 keeps the colours)."""
    L = I.luma(x)
    return np.clip(L[..., None] + (x - L[..., None]) * k, 0.0, 1.0)


def gradient(x):
    """(magnitude, nx, ny) of the luma gradient; the normal points to the
    brighter side of the edge."""
    L = I.luma(x)
    gy, gx = np.gradient(L)
    mag = np.hypot(gx, gy)
    n = np.maximum(mag, 1e-6)
    return mag, gx / n, gy / n


def hue_shift(x, deg):
    """Rotate hue by `deg` degrees (SDR HSV wheel, saturation/value kept)."""
    if not deg:
        return x
    a = np.asarray(Image.fromarray(I.to_u8(x)).convert('HSV')).astype(np.int16)
    a[..., 0] = (a[..., 0] + int(round(deg * 255.0 / 360.0))) % 256
    return I.to_f(np.asarray(Image.fromarray(a.astype(np.uint8), 'HSV').convert('RGB')))


# --------------------------------------------------------------------------
# presets
# --------------------------------------------------------------------------

def grade(x, ctx, contrast=0.55, sat=1.15, vibrance=0.70,
          shadow=(-0.004, -0.014, 0.032), high=(0.034, 0.012, -0.018)):
    """S-curve contrast + vibrance + split tone (cool shadows, warm highlights)."""
    src = x
    L = I.luma(x)
    Lc = np.clip(L + contrast * (L * L * (3.0 - 2.0 * L) - L), 0.0, 1.0)
    x = rescale_luma(x, Lc, L)

    L2 = I.luma(x)
    c = x - L2[..., None]
    face = 1.0 - I.soft(np.abs(c).max(-1), 0.05, 0.45)     # muted colours get more
    x = L2[..., None] + c * (1.0 + vibrance * face)[..., None]
    x = saturate(x, sat)

    L3 = I.luma(x)
    x = (x + ((1.0 - L3) ** 2)[..., None] * np.array(shadow, np.float32)
           + (L3 ** 2)[..., None] * np.array(high, np.float32))
    # the split tone must not tint the (pure black) background of the sheets
    return hue_shift(np.clip(I.keep_black(x, src, 0.02), 0.0, 1.0), ctx.get('hue_deg', 0))


def cel(x, ctx, bands=5, soft_band=0.55, pre_smooth=0.035, sat=1.20):
    """Banded luminance = flat "drawn" tones, 5 steps from black to white."""
    y = I.guided(x, I.luma(x), r=2, eps=pre_smooth ** 2)
    L = I.luma(y)
    t = np.clip(L, 0.0, 1.0) * bands
    i = np.floor(t)
    f = t - i
    w = 0.5 * soft_band
    Lb = np.clip((i + I.soft(f, 0.5 - w, 0.5 + w)) / bands, 0.0, 1.0)
    return saturate(rescale_luma(y, Lb, L), sat)


def ink(x, ctx, amount=0.80, lo=0.020, hi=0.095, width=1.0, coh_gain=0.6):
    """Darken the contours themselves (gradient gated by tensor coherence)."""
    L = I.luma(x)
    mag, _nx, _ny = gradient(x)
    _l1, coh2, _a, _b = I.orientation(L, rho=1.8)
    line = I.soft(mag, lo, hi) * (1.0 - coh_gain + coh_gain * np.clip(coh2 * 4.0, 0.0, 1.0))
    line = np.clip(I.blur(line, width), 0.0, 1.0)
    return np.clip(x * (1.0 - amount * line)[..., None], 0.0, 1.0)


def rim(x, ctx, amount=0.90, color=(0.55, 0.75, 1.00), lo=0.025, hi=0.115, spread=2.0):
    """Lit edge on the top-left side of the silhouette."""
    mag, nx, ny = gradient(x)
    d = np.array([0.62, 0.78], np.float32)                 # light from top-left
    face = np.clip(nx * d[0] + ny * d[1], 0.0, 1.0) ** 2
    r = I.blur(I.soft(mag, lo, hi) * face, spread)
    out = x + r[..., None] * np.array(color, np.float32) * amount
    return np.clip(I.keep_black(out, x, 0.02), 0.0, 1.0)


def glow(x, ctx, amount=0.85, lo=0.30, hi=0.85, sigma=6.0, gate_lo=0.010, gate_hi=0.070):
    """Bloom around the bright parts only: potions, gems, magic.

    The gate keeps the flat background black, so the halo hugs the item instead
    of raising the whole tile - which is what makes a glow look cheap in a grid.
    """
    L = I.luma(x)
    b = x * I.soft(L, lo, hi)[..., None]
    bl = I.blur(b, sigma)
    add = np.clip(bl * amount, 0.0, 0.8)
    gate = I.soft(I.luma(bl), gate_lo, gate_hi)[..., None]
    return np.clip(1.0 - (1.0 - x) * (1.0 - add * gate), 0.0, 1.0)


def kuwahara(x, r=6, mix=0.90):
    """Quadrant Kuwahara: the flattest of four windows wins - facets, not noise."""
    means, vars_ = [], []
    for dy, dx in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
        s = np.roll(np.roll(x, dy * (r // 2), 0), dx * (r // 2), 1)
        m = I.box(s, r)
        means.append(m)
        vars_.append(np.maximum(I.box(s * s, r) - m * m, 0.0).mean(-1, keepdims=True))
    idx = np.concatenate(vars_, -1).argmin(-1)             # HxW
    pick = np.take_along_axis(np.stack(means, 0), idx[None, :, :, None], 0)[0]
    return x * (1.0 - mix) + pick * mix


def paint(x, ctx, r=6, mix=0.90, detail=0.25, sat=1.16):
    """Painterly: Kuwahara facets + a little detail back + a warm grade."""
    y = kuwahara(x, r=r, mix=mix)
    y = np.clip(y + detail * (x - I.blur(x, 1.2)), 0.0, 1.0)
    return grade(y, ctx, contrast=0.40, sat=sat, vibrance=0.45)


def quiet(x, ctx, amount=0.55, knee=0.35, tone=TONE_QUIET, spread=6.0):
    """Quiet the ambient glow behind the item.

    Everything below `knee` luma (the coloured smoke/aura the sheets draw behind
    their items) is pulled towards one flat dark tone, so in a loot grid the
    items stop sitting on 4100 differently-coloured backdrops.  The mask is
    blurred so the transition is a gradient, not a cut-out.
    """
    L = I.luma(x)
    w = I.blur(1.0 - I.soft(L, 0.05, knee), spread)
    tgt = np.array(tone, np.float32)
    y = x * (1.0 - amount * w)[..., None] + tgt * (amount * w)[..., None]
    return np.clip(y, 0.0, 1.0)


def vign(x, ctx, amount=0.45, inner=0.50, power=1.6):
    """Fixed corner falloff - the frame reads the same on every icon."""
    H, W = x.shape[:2]
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    r = np.hypot((yy - (H - 1) / 2.0) / (H / 2.0), (xx - (W - 1) / 2.0) / (W / 2.0)) / np.sqrt(2.0)
    m = 1.0 - amount * np.clip((r - inner) / (1.0 - inner), 0.0, 1.0) ** power
    return np.clip(x * m[..., None], 0.0, 1.0)


def warm(x, ctx):
    """Torchlight: warm highlights, brown shadows."""
    return grade(x, ctx, contrast=0.50, sat=1.10, vibrance=0.60,
                 shadow=(-0.002, -0.010, 0.026), high=(0.055, 0.024, -0.014))


def cold(x, ctx):
    """Moonlight: desaturated mid, blue shadows, cool highlights."""
    return grade(x, ctx, contrast=0.50, sat=0.95, vibrance=0.55,
                 shadow=(-0.012, -0.002, 0.034), high=(0.008, 0.020, 0.046))


def palette(x, ctx, k=48):
    """Nearest colour of the set-wide k-means palette."""
    pal = ctx.get('pal')
    if pal is None:
        return x
    return I.to_f(apply_palette(I.to_u8(x), pal))


def pixel(x, ctx, target=64, scale_to=256, k=32):
    """Pixel-art: render at `target` px, quantise, then nearest-neighbour x f."""
    pal = ctx.get('pal_small')
    if pal is None:
        return x
    small = I.to_u8(I.resample_ar(x, target, kernel='mitchell'))
    small = apply_palette(small, pal)
    f = max(1, int(round(scale_to / float(target))))
    big = np.repeat(np.repeat(small, f, 0), f, 1)
    if big.shape[0] != scale_to:
        big = np.asarray(Image.fromarray(big).resize((scale_to, scale_to), Image.NEAREST))
    return I.to_f(big)


def loot(x, ctx):
    """What most "premium loot" icon packs look like: grade + quiet + rim + glow."""
    y = grade(x, ctx, contrast=0.55, sat=1.16, vibrance=0.75)
    y = quiet(y, ctx, amount=0.50)
    y = rim(y, ctx, amount=0.70)
    return glow(y, ctx, amount=0.60, lo=0.35, hi=0.90, sigma=5.0, gate_lo=0.012, gate_hi=0.080)


PRESETS = OrderedDict([
    ('grade',   ('Кисть / градиент',     grade)),
    ('cel',     ('Cel-shading',          cel)),
    ('ink',     ('Тушь (обводка)',       ink)),
    ('rim',     ('Контровой свет',       rim)),
    ('glow',    ('Свечение',             glow)),
    ('paint',   ('Живопись (Kuwahara)',  paint)),
    ('quiet',   ('Тихий фон',            quiet)),
    ('vign',    ('Виньетка',             vign)),
    ('warm',    ('Тёплый свет',          warm)),
    ('cold',    ('Ночной холод',         cold)),
    ('palette', ('Палитра 48',           palette)),
    ('pixel',   ('Пиксель-арт 64',       pixel)),
    ('loot',    ('Loot-стиль',           loot)),
])

# combinations worth baking: `--presets quiet+vign`, `--apply grade+rim`, ...
RECIPES = OrderedDict([
    ('quiet+vign', 'flat quiet backdrop + fixed corner falloff - the cheapest way '
                   'to make 4100 sheets read as one set (also ~14% smaller files)'),
    ('grade+rim', 'deliberate light + a lit edge, no bloom'),
    ('grade+quiet+rim', 'loot without the halo (no raised background)'),
    ('cel+ink', 'banded tones plus drawn contours - the most "illustrated" pair'),
])


def build_chain(spec):
    """('grade+rim') -> (label, callable(icon, ctx))."""
    names = [n.strip() for n in spec.split('+') if n.strip()]
    for n in names:
        if n not in PRESETS:
            raise KeyError('%s (known: %s)' % (n, ', '.join(PRESETS)))
    if len(names) == 1:
        title, fn = PRESETS[names[0]]
        return title, fn
    chain = [PRESETS[n][1] for n in names]

    def run(x, ctx):
        for f in chain:
            x = f(x, ctx)
        return x

    return ' + '.join(PRESETS[n][0] for n in names), run


# --------------------------------------------------------------------------
# palettes (k-means on the set, deterministic)
# --------------------------------------------------------------------------

def collect_pixels(imgs_u8, max_px=200000, seed=0):
    rng = np.random.default_rng(seed)
    px = np.concatenate([im.reshape(-1, 3) for im in imgs_u8], 0)
    if len(px) > max_px:
        px = px[rng.choice(len(px), max_px, replace=False)]
    return px.astype(np.float32)


def kmeans(px, k, iters=14, seed=0):
    rng = np.random.default_rng(seed)
    c = px[rng.choice(len(px), 1)].copy()
    for _ in range(k - 1):                                  # k-means++ init
        d = ((px[:, None, :] - c[None, :, :]) ** 2).sum(-1).min(1)
        p = d / max(d.sum(), 1e-9)
        c = np.vstack([c, px[rng.choice(len(px), p=p)]])
    for _ in range(iters):
        lab = ((px[:, None, :] - c[None, :, :]) ** 2).sum(-1).argmin(1)
        for j in range(k):
            m = lab == j
            if m.any():
                c[j] = px[m].mean(0)
    return np.clip(c + 0.5, 0, 255).astype(np.uint8)


def build_palette(imgs_u8, k, seed=0, max_px=200000):
    """k-means palette with pure black pinned as entry 0 (the sheets' background).

    Deterministic for a given sample: seed fixed, pixels subsampled by a fixed
    RNG, so the same set always yields the same palette.
    """
    px = collect_pixels(imgs_u8, max_px=max_px, seed=seed)
    keep = px.max(1) > 8                                    # background is not a colour
    px = px[keep] if keep.sum() > k * 20 else px
    pal = kmeans(px, max(1, k - 1), seed=seed)
    pal = np.vstack([np.zeros((1, 3), np.uint8), pal])
    return pal[np.argsort(I.luma(pal / 255.0))]


def apply_palette(img_u8, pal, chunk=65536):
    h, w, _ = img_u8.shape
    a = img_u8.reshape(-1, 3).astype(np.int32)
    P = pal.astype(np.int32)
    out = np.empty_like(a)
    for s in range(0, len(a), chunk):
        blk = a[s:s + chunk]
        out[s:s + chunk] = P[((blk[:, None, :] - P[None, :, :]) ** 2).sum(-1).argmin(1)]
    return out.reshape(h, w, 3).astype(np.uint8)


# --------------------------------------------------------------------------
# measurement (all numbers relative to the canon set)
# --------------------------------------------------------------------------

def metrics(got, ref):
    """colours, body chroma, edge width, backdrop brightness/chroma and deltas.

    `bg`/`bgchr` are the mean luma and chroma of the pixels that are black in the
    canon icon: that is exactly the inconsistent part - every sheet draws its own
    coloured glow behind its items, so a low `bgchr` means a quieter grid.
    """
    got_f, ref_f = I.to_f(got), I.to_f(ref)
    uniq = len(np.unique(got.reshape(-1, 3), axis=0))
    L = I.luma(got_f)
    chroma = np.abs(got_f - L[..., None]).max(-1)
    body = L > 0.06
    chr_body = float(chroma[body].mean()) if body.any() else 0.0
    back = I.luma(ref_f) < 0.01
    bg = float(L[back].mean() * 255) if back.sum() > 50 else 0.0
    bgchr = float(chroma[back].mean() * 255) if back.sum() > 50 else 0.0
    _o, width, n = M.edge_stats(L)
    d256 = float(np.abs(got.astype(np.float32) - ref.astype(np.float32)).mean())
    ga = np.asarray(Image.fromarray(got).resize((64, 64), Image.LANCZOS), np.float32)
    gb = np.asarray(Image.fromarray(ref).resize((64, 64), Image.LANCZOS), np.float32)
    return uniq, chr_body, width, bg, bgchr, d256, float(np.abs(ga - gb).mean()), n


# --------------------------------------------------------------------------
# lab: sample icons, render the sheet, print the table
# --------------------------------------------------------------------------

def sample_icons(set_dir, n, part=1):
    """Evenly spread icon files over the packs (deterministic)."""
    packs = sorted(d for d in os.listdir(os.path.join(ROOT, set_dir))
                   if d.startswith('RPG Loot Icons'))
    step = max(1, len(packs) // n)
    picks = packs[::step][:n]
    out = []
    for p in picks:
        d = os.path.join(ROOT, set_dir, p, 'part%d' % part)
        if not os.path.isdir(d):
            continue
        fs = sorted(f for f in os.listdir(d) if f.endswith('.webp'))
        f = fs[len(fs) // 3]
        out.append((p, 'part%d' % part, f, os.path.join(d, f)))
    return out


def sheet_fonts():
    try:
        return (ImageFont.truetype(SHEET_FONT, 15),
                ImageFont.truetype(SHEET_FONT_BOLD, 16),
                ImageFont.truetype(SHEET_FONT_BOLD, 19))
    except OSError:
        f = ImageFont.load_default()
        return f, f, f


def render_sheet(rows, header, path, dst, pad=12, cell=200, label_w=232, note=''):
    """rows = [(label, [ndarray HxWx3 u8, ...]), ...] - one column per icon."""
    f_small, f_bold, f_title = sheet_fonts()
    cols = len(header)
    W = label_w + cols * (cell + pad) + pad
    H = 84 + len(rows) * (cell + pad + 8)
    img = Image.new('RGB', (W, H), (26, 26, 30))
    dr = ImageDraw.Draw(img)
    dr.text((pad, 14), 'RPG Loot Icons — стили поверх базы icons-256-base (WebP q92, 256×256)',
            font=f_title, fill=(240, 240, 245))
    dr.text((pad, 42), note or 'апскейл 1:1 для просмотра', font=f_small, fill=(150, 150, 160))
    for c, name in enumerate(header):
        x = label_w + c * (cell + pad)
        dr.text((x + 2, 84 - 22), name, font=f_small, fill=(140, 145, 160))
    for r, (label, icons) in enumerate(rows):
        y = 84 + r * (cell + pad + 8)
        dr.rectangle([pad, y, pad + 4, y + cell - 1], fill=(90, 95, 110))
        txt = label.split(' — ')
        dr.text((pad + 12, y + 6), txt[0], font=f_bold, fill=(235, 235, 240))
        if len(txt) > 1:
            dr.text((pad + 12, y + 28), txt[1], font=f_small, fill=(160, 165, 175))
        for c, im in enumerate(icons):
            x = label_w + c * (cell + pad)
            big = Image.fromarray(im).resize((dst, dst), Image.LANCZOS)
            img.paste(big, (x + (cell - dst) // 2, y + (cell - dst) // 2))
    img.save(path)
    return path


def zoom_sheet(rows, header, path, cell=170, zoom=2.0, crop=118):
    """Two tiles per cell: the icon 1:1 and the centre at `zoom`x (nearest)."""
    f_small, f_bold, f_title = sheet_fonts()
    tw = int(cell * zoom)
    W = 240 + len(header) * (cell + tw + 3 * 8)
    H = 96 + len(rows) * (max(cell, tw) + 10)
    img = Image.new('RGB', (W, H), (26, 26, 30))
    dr = ImageDraw.Draw(img)
    dr.text((12, 14), 'Стили в 1:1 и в 200%% (центр %d px) — база icons-256-base' % crop,
            font=f_title, fill=(240, 240, 245))
    dr.text((12, 44), 'слева каждая иконка целиком в 256 px, справа — фрагмент x%.1f без сглаживания'
            % zoom, font=f_small, fill=(150, 150, 160))
    for c, (name, sub) in enumerate(header):
        x = 240 + c * (cell + tw + 3 * 8)
        dr.text((x, 96 - 22), name, font=f_bold, fill=(200, 205, 215))
        dr.text((x, 96 - 22 + 18), sub, font=f_small, fill=(140, 145, 160))
    for r, (label, icons) in enumerate(rows):
        y = 96 + r * (max(cell, tw) + 10)
        dr.rectangle([12, y, 16, y + max(cell, tw) - 1], fill=(90, 95, 110))
        txt = label.split(' — ')
        dr.text((24, y + 6), txt[0], font=f_bold, fill=(235, 235, 240))
        if len(txt) > 1:
            dr.text((24, y + 28), txt[1], font=f_small, fill=(160, 165, 175))
        for c, im in enumerate(icons):
            a = Image.fromarray(im)
            x = 240 + c * (cell + tw + 3 * 8)
            img.paste(a.resize((cell, cell), Image.LANCZOS), (x, y))
            hl = crop // 2
            centre = (256 - crop) // 2
            z = a.crop((centre, centre, centre + crop, centre + crop)).resize((tw, tw), Image.NEAREST)
            img.paste(z, (x + cell + 8, y))
    img.save(path)
    return path


def render_grid_sheet(named_icons, path, cell=88, pad=6, label_w=190, cols=7, note=''):
    """A loot-grid simulation: `named_icons` = [(label, [icon, ...]), ...] panels."""
    f_small, f_bold, f_title = sheet_fonts()
    rows = int(np.ceil(len(named_icons[0][1]) / float(cols)))
    W = label_w + cols * (cell + pad) + pad
    H = 84 + len(named_icons) * (rows * (cell + pad) + 26)
    img = Image.new('RGB', (W, H), (23, 23, 27))
    dr = ImageDraw.Draw(img)
    dr.text((pad, 14), 'Сетка лута: 4100 иконок в одном гриде — что делает стиль', font=f_title,
            fill=(240, 240, 245))
    dr.text((pad, 42), note or 'иконки разных паков подряд, %d px на ячейку' % cell,
            font=f_small, fill=(150, 150, 160))
    y = 84
    for label, icons in named_icons:
        dr.rectangle([pad, y, pad + 4, y + rows * (cell + pad) - 1], fill=(90, 95, 110))
        txt = label.split(' — ')
        dr.text((pad + 12, y + 6), txt[0], font=f_bold, fill=(235, 235, 240))
        if len(txt) > 1:
            dr.text((pad + 12, y + 28), txt[1], font=f_small, fill=(160, 165, 175))
        y += 26
        for i, im in enumerate(icons):
            x = label_w + (i % cols) * (cell + pad)
            img.paste(Image.fromarray(im).resize((cell, cell), Image.LANCZOS),
                      (x, y + (i // cols) * (cell + pad)))
        y += rows * (cell + pad) + 6
    img.save(path)
    return path


def run_lab(args):
    picks = sample_icons(args.set, args.samples)
    if not picks:
        print('no icons found in', args.set)
        return 1
    header = ['%s / %s' % (p.replace('RPG Loot Icons ', ''), f.split('.')[0]) for p, _pt, f, _ in picks]
    canon = [np.asarray(Image.open(path).convert('RGB')) for *_r, path in picks]

    ctx = {'hue_deg': args.hue_deg}
    need_pal = any(n in ('palette', 'pixel') for spec in args.presets for n in spec.split('+'))
    if need_pal:
        imgs = [np.asarray(Image.open(p).convert('RGB')) for *_r, p in sample_icons(args.set, 64)]
        ctx['pal'] = build_palette(imgs, 48, seed=0)
        ctx['pal_small'] = build_palette(imgs, 32, seed=1)
        print('palette built from %d icons' % len(imgs), flush=True)

    rows = [('База icons-256-base — без стиля', canon)]
    spec_rows = {'canon': ('База icons-256-base — без стиля', canon)}
    table = []
    for spec in args.presets:
        try:
            title, fn = build_chain(spec)
        except KeyError as e:
            print('unknown preset:', e.args[0])
            return 2
        t0 = time.time()
        out = [I.to_u8(fn(I.to_f(c), ctx)) for c in canon]
        m = np.array([metrics(o, c) for o, c in zip(out, canon)], np.float64)
        rows.append((title + ' — ' + spec, out))
        spec_rows[spec] = rows[-1]
        table.append((title, spec, m, time.time() - t0))
        print('  %-18s %5.2fs/icon  colours %6.0f  body chr %.3f  width %.2f  '
              'bg %.2f/%5.1f  d256 %4.1f  e64 %4.1f'
              % (spec, (time.time() - t0) / len(canon), m[:, 0].mean(), m[:, 1].mean(),
                 m[:, 2].mean(), m[:, 3].mean(), m[:, 4].mean(), m[:, 5].mean(), m[:, 6].mean()),
              flush=True)

    os.makedirs(os.path.join(ROOT, args.out), exist_ok=True)
    note = 'база %s, один разрез, один ряд — апскейл 1:1 до %d px для просмотра' % (args.set, args.cell)
    for dst, name in ((args.cell, 'style-sheet.png'), (args.cell // 2, 'style-sheet-small.png')):
        p = render_sheet(rows, header, os.path.join(ROOT, args.out, name), dst, cell=args.cell, note=note)
        print('wrote', os.path.relpath(p, ROOT), '(%d KB)' % (os.path.getsize(p) // 1024))
    if args.zoom:
        wanted = [s.strip() for s in args.zoom_presets.split(',') if s.strip()]
        zrows = [spec_rows[s] for s in wanted if s in spec_rows]
        if zrows:
            p = zoom_sheet(zrows, [(h.split(' / ')[0], h.split(' / ')[1]) for h in header],
                           os.path.join(ROOT, args.out, 'style-zoom.png'))
            print('wrote', os.path.relpath(p, ROOT), '(%d KB)' % (os.path.getsize(p) // 1024))

    if args.grid:
        gpicks = sample_icons(args.set, args.grid_icons)
        gcanon = [np.asarray(Image.open(p).convert('RGB')) for *_r, p in gpicks]
        panels = [('Канон icons-256 — база', gcanon)]
        for spec in [s.strip() for s in args.grid_presets.split(',') if s.strip()]:
            try:
                gtitle, gfn = build_chain(spec)
            except KeyError as e:
                print('unknown preset:', e.args[0])
                return 2
            panels.append(('%s — %s' % (gtitle, spec),
                           [I.to_u8(gfn(I.to_f(c), ctx)) for c in gcanon]))
        p = render_grid_sheet(panels, os.path.join(ROOT, args.out, 'grid-compare.png'),
                              note='%d иконок разных паков подряд, %d px на ячейку — видно '
                                   'уровень фона за предметами' % (len(gcanon), 88))
        print('wrote', os.path.relpath(p, ROOT), '(%d KB)' % (os.path.getsize(p) // 1024))

    base = np.array([metrics(c, c) for c in canon], np.float64)
    print()
    print('| стиль | цветов | хрома тела | контур, px | фон (ярк/хрома) | Δ от канона | e64 | мс/иконку |')
    print('|---|---|---|---|---|---|---|---|')
    print('| база icons-256-base | %.0f | %.3f | %.2f | %.2f / %.1f | 0 | 0 | 0 |'
          % (base[:, 0].mean(), base[:, 1].mean(), base[:, 2].mean(),
             base[:, 3].mean(), base[:, 4].mean()))
    for title, name, m, dt in table:
        print('| %s (`%s`) | %.0f | %.3f | %.2f | %.2f / %.1f | %.1f | %.1f | %.0f |'
              % (title, name, m[:, 0].mean(), m[:, 1].mean(), m[:, 2].mean(),
                 m[:, 3].mean(), m[:, 4].mean(), m[:, 5].mean(), m[:, 6].mean(),
                 dt / len(canon) * 1000))
    print('\n(выборка: %d иконок; цвета — среднее уникальных цветов, фон — средняя яркость и '
          'хрома чёрных пикселей базы, Δ/e64 — средняя |разница| с базой в уровнях 255)'
          % len(canon))
    return 0


# --------------------------------------------------------------------------
# apply a chosen preset to the whole set
# --------------------------------------------------------------------------

_WORK = {}


def _init_work(spec, ctx, src, out_root, quality):
    _WORK['fn'] = build_chain(spec)[1]
    _WORK['ctx'] = ctx
    _WORK['src'] = src
    _WORK['out'] = out_root
    _WORK['quality'] = quality


def _work_one(rel):
    """Encode one icon with the preset of this worker; returns (rel, md5, bytes)."""
    import hashlib
    im = np.asarray(Image.open(os.path.join(_WORK['src'], rel)).convert('RGB'))
    y = I.to_u8(_WORK['fn'](I.to_f(im), _WORK['ctx']))
    dst = os.path.join(_WORK['out'], rel)
    Image.fromarray(y).save(dst, quality=_WORK['quality'], method=6)
    return rel, hashlib.md5(open(dst, 'rb').read()).hexdigest(), os.path.getsize(dst)


def set_label(root):
    """Human label for a set directory, from its manifest (role + baked style)."""
    name = os.path.basename(root.rstrip('/'))
    man_path = os.path.join(ROOT, root, 'manifest.json')
    if not os.path.exists(man_path):
        return name
    try:
        man = json.load(open(man_path))
    except ValueError:
        return name
    role = {'actual': 'актуальный', 'base': 'база', 'previous': 'предыдущий'}.get(
        man.get('role'), man.get('role') or '')
    style = man.get('style') or {}
    presets = '+'.join(style.get('presets') or [])
    if style:
        return '%s %s (%s)' % (role, name, presets)
    return '%s %s (%s)' % (role, name, 'без стиля')


def run_compare(args):
    """Render a before/after grid for two sets - files read from disk, not recomputed."""
    pairs = [s.strip() for s in args.compare.split(',')]
    if len(pairs) != 2:
        print('--compare takes exactly two set dirs, e.g. --compare icons-256-base,icons-256')
        return 2
    a_dir, b_dir = pairs
    man = json.load(open(os.path.join(ROOT, b_dir, 'manifest.json')))
    files = [e['file'] for e in man['icons']]
    step = max(1, len(files) // args.grid_icons)
    picks = files[::step][:args.grid_icons]
    panels = []
    for d in (a_dir, b_dir):
        imgs = [np.asarray(Image.open(os.path.join(ROOT, d, f)).convert('RGB')) for f in picks]
        panels.append(('%s — %d иконок' % (set_label(d), len(picks)), imgs))
    da = [np.abs(x.astype(np.int16) - y.astype(np.int16)).mean()
          for (x, y) in zip(panels[0][1], panels[1][1])]
    out = args.compare_out or os.path.join(args.out, 'compare-%s-%s.png'
                                           % (os.path.basename(a_dir), os.path.basename(b_dir)))
    os.makedirs(os.path.dirname(os.path.join(ROOT, out)), exist_ok=True)
    note = ('оба ряда прочитаны с диска: %s против %s; средняя |разница| %.1f уровня '
            '(мин %.1f, макс %.1f)' % (a_dir, b_dir, float(np.mean(da)), float(min(da)), float(max(da))))
    p = render_grid_sheet(panels, os.path.join(ROOT, out), note=note)
    print('mean |%s - %s| over %d icons: %.1f levels' % (a_dir, b_dir, len(picks), float(np.mean(da))))
    print('wrote', os.path.relpath(p, ROOT), '(%d KB)' % (os.path.getsize(p) // 1024))
    return 0


def run_apply(args):
    spec = args.apply
    try:
        title, fn = build_chain(spec)
    except KeyError as e:
        print('unknown preset:', e.args[0])
        return 2
    src = os.path.join(ROOT, args.set)
    man = json.load(open(os.path.join(src, 'manifest.json')))
    icons = man['icons']
    out_root = os.path.join(ROOT, args.out_dir)

    ctx = {'hue_deg': args.hue_deg}
    if any(n in ('palette', 'pixel') for n in spec.split('+')):
        imgs = [np.asarray(Image.open(p).convert('RGB')) for *_r, p in sample_icons(args.set, 64)]
        ctx['pal'] = build_palette(imgs, 48, seed=0)
        ctx['pal_small'] = build_palette(imgs, 32, seed=1)
        print('palette built from %d icons' % len(imgs))

    total = min(len(icons), args.limit or len(icons))
    quality = man.get('quality', 92)
    rels = [e['file'] for e in icons[:total]]
    for r in rels:
        os.makedirs(os.path.dirname(os.path.join(out_root, r)), exist_ok=True)

    t0 = time.time()
    jobs = args.jobs or (os.cpu_count() or 1)
    got = {}
    if jobs <= 1 or total < 32:
        _init_work(spec, ctx, src, out_root, quality)
        for i, r in enumerate(rels):
            rel, md5, n = _work_one(r)
            got[rel] = (md5, n)
            if (i + 1) % 250 == 0 or i + 1 == total:
                print('  %d/%d  %.1fs' % (i + 1, total, time.time() - t0), flush=True)
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=jobs, initializer=_init_work,
                                 initargs=(spec, ctx, src, out_root, quality)) as ex:
            futs = [ex.submit(_work_one, r) for r in rels]
            for i, f in enumerate(as_completed(futs)):
                rel, md5, n = f.result()
                got[rel] = (md5, n)
                if (i + 1) % 500 == 0 or i + 1 == total:
                    print('  %d/%d  %.1fs' % (i + 1, total, time.time() - t0), flush=True)

    out_icons = []
    for e in icons[:total]:
        md5, n = got[e['file']]
        rec = dict(e)
        rec['bytes'] = n
        rec['md5'] = md5                      # the styled file has its own hash
        out_icons.append(rec)
    print('  encoded %d icons in %.0fs' % (total, time.time() - t0))

    m2 = dict(man)
    m2['icons'] = out_icons
    m2['count'] = len(out_icons)
    m2['total_bytes'] = sum(e['bytes'] for e in out_icons)
    # the styled set was produced now; the base's timestamp stays in base_pipeline
    base_generated = m2.get('generated')
    m2['generated'] = m2['style']['generated']
    if base_generated and 'base_pipeline' in m2:
        m2['base_pipeline']['generated'] = base_generated
    m2['style'] = {'presets': [n.strip() for n in spec.split('+') if n.strip()],
                   'title': title, 'source_set': args.set, 'hue_deg': args.hue_deg,
                   'generated': time.strftime('%Y-%m-%dT%H:%M:%S%z')}
    json.dump(m2, open(os.path.join(out_root, 'manifest.json'), 'w'), indent=1)
    print('wrote %d icons to %s (%.1f MB, %.0fs)'
          % (len(out_icons), os.path.relpath(out_root, ROOT), m2['total_bytes'] / 1e6,
             time.time() - t0))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--set', default='icons-256-base',
                    help='source set (default icons-256-base, the clean set; '
                         'icons-256 already has quiet+vign baked in)')
    ap.add_argument('--presets', default=','.join(PRESETS),
                    help='comma-separated subset of: %s (chains allowed: grade+rim)'
                         % ', '.join(PRESETS))
    ap.add_argument('--samples', type=int, default=8, help='icons per sheet (default 8)')
    ap.add_argument('--cell', type=int, default=200, help='preview size, px (default 200)')
    ap.add_argument('--out', default='docs/style-lab', help='where the sheets go')
    ap.add_argument('--zoom', action='store_true', default=True, help='also write the 1:1/200%% sheet')
    ap.add_argument('--no-zoom', dest='zoom', action='store_false')
    ap.add_argument('--zoom-presets', default='canon,grade,cel,ink,rim,paint,palette,pixel',
                    dest='zoom_presets', help='which rows go into the 1:1/200%% sheet')
    ap.add_argument('--grid', action='store_true', default=True,
                    help='also write the loot-grid comparison')
    ap.add_argument('--no-grid', dest='grid', action='store_false')
    ap.add_argument('--grid-presets', default='quiet+vign,grade+quiet+rim', dest='grid_presets',
                    help='panels of the loot-grid sheet')
    ap.add_argument('--grid-icons', type=int, default=28, dest='grid_icons',
                    help='how many icons in the loot-grid sheet')
    ap.add_argument('--hue-deg', type=float, default=0.0, dest='hue_deg',
                    help='global hue rotation in degrees (grade/warm/cold/loot)')
    ap.add_argument('--compare', default=None,
                    help='render a before/after grid for two set dirs, '
                         'e.g. icons-256-base,icons-256')
    ap.add_argument('--compare-out', default=None, dest='compare_out',
                    help='where the --compare sheet goes (default <out>/compare-A-B.png)')
    ap.add_argument('--apply', default=None, help='bake this preset over the whole set')
    ap.add_argument('--out-dir', default=None,
                    help='target dir for --apply (default <set>-<preset>; the actual set '
                         'is icons-256, e.g. --apply quiet+vign --out-dir icons-256)')
    ap.add_argument('--limit', type=int, default=0, help='stop after N icons (--apply)')
    ap.add_argument('--jobs', type=int, default=0, help='0 = cpu count (--apply)')
    args = ap.parse_args(argv)
    args.presets = [p.strip() for p in args.presets.split(',') if p.strip()]
    if args.compare:
        return run_compare(args)
    if args.apply:
        if not args.out_dir:
            args.out_dir = '%s-%s' % (args.set, args.apply)
        return run_apply(args)
    return run_lab(args)


if __name__ == '__main__':
    sys.exit(main())
