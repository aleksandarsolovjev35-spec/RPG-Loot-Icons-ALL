"""Measure what the enhance pass actually does to icons - the numbers in the README.

The tool works on the sheets themselves (via the same detector as the cutters),
so it can compare any stage of the pipeline against the plain resize that the
first version of `repack_icons.py` used:

    python3 tools/measure_quality.py                  # 7 sample icons, markdown table
    python3 tools/measure_quality.py --sheets 14      # more samples (slower)
    python3 tools/measure_quality.py --set icons-512  # add a row for the encoded set

Metrics, all on the produced 512 px image (see the README):

  noise    sigma of the high-frequency residual inside flat areas.  The mask is
           built on the *native* crop (3x3 luma range < 2%), so any energy
           measured there is an artefact, not art.
  bleed    luma a dark pixel got ABOVE the brightest source pixel in its
           native 3x3 neighbourhood, i.e. light that leaked out of the artwork
           (a halo).  Positive only - correcting a dark ringing trough back up
           to the true value does not count, a white fringe in the black does.
  bleed99  the same, 99th percentile: catches a halo that sits on a small part
           of the silhouette (mean over the whole outline hides it).
  over     overshoot of an edge beyond the levels it connects, normalised by its
           step height and averaged over all detected edges.
  width    effective transition length of those edges, in px.
  hf       high-frequency energy (luma minus 2 px box) inside textured areas,
           relative to the plain-resize reference: 1.0 = detail untouched.
  e64      mean |error| against a direct render of the source at 64 px, i.e.
           what survives at the size an icon is actually shown at in a VTT.
"""
import argparse
import glob
import os
import sys

import numpy as np
from PIL import Image

try:
    import cut_icons as C
    import imgproc as I
except ImportError:
    from tools import cut_icons as C
    from tools import imgproc as I

ROOT = C.ROOT
S = 512

# the stages of the pipeline, in order, as (label, config override)
STAGES = [
    ('plain resize (old)', dict(denoise=0, black=0, steer=0, sharpen=0, clamp=False)),
    ('anti-ringing resize', dict(denoise=0, black=0, steer=0, sharpen=0)),
    ('+ denoise', dict(denoise=0.014, black=0, steer=0, sharpen=0)),
    ('+ black floor', dict(denoise=0.014, black=0.010, steer=0, sharpen=0)),
    ('+ steered blur', dict(denoise=0.014, black=0.010, steer=3.0, steer_range=0.030, sharpen=0)),
    ('+ masked unsharp (full)', dict(denoise=0.014, black=0.010, steer=3.0, steer_range=0.030,
                                     sharpen=0.30, sharpen_clamp=0.05, sharpen_thr=0.010)),
]


def sample_icons(n_samples):
    """Evenly spread over the packs, 1 icon per sheet."""
    sheets = sorted(glob.glob(os.path.join(ROOT, 'RPG Loot Icons */*.jpg')))
    step = max(1, len(sheets) // n_samples)
    picks = sheets[::step][:n_samples]
    out = []
    for path in picks:
        crops = [(k, xy, c) for k, xy, c in C.iter_cell_crops(path, None)]
        if not crops:
            continue
        k, xy, crop = crops[len(crops) // 3]
        out.append((path, k, crop))
    return out


def masks(crop):
    """flat / textured / dark-neighbourhood masks plus the local source maximum.

    `cap` is the brightest source luma in the native 3x3 around each pixel,
    blown up with NEAREST (so it is the value of the pixel's own native cell,
    a deliberately conservative bound); anything above it in the produced image
    is light that did not exist in the artwork.
    """
    Ln = I.luma(I.to_f(crop))
    pad = np.pad(Ln, 2, mode='reflect')
    rng = np.zeros_like(Ln)
    for dy in range(3):
        for dx in range(3):
            rng = np.maximum(rng, np.abs(pad[dy:dy + Ln.shape[0], dx:dx + Ln.shape[1]] - Ln))
    mx = np.maximum.reduce([np.roll(np.roll(Ln, dy, 0), dx, 1)
                            for dy in (-1, 0, 1) for dx in (-1, 0, 1)])

    def up(m, mode=Image.BILINEAR, thr=0.999):
        a = np.asarray(Image.fromarray((m * 255).astype(np.uint8)).resize((S, S), mode), np.float32)
        return a > 255 * thr

    def upnear(a):
        return np.asarray(Image.fromarray(a, 'F').resize((S, S), Image.NEAREST), np.float32)

    flat = rng < 0.02                       # no structure at native scale
    tex = rng > 0.02                        # real artwork detail
    return (up(flat.astype(np.float32)), up(tex.astype(np.float32)), upnear(mx.astype(np.float32)),
            upnear(Ln.astype(np.float32)))


def edge_stats(L):
    """(overshoot, width) averaged over ISOLATED edges, rows and columns.

    Only windows holding a single dominant step are used: on a window with two
    edges (a thin line) the "levels the edge connects" cannot be estimated and
    the overshoot would come out meaningless.
    """
    widths, overs = [], []
    for axis in (0, 1):
        A = L if axis == 0 else L.T
        d = np.diff(A, axis=1)
        H, W = A.shape
        for y in range(0, H, 3):
            for x in np.where(np.abs(d[y]) > 0.08)[0]:
                if x < 10 or x > W - 11:
                    continue
                dd = np.abs(d[y, x - 9:x + 11])
                strong = np.where(dd > 0.5 * dd.max())[0]
                if len(strong) == 0 or strong.max() - strong.min() > 2:
                    continue                      # more than one edge in the window
                p = A[y, x - 9:x + 11]
                lo, hi = p[:8].mean(), p[12:].mean()
                step = hi - lo
                if abs(step) < 0.05:
                    continue
                t = np.clip((p - lo) / step, 0, 1)
                widths.append(np.clip(1 - np.abs(2 * t - 1), 0, None).sum() / 2)
                overs.append(max(0.0, (p.max() - max(lo, hi)) / abs(step)) +
                             max(0.0, (min(lo, hi) - p.min()) / abs(step)))
    if not widths:
        return np.nan, np.nan, 0
    return float(np.mean(overs)), float(np.mean(widths)), len(widths)


def measure(out, crop, ref):
    m, hf_ref, ref64 = ref
    flat, tex, cap, Ln = m
    L = I.luma(I.to_f(out))
    hp = L - I.box(L, 2)
    noise = 1.4826 * np.median(np.abs(hp[flat] - np.median(hp[flat]))) * 255 if flat.sum() > 200 else np.nan
    bleed_map = np.clip(L - cap, 0, None)[Ln < 0.10] * 255      # light in dark places
    bleed = bleed_map.mean() if bleed_map.size > 200 else np.nan
    bleed99 = np.percentile(bleed_map, 99) if bleed_map.size > 200 else np.nan
    over, width, _n = edge_stats(L)
    hf = (np.abs(hp[tex]).mean() / hf_ref) if (tex.sum() > 200 and hf_ref > 0) else np.nan
    got64 = np.asarray(Image.fromarray(out).resize((64, 64), Image.LANCZOS), np.float32)
    e64 = np.abs(ref64 - got64).mean()
    return noise, bleed, bleed99, over, width, hf, e64


def set_file(set_dir, path, k):
    pack = os.path.basename(os.path.dirname(path))
    part = 'part' + os.path.basename(path).split('Part ')[-1].replace('.jpg', '')
    for ext in ('webp', 'avif', 'png'):
        f = os.path.join(ROOT, set_dir, pack, part, 'icon_%03d.%s' % (k, ext))
        if os.path.exists(f):
            return f
    return None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--sheets', type=int, default=7, help='how many sample icons (default 7)')
    ap.add_argument('--set', default=None, help='also measure this set dir, e.g. icons-512')
    ap.add_argument('--size', type=int, default=512,
                    help='output side to measure (default 512; use 256 for the 256 sets)')
    ap.add_argument('--jobs', type=int, default=0, help='0 = cpu count')
    args = ap.parse_args(argv)
    global S
    if args.size < 16:
        ap.error('--size must be at least 16')
    S = args.size

    samples = sample_icons(args.sheets)
    rows = {label: [] for label, _ in STAGES}
    set_label = None
    if args.set:
        set_label = ('%s (encoded)' % os.path.basename(args.set.rstrip('/')) if args.set.startswith(ROOT)
                     else '%s (encoded)' % args.set.rstrip('/'))
        rows[set_label] = []
    for path, k, crop in samples:
        m = masks(crop)
        ref_arr = np.asarray(Image.fromarray(crop).resize((S, S), Image.LANCZOS), np.float32)
        Lr = I.luma(I.to_f(ref_arr))
        ref64 = np.asarray(Image.fromarray(crop).resize((64, 64), Image.LANCZOS), np.float32)
        ref = (m, np.abs((Lr - I.box(Lr, 2))[m[1]]).mean(), ref64)
        for label, cfg in STAGES:
            stage_cfg = dict(cfg, size=S)
            rows[label].append(measure(I.enhance(crop, stage_cfg), crop, ref))
        if args.set:
            f = set_file(args.set, path, k)
            if f:
                rows[set_label].append(
                    measure(np.asarray(Image.open(f).convert('RGB')), crop, ref))
        print('  %-40s icon %3d' % (os.path.relpath(path, ROOT), k), flush=True)

    print()
    print('| stage | noise | bleed | bleed99 | overshoot | width | hf | err@64 |')
    print('|---|---|---|---|---|---|---|---|')
    for label in rows:
        a = np.array(rows[label], np.float64)
        if not len(a):
            continue
        mu = np.nanmean(a, 0)
        print('| %s | %.2f | %+.2f | %+.2f | %.3f | %.2f | %.2f | %.2f |'
              % (label, mu[0], mu[1], mu[2], mu[3], mu[4], mu[5], mu[6]))
    print('\n(sample: %d sheets; noise/bleed/width in levels of 255 and px, '
          'hf relative to a plain resize of the same crop)' % len(samples))
    return 0


if __name__ == '__main__':
    sys.exit(main())
