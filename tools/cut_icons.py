"""Universal icon cutter for RPG loot sprite sheets.

Output contract (agreed with the user):
* keep the black background - crops are opaque rectangles, not tight
  transparent cutouts;
* a crop is exactly the area INSIDE the gray/white frame ("кайма"),
  the frame itself is not included (cut strictly along its inner edge,
  frame pixels that bleed inside are painted black);
* cells / sheets without a frame are skipped entirely.

Algorithm
---------
1.  Content mask = luminance > 30 or saturated color (background is
    near-black).
2.  Frame detection: long thin gray runs (the frame lines) are clustered
    into horizontal/vertical grid lines; a cell rectangle is validated by
    side coverage (icons may occlude sides): 3+ sides >=30% visible, or
    2 sides >=80%, or one full side.
3.  Frame pixel map: along each validated rectangle's perimeter, gray runs
    (the line cross-sections, 1-4 px) are marked for removal when the
    perpendicular profile says "frame": outside is background and either a
    gap separates the line from the icon, or the gray slab is >=4px thick
    (flush line + anti-alias).  Icon highlights backed by the body are kept
    and simply fall outside the crop because we cut inside the line.
4.  Crop = rectangle shrunk to the inner edge of the line; any leftover
    frame pixels inside the crop are painted black.
5.  Saved as compact 255-color palette PNG.
"""
import glob
import os
import sys
import numpy as np
from PIL import Image

try:
    from analyze import grayish, h_segments, v_segments
    from scan_frames import cluster, coverage
except ImportError:
    from tools.analyze import grayish, h_segments, v_segments
    from tools.scan_frames import cluster, coverage

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def find_rectangles(hs, vs):
    hlines = cluster([m['y0'] for m in hs if m['x1'] - m['x0'] >= 80])
    hlines = [y for y in hlines
              if sum(1 for m in hs if abs(m['y0'] - y) <= 2 and m['x1'] - m['x0'] >= 60) >= 3]
    vlines = cluster([m['x0'] for m in vs if m['y1'] - m['y0'] >= 80])
    vlines = [x for x in vlines
              if sum(1 for m in vs if abs(m['x0'] - x) <= 2 and m['y1'] - m['y0'] >= 60) >= 3]
    hseg_at = lambda y: [m for m in hs if abs(m['y0'] - y) <= 2]
    vseg_at = lambda x: [m for m in vs if abs(m['x0'] - x) <= 2]

    def complete(lines, mode):
        """restore a grid line hidden by icons: a gap of mode+~gap means one
        missing line at position prev+mode."""
        if mode is None:
            return lines
        out = set(lines)
        for a, b in zip(lines, lines[1:]):
            if mode + 15 <= b - a <= mode + 70:
                out.add(a + mode)
        return sorted(out)

    dh = [b - a for a, b in zip(hlines, hlines[1:]) if 100 <= b - a <= 200]
    dw = [b - a for a, b in zip(vlines, vlines[1:]) if 100 <= b - a <= 200]
    mode_h = int(sorted(dh)[len(dh) // 2]) if len(dh) >= 3 else None
    mode_w = int(sorted(dw)[len(dw) // 2]) if len(dw) >= 3 else None
    hlines = complete(hlines, mode_h)
    vlines = complete(vlines, mode_w)

    def ok_size(h, w):
        if mode_h is not None and not abs(h - mode_h) <= 12:
            return False
        if mode_w is not None and not abs(w - mode_w) <= 12:
            return False
        return 90 <= h <= 220 and 90 <= w <= 220

    rects = []
    for i in range(len(hlines) - 1):
        for j in range(len(vlines) - 1):
            yT, yB = hlines[i], hlines[i + 1]
            xL, xR = vlines[j], vlines[j + 1]
            if not ok_size(yB - yT, xR - xL):
                continue
            covT = coverage(hseg_at(yT), xL, xR, lambda m: (m['x0'], m['x1']))
            covB = coverage(hseg_at(yB), xL, xR, lambda m: (m['x0'], m['x1']))
            covL = coverage(vseg_at(xL), yT, yB, lambda m: (m['y0'], m['y1']))
            covR = coverage(vseg_at(xR), yT, yB, lambda m: (m['y0'], m['y1']))
            covs = (covT, covB, covL, covR)
            # 3+ partially visible sides, or 2 fully visible sides, or one full side
            if sum(c >= 0.3 for c in covs) >= 3 or sum(c >= 0.8 for c in covs) >= 2 or max(covs) >= 0.9:
                rects.append((xL, yT, xR, yB))
    return rects, hlines, vlines


def dilate(m, it=1):
    for _ in range(it):
        m = m | np.roll(m, 1, 0) | np.roll(m, -1, 0) | np.roll(m, 1, 1) | np.roll(m, -1, 1)
    return m


def erode(m, it=1):
    for _ in range(it):
        m = m & np.roll(m, 1, 0) & np.roll(m, -1, 0) & np.roll(m, 1, 1) & np.roll(m, -1, 1)
    return m


def _runs_of(mask1):
    idx = np.nonzero(mask1)[0]
    out = []
    if len(idx) == 0:
        return out
    s = p = idx[0]
    for i in idx[1:]:
        if i - p > 1:
            out.append((s, p))
            s = i
        p = i
    out.append((s, p))
    return out


def classify_band(content, thin, gray, rect, width=3):
    """Frame-line pixels (to be painted black inside the crop).

    The line cross-section (1-4 px) is located per row/column inside the
    perimeter band; it is a frame when outside the cross-section is
    background and either a gap separates it from the icon or the gray slab
    is thick (>=4 px: flush line + anti-alias).  Icon highlights/outlines
    are 1-3 px of gray backed immediately by the body -> not frame.
    """
    H, W = content.shape
    rem = np.zeros((H, W), bool)
    xL, yT, xR, yB = rect
    cand = thin & gray
    for xv, sgn in ((xL, 1), (xR, -1)):
        xa = max(0, xv - width)
        xb = min(W, xv + width + 1)
        for y in range(max(0, yT - width), min(H, yB + width + 1)):
            for a, b in _runs_of(cand[y, xa:xb]):
                a += xa
                b += xa
                seg_in = slice(b + 2, b + 4) if sgn == 1 else slice(max(0, a - 3), a - 1)
                seg_out = slice(max(0, a - 3), a - 1) if sgn == 1 else slice(b + 2, b + 4)
                out_bg = 1.0 - content[y, seg_out].mean()
                in_adj = content[y, seg_in].mean()
                if out_bg >= 0.8 and (in_adj < 0.5 or (b - a + 1) >= 4):
                    rem[y, a:b + 1] = True
    for yh, sgn in ((yT, 1), (yB, -1)):
        ya = max(0, yh - width)
        yb = min(H, yh + width + 1)
        for x in range(max(0, xL - width), min(W, xR + width + 1)):
            for a, b in _runs_of(cand[ya:yb, x]):
                a += ya
                b += ya
                seg_in = slice(b + 2, b + 4) if sgn == 1 else slice(max(0, a - 3), a - 1)
                seg_out = slice(max(0, a - 3), a - 1) if sgn == 1 else slice(b + 2, b + 4)
                out_bg = 1.0 - content[seg_out, x].mean()
                in_adj = content[seg_in, x].mean()
                if out_bg >= 0.8 and (in_adj < 0.5 or (b - a + 1) >= 4):
                    rem[a:b + 1, x] = True
    return rem


def save_icon(rgb, path):
    """Compact 255-color palette PNG."""
    q = Image.fromarray(rgb, 'RGB').quantize(colors=255, dither=Image.Dither.NONE)
    q.save(path, compress_level=9)


def process_sheet(path, out_dir, montage_path=None):
    im = np.array(Image.open(path).convert('RGB')).astype(np.uint8)
    H, W, _ = im.shape
    r = im[:, :, 0].astype(int)
    g = im[:, :, 1].astype(int)
    b = im[:, :, 2].astype(int)
    lum = (r + g + b) / 3.0
    sat = np.maximum(np.maximum(abs(r - g), abs(g - b)), abs(r - b))
    content = (lum > 30) | (sat > 25)

    gray, _ = grayish(im)
    gray2 = (sat <= 14) & (lum >= 30) & (lum <= 240)
    rects, hlines, vlines = find_rectangles(h_segments(gray, 30), v_segments(gray, 30))
    if not rects:
        return [], []                      # sheet without frames: skip

    thick = dilate(erode(content, 1), 1)
    thin = content & ~thick
    remove = np.zeros((H, W), bool)
    for rect in rects:
        remove |= classify_band(content, thin, gray2, rect)

    # reading order: rows by yT, then xL
    rects = sorted(rects, key=lambda t: (t[1], t[0]))
    rows = []
    for rect in rects:
        if rows and rect[1] - rows[-1][0] <= 40:
            rows[-1][1].append(rect)
        else:
            rows.append((rect[1], [rect]))
    ordered = [rect for _, rr in rows for rect in sorted(rr, key=lambda t: t[0])]

    os.makedirs(out_dir, exist_ok=True)
    saved = []
    for k, (xL, yT, xR, yB) in enumerate(ordered, 1):
        x0, x1 = xL + 2, xR - 2            # inside the frame line
        y0, y1 = yT + 2, yB - 2
        crop = im[y0:y1 + 1, x0:x1 + 1].copy()
        crop[remove[y0:y1 + 1, x0:x1 + 1]] = (0, 0, 0)
        name = 'icon_%03d.png' % k
        save_icon(crop, os.path.join(out_dir, name))
        saved.append((name, (x0, y0)))

    if montage_path:
        mon = np.zeros((H, W, 3), np.uint8)
        for name, (x0, y0) in saved:
            a = np.array(Image.open(os.path.join(out_dir, name)).convert('RGB'))
            h, w = a.shape[:2]
            mon[y0:y0 + h, x0:x0 + w] = a
        Image.fromarray(mon).save(montage_path)
    return saved, rects


if __name__ == '__main__':
    paths = sys.argv[1:] or sorted(glob.glob(os.path.join(ROOT, 'RPG Loot Icons */*.jpg')))
    total = 0
    for p in paths:
        pack = os.path.basename(os.path.dirname(p))
        part = 'part' + os.path.basename(p).split('Part ')[-1].replace('.jpg', '')
        out_dir = os.path.join(ROOT, 'cut_icons', pack, part)
        saved, rects = process_sheet(p, out_dir)
        total += len(saved)
        print('%-55s frames=%2d icons=%2d' % (os.path.relpath(p, ROOT), len(rects), len(saved)))
    print('TOTAL icons:', total)
