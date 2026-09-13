"""Scan all sheets, detect frame rectangles via grid-line clustering."""
import glob
import os
import numpy as np
from PIL import Image
try:
    from analyze import grayish, h_segments, v_segments
except ImportError:
    from tools.analyze import grayish, h_segments, v_segments


def cluster(vals, tol=3):
    vals = sorted(vals)
    out = []
    for v in vals:
        if out and v - out[-1][-1] <= tol:
            out[-1].append(v)
        else:
            out.append([v])
    return [int(round(sum(c) / len(c))) for c in out]


def coverage(segs, lo, hi, get):
    """fraction of [lo,hi] covered by segments' spans."""
    iv = []
    for s in segs:
        a, b = get(s)
        a, b = max(a, lo), min(b, hi)
        if b > a:
            iv.append((a, b))
    if not iv:
        return 0.0
    iv.sort()
    tot = 0
    ca, cb = iv[0]
    for a, b in iv[1:]:
        if a <= cb + 2:
            cb = max(cb, b)
        else:
            tot += cb - ca
            ca, cb = a, b
    tot += cb - ca
    return tot / max(1, hi - lo)


def find_rectangles(hs, vs):
    hlines = cluster([m['y0'] for m in hs if m['x1'] - m['x0'] >= 80])
    hlines = [y for y in hlines if sum(1 for m in hs if abs(m['y0'] - y) <= 2 and m['x1'] - m['x0'] >= 60) >= 4]
    vlines = cluster([m['x0'] for m in vs if m['y1'] - m['y0'] >= 80])
    vlines = [x for x in vlines if sum(1 for m in vs if abs(m['x0'] - x) <= 2 and m['y1'] - m['y0'] >= 60) >= 4]
    rects = []
    for i in range(len(hlines) - 1):
        for j in range(len(vlines) - 1):
            yT, yB = hlines[i], hlines[i + 1]
            xL, xR = vlines[j], vlines[j + 1]
            h = yB - yT
            w = xR - xL
            if not (90 <= h <= 220 and 90 <= w <= 220):
                continue
            covT = coverage([m for m in hs if abs(m['y0'] - yT) <= 2], xL, xR, lambda m: (m['x0'], m['x1']))
            covB = coverage([m for m in hs if abs(m['y0'] - yB) <= 2], xL, xR, lambda m: (m['x0'], m['x1']))
            covL = coverage([m for m in vs if abs(m['x0'] - xL) <= 2], yT, yB, lambda m: (m['y0'], m['y1']))
            covR = coverage([m for m in vs if abs(m['x0'] - xR) <= 2], yT, yB, lambda m: (m['y0'], m['y1']))
            covs = [covT, covB, covL, covR]
            if sum(c >= 0.4 for c in covs) >= 3 and covT + covB >= 0.8:
                rects.append((xL, yT, xR, yB))
    return rects


if __name__ == '__main__':
    paths = sorted(glob.glob('RPG Loot Icons */*.jpg'))
    tot = 0
    for p in paths:
        im = np.array(Image.open(p).convert('RGB'))
        gray, _ = grayish(im)
        rects = find_rectangles(h_segments(gray, 60), v_segments(gray, 60))
        tot += len(rects)
        print('%-40s rects=%3d' % (p, len(rects)))
    print('TOTAL rects:', tot)
