"""Analysis: detect gray/white frame line segments in sheets."""
import sys
import numpy as np
from PIL import Image

def grayish(im):
    r = im[:, :, 0].astype(int); g = im[:, :, 1].astype(int); b = im[:, :, 2].astype(int)
    lum = (r + g + b) / 3.0
    sat = np.maximum(np.maximum(abs(r - g), abs(g - b)), abs(r - b))
    return (sat <= 14) & (lum >= 32) & (lum <= 240), lum

def runs(mask_1d):
    out = []
    start = None
    for i, v in enumerate(mask_1d):
        if v and start is None:
            start = i
        elif not v and start is not None:
            out.append((start, i - 1))
            start = None
    if start is not None:
        out.append((start, len(mask_1d) - 1))
    return out

def h_segments(gray, minlen):
    segs = []
    H, W = gray.shape
    for y in range(H):
        for (x0, x1) in runs(gray[y]):
            if x1 - x0 + 1 >= minlen:
                segs.append((y, x0, x1))
    # merge consecutive rows with similar extents (line thickness)
    merged = []
    for (y, x0, x1) in segs:
        placed = False
        for m in merged:
            if abs(m['y1'] - y) <= 1 and not (x1 < m['x0'] - 8 or x0 > m['x1'] + 8):
                m['y1'] = y
                m['x0'] = min(m['x0'], x0)
                m['x1'] = max(m['x1'], x1)
                placed = True
                break
        if not placed:
            merged.append({'y0': y, 'y1': y, 'x0': x0, 'x1': x1})
    return [m for m in merged if m['y1'] - m['y0'] <= 3]

def v_segments(gray, minlen):
    segs = []
    H, W = gray.shape
    for x in range(W):
        for (y0, y1) in runs(gray[:, x]):
            if y1 - y0 + 1 >= minlen:
                segs.append((x, y0, y1))
    merged = []
    for (x, y0, y1) in segs:
        placed = False
        for m in merged:
            if abs(m['x1'] - x) <= 1 and not (y1 < m['y0'] - 8 or y0 > m['y1'] + 8):
                m['x1'] = x
                m['y0'] = min(m['y0'], y0)
                m['y1'] = max(m['y1'], y1)
                placed = True
                break
        if not placed:
            merged.append({'x0': x, 'x1': x, 'y0': y0, 'y1': y1})
    return [m for m in merged if m['x1'] - m['x0'] <= 3]

if __name__ == '__main__':
    path = sys.argv[1]
    minlen = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    im = np.array(Image.open(path).convert('RGB'))
    gray, lum = grayish(im)
    hs = h_segments(gray, minlen)
    vs = v_segments(gray, minlen)
    print(f'{path}: {len(hs)} h-segs, {len(vs)} v-segs')
    for m in hs[:40]:
        print(' H y=%d..%d x=%d..%d len=%d' % (m['y0'], m['y1'], m['x0'], m['x1'], m['x1'] - m['x0'] + 1))
    print(' ...' if len(hs) > 40 else '')
    for m in vs[:40]:
        print(' V x=%d..%d y=%d..%d len=%d' % (m['x0'], m['x1'], m['y0'], m['y1'], m['y1'] - m['y0'] + 1))
