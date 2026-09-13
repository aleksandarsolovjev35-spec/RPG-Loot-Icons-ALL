"""Universal icon cutter for RPG loot sprite sheets.

Output contract (agreed with the user):
* keep the black background - crops are opaque rectangles, not tight
  transparent cutouts;
* a crop is exactly the area INSIDE the gray/white frame ("кайма"),
  the frame itself is not included (cut strictly along its inner edge,
  frame pixels that bleed inside are painted black);
* sheets WITHOUT a frame are cut too - see "ghost grid" below.

Two sheet kinds
---------------
A. Framed sheets (all packs except 39/40): every cell is outlined by a thin
   gray box.  The box is the ruler, so the cut is the box interior.

B. Frameless sheets (packs 39/40): icons sit on bare black, there is nothing
   to trace.  What a frame would have done is reproducible from the layout
   alone: cells are separated by a common all-background gutter (~28 px), so
   *rows and columns can be cut independently* by looking at where content
   exists at all.  Algorithm ("ghost grid"):

     1. content mask = luminance > 30 or saturated color (same as before);
     2. project the mask on both axes; maximal all-background runs (>= 4 px,
        margins included) are gutters -> the spans between consecutive
        gutters are the rows and the columns of the grid.  Because a gutter
        is background for the WHOLE sheet width/height, the resulting grid is
        consistent (all cells of a column share their x-range) and clips
        nothing, exactly like a real frame box would;
     3. a cell that overflows the frame is the only way an icon can eat a
        gutter, which would merge two columns/rows.  The gutters that ARE
        found fix the pitch, so the grid line is predicted by least squares
        (pitch + origin) and inserted where a span is ~2x the pitch;
     4. cells with (almost) no content are dropped; the rest are cropped
        as-is, no inset, no repainting (there is no frame to remove).

Sizes are not normalised - a crop is what the cell is (~148x148 on framed
packs, ~152x152 on frameless ones).
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

# a sheet is "framed" when it has long thin gray lines: at least this many
# sheet rows / columns are >= this many gray pixels.  Measured on all 82
# sheets: framed ones score 10..60 rows, frameless (39/40) score exactly 0.
FRAME_ROWS = 4
FRAME_ROW_PX = 400
FRAME_COLS = 4
FRAME_COL_PX = 300


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


# --------------------------------------------------------------------------
# frameless sheets: the "ghost grid"
# --------------------------------------------------------------------------

def bg_gaps(profile, minlen=4):
    """maximal all-background runs of a content profile, margins included"""
    return [(a, b) for a, b in _runs_of(~np.asarray(profile, bool))
            if b - a + 1 >= minlen]


def fit_pitch(gaps, n, lo=90, hi=240):
    """pitch + origin of the grid, fitted on the centres of interior gutters.

    Returns (pitch, origin) or (None, None) when there is not enough evidence.
    """
    c = [0.5 * (a + b) for a, b in gaps if a > 0 and b < n - 1]
    if len(c) < 3:
        return None, None
    d = np.diff(c)
    d = d[(d >= lo) & (d <= hi)]
    if len(d) == 0:
        return None, None
    pitch = float(np.median(d))
    c = np.asarray(c)
    k = np.round((c - c[0]) / pitch)
    origin = float(np.average(c - k * pitch))
    return pitch, origin


def _split_span(span, pitch, minlen):
    """a span of k*pitch (a gutter eaten by an overlapping icon) -> k cells"""
    lo, hi = span
    k = int(round((hi - lo + 1) / pitch)) if pitch else 1
    if k < 2:
        return [span]
    step = (hi - lo + 1) / k
    parts = [(lo + int(round(i * step)), lo + int(round((i + 1) * step)) - 1)
             for i in range(k)]
    parts = [p for p in parts if p[1] - p[0] + 1 >= minlen]
    return parts or [span]


def cell_spans(gaps, n, pitch, minlen=100):
    """spans between gutters, over-long spans split on the fitted lattice."""
    spans, lo = [], 0
    for a, b in gaps:
        if a - lo >= minlen:
            spans.append((lo, a - 1))
        lo = b + 1
    if n - lo >= minlen:
        spans.append((lo, n - 1))
    out = []
    for s in spans:
        out.extend(_split_span(s, pitch, minlen) if pitch else [s])
    return out


def find_cells_borderless(content, min_gutter=4, min_content=200):
    """rectangles (xL, yT, xR, yB) of a frameless sheet + grid diagnostics.

    Returns ([], stats) when the layout gives no trustworthy grid (no gutters
    at all, i.e. icons that touch each other everywhere) - such a sheet is
    skipped rather than cut blindly.
    """
    H, W = content.shape
    rows = bg_gaps(content.sum(1), min_gutter)
    cols = bg_gaps(content.sum(0), min_gutter)
    ph, _ = fit_pitch(rows, H)
    pw, _ = fit_pitch(cols, W)
    ys = cell_spans(rows, H, ph)
    xs = cell_spans(cols, W, pw)
    stats = dict(rows=len(ys), cols=len(xs), pitch_h=ph, pitch_w=pw)
    if len(ys) < 2 or len(xs) < 2:
        return [], stats
    rects = []
    for y0, y1 in ys:
        for x0, x1 in xs:
            if int(content[y0:y1 + 1, x0:x1 + 1].sum()) < min_content:
                continue
            rects.append((x0, y0, x1, y1))
    hs = [b - a + 1 for a, b in ys]
    ws = [b - a + 1 for a, b in xs]
    stats.update(cell=(max(ws), max(hs)), spread=(max(ws) - min(ws), max(hs) - min(hs)))
    return rects, stats


def has_frame(gray):
    """long thin gray lines present -> the sheet is outlined by frames"""
    return ((gray.sum(1) >= FRAME_ROW_PX).sum() >= FRAME_ROWS and
            (gray.sum(0) >= FRAME_COL_PX).sum() >= FRAME_COLS)


def order_reading(rects):
    """rows by yT, cells inside a row by xL"""
    rows = []
    for rect in sorted(rects, key=lambda t: t[1]):
        if rows and rect[1] - rows[-1][0] <= 40:
            rows[-1][1].append(rect)
        else:
            rows.append((rect[1], [rect]))
    return [rect for _, rr in rows for rect in sorted(rr, key=lambda t: t[0])]


def save_icon(rgb, path):
    """Compact 255-color palette PNG."""
    q = Image.fromarray(rgb, 'RGB').quantize(colors=255, dither=Image.Dither.NONE)
    q.save(path, compress_level=9)


def process_sheet(path, out_dir, montage_path=None, stats=None):
    """Cut one sheet. Returns (saved icons, cell rectangles); `stats` (dict)
    receives diagnostics: mode, rows, cols, cell size, pitch."""
    im = np.array(Image.open(path).convert('RGB')).astype(np.uint8)
    H, W, _ = im.shape
    r = im[:, :, 0].astype(int)
    g = im[:, :, 1].astype(int)
    b = im[:, :, 2].astype(int)
    lum = (r + g + b) / 3.0
    sat = np.maximum(np.maximum(abs(r - g), abs(g - b)), abs(r - b))
    content = (lum > 30) | (sat > 25)

    gray, _ = grayish(im)
    framed = has_frame(gray)
    info = {}
    if framed:
        rects, _hl, _vl = find_rectangles(h_segments(gray, 30), v_segments(gray, 30))
        if not rects:
            return [], []                      # framed sheet, frames unreadable
        mode, inset = 'frame', 2               # cut inside the frame line
    else:
        rects, info = find_cells_borderless(content)
        mode, inset = 'grid', 0                # nothing to trim off
        if not rects:
            return [], []                      # no gutters -> no trustworthy grid

    if stats is not None:
        stats.update(mode=mode, rows=len({q[1] for q in rects}),
                     cols=len({q[0] for q in rects}),
                     pitch_w=info.get('pitch_w'), pitch_h=info.get('pitch_h'),
                     cell=(max(q[2] - q[0] for q in rects) - 2 * inset + 1,
                           max(q[3] - q[1] for q in rects) - 2 * inset + 1))

    if mode == 'frame':
        # repaint whatever of the frame line bled inside the crop
        gray2 = (sat <= 14) & (lum >= 30) & (lum <= 240)
        thick = dilate(erode(content, 1), 1)
        thin = content & ~thick
        remove = np.zeros((H, W), bool)
        for rect in rects:
            remove |= classify_band(content, thin, gray2, rect)
    else:
        remove = np.zeros((H, W), bool)
    rects = order_reading(rects)

    os.makedirs(out_dir, exist_ok=True)
    saved = []
    for k, (xL, yT, xR, yB) in enumerate(rects, 1):
        x0, x1 = xL + inset, xR - inset
        y0, y1 = yT + inset, yB - inset
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
        st = {}
        saved, rects = process_sheet(p, out_dir, stats=st)
        total += len(saved)
        grid = '%dx%d' % (st.get('rows') or 0, st.get('cols') or 0)
        cell = 'x'.join(str(v) for v in st['cell']) if st.get('cell') else '-'
        pitch = st.get('pitch_w')
        pitch = 'pitch=%.1f' % pitch if pitch else ''
        print('%-46s %-5s grid=%-6s cell=%-8s %-11s icons=%2d'
              % (os.path.relpath(p, ROOT), st.get('mode', '?'), grid, cell, pitch, len(saved)))
    print('TOTAL icons:', total)
