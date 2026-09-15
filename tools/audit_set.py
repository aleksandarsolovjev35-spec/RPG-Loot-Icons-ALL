"""Audit a finished icon set - find what still makes 4100 icons read as 4100 pictures.

`stylize.py` measures styles against each other; this tool measures the *set*:
how consistent the icons are with one another, and which specific files are the
outliers.  Every number below is computed per icon and then the spread over the
whole set is what matters - in a loot grid an icon is never looked at alone.

    python3 tools/audit_set.py                      # audit the actual set
    python3 tools/audit_set.py --set icons-256-base # compare with the clean base
    python3 tools/audit_set.py --md docs/quality-lab/audit.md --sheets docs/quality-lab
    python3 tools/audit_set.py --json               # machine readable

Per-icon metrics (see `measure_one`):

  bg        luma of the 8 px border ring - what the sheet drew behind the item
  fill      share of the frame the item covers
  bbox/c    bounding box and centroid of the item, in fractions of the frame
  touch     share of the border ring that is *item* - art running off the frame
  expo/p95  mean and 95th percentile luma of the item - exposure
  contrast  std of the item's luma
  chroma    mean (max-min) of the item's RGB - how saturated it is
  hf64      detail energy left at 64 px, the size a VTT actually shows
  micro     hf64 divided by the contrast at 64 px - detail *relative* to the
            art, so a flat icon is not punished for being flat

Outlier classes (each one is a fixable defect, not a style question):

  bright-bg   the backdrop is not black (the `quiet` pass did not reach it)
  blank       almost nothing in the frame
  clipped     the item runs into the frame edge, or a neighbour cell bled in
  dull-wash   washed out at 64 px: no micro-contrast left, the icon turns to mush
  exposure    far from the set's exposure - reads as a different render
  dupes       the same art twice (perceptual hash, hamming <= 4/64)
"""
import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    import imgproc as I
except ImportError:
    from tools import imgproc as I

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
FONT_BOLD = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'

RING = 8          # border ring used for the backdrop estimate
THR = 0.20        # luma above which a pixel counts as "item"

# Fixed thresholds for the outlier classes.  They are absolute on purpose: a
# relative one (lowest 2%, median +- 2.5 MAD) would always report the same
# number of icons however good the set gets, which hides the improvement.
LIMITS = dict(bg=10.0,        # backdrop brighter than 10/255 is no longer black
              blank=0.08,     # the item covers less than 8% of the frame
              touch=0.35,     # a third of the border ring is item -> clipped
              micro=0.09,     # detail/contrast at 64 px: below this it is mush
              dark=100.0,     # item luma, in 255ths (p05 of the actual set)
              bright=165.0)   # (p95 of the actual set)


def box(a, r):
    p = np.pad(a, ((r, r), (r, r)), mode='edge')
    c = p.cumsum(0).cumsum(1)
    c = np.pad(c, ((1, 0), (1, 0)), constant_values=0)
    n = (2 * r + 1) ** 2
    return (c[2 * r + 1:, 2 * r + 1:] - c[:-2 * r - 1, 2 * r + 1:]
            - c[2 * r + 1:, :-2 * r - 1] + c[:-2 * r - 1, :-2 * r - 1]) / n


def subject_mask(L, bg, thr=THR):
    m = L > max(bg + 0.05, thr)
    return m & (box(m.astype(np.float32), 3) > 0.35)


def measure_one(path):
    im = Image.open(path).convert('RGB')
    a = np.asarray(im).astype(np.float32) / 255.0
    L = a @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    ring = np.concatenate([L[:RING].ravel(), L[-RING:].ravel(),
                           L[:, :RING].ravel(), L[:, -RING:].ravel()])
    bg = float(np.median(ring))
    m = subject_mask(L, bg)
    d = {'bg': bg * 255.0, 'bgmax': float(np.percentile(ring, 99.5)) * 255.0}
    d['fill'] = float(m.mean())
    edge = np.concatenate([m[:RING].ravel(), m[-RING:].ravel(),
                           m[:, :RING].ravel(), m[:, -RING:].ravel()])
    d['touch'] = float(edge.mean())
    if m.sum() < 30:
        d.update(bw=0.0, bh=0.0, cx=0.5, cy=0.5, expo=0.0, p95=0.0,
                 contrast=0.0, chroma=0.0)
    else:
        ys, xs = np.nonzero(m)
        d['bw'] = (xs.max() - xs.min() + 1) / float(L.shape[1])
        d['bh'] = (ys.max() - ys.min() + 1) / float(L.shape[0])
        d['cx'] = float(xs.mean()) / float(L.shape[1])
        d['cy'] = float(ys.mean()) / float(L.shape[0])
        sel = L[m]
        d['expo'] = float(sel.mean()) * 255.0
        d['p95'] = float(np.percentile(sel, 95)) * 255.0
        d['contrast'] = float(sel.std()) * 255.0
        d['chroma'] = float((a.max(2) - a.min(2))[m].mean()) * 255.0
    # what is left of the icon at the size a VTT shows it
    L64 = np.asarray(Image.fromarray(np.round(L * 255).astype(np.uint8)).resize((64, 64), Image.LANCZOS), np.float32) / 255.0
    hf = np.abs(L64 - box(L64, 1))
    d['hf64'] = float(hf.mean()) * 255.0
    d['micro'] = float(hf.mean() / max(float(L64.std()), 1e-3))
    return d


def dhash(path):
    g = Image.open(path).convert('L').resize((9, 8), Image.LANCZOS)
    a = np.asarray(g, np.float32)
    bits = (a[:, 1:] > a[:, :-1]).ravel()
    v = 0
    for b in bits:
        v = (v << 1) | int(b)
    return np.uint64(v)


def _work(args):
    rel, root = args
    p = os.path.join(root, rel)
    d = measure_one(p)
    d['hash'] = dhash(p)
    d['file'] = rel
    return d


def spread(vals, name, unit=''):
    v = np.asarray(vals, np.float64)
    return {
        'metric': name, 'unit': unit,
        'min': float(v.min()), 'p05': float(np.percentile(v, 5)),
        'median': float(np.median(v)), 'p95': float(np.percentile(v, 95)),
        'max': float(v.max()), 'mean': float(v.mean()), 'std': float(v.std()),
        # robust spread: the middle 90% of the set, immune to a few freaks
        'p05_p95': float(np.percentile(v, 95) - np.percentile(v, 5)),
        'iqr': float(np.percentile(v, 75) - np.percentile(v, 25)),
    }


def find_outliers(rows):
    """Pick the files that are actually broken, with a severity per file."""
    # every entry is (severity, file), severity > 0, worst first
    def _build(pairs):
        return sorted(((s, f) for s, f in pairs if f), key=lambda t: -t[0])

    out = {}
    out['bright-bg'] = _build([(r['bg'], r['file'])
                               for r in rows if r['bg'] > LIMITS['bg']])
    out['blank'] = _build([(1.0 - r['fill'] / LIMITS['blank'], r['file'])
                           for r in rows if r['fill'] < LIMITS['blank']])
    out['clipped'] = _build([(r['touch'], r['file'])
                             for r in rows if r['touch'] > LIMITS['touch']])
    out['dull-wash'] = _build([(1.0 - r['micro'] / LIMITS['micro'], r['file'])
                               for r in rows if r['micro'] < LIMITS['micro']])
    out['dark'] = _build([(LIMITS['dark'] - r['expo'], r['file'])
                          for r in rows if r['expo'] < LIMITS['dark']])
    out['bright'] = _build([(r['expo'] - LIMITS['bright'], r['file'])
                            for r in rows if r['expo'] > LIMITS['bright']])
    return out


def _count(rows, kind):
    """Same classes as `find_outliers`, but counted on saved CSV rows."""
    key, lim = {
        'bright-bg': ('bg', LIMITS['bg']),
        'blank': ('fill', LIMITS['blank']),
        'clipped': ('touch', LIMITS['touch']),
        'dull-wash': ('micro', LIMITS['micro']),
        'dark': ('expo', LIMITS['dark']),
        'bright': ('expo', LIMITS['bright']),
    }[kind]
    above = kind in ('bright-bg', 'clipped', 'bright')
    out = []
    for r in rows:
        v = float(r[key])
        if (v > lim) if above else (v < lim):
            out.append(r['file'])
    return out


def pair_distance(root, a, b):
    """Is this pair the same art?  (mean |delta| in 255ths, 32 px correlation)."""
    xa = np.asarray(Image.open(os.path.join(root, a)).convert('RGB'), np.float32)
    xb = np.asarray(Image.open(os.path.join(root, b)).convert('RGB'), np.float32)
    mean = float(np.abs(xa - xb).mean())
    la = np.asarray(Image.open(os.path.join(root, a)).convert('L').resize((32, 32), Image.LANCZOS), np.float32)
    lb = np.asarray(Image.open(os.path.join(root, b)).convert('L').resize((32, 32), Image.LANCZOS), np.float32)
    corr = float(np.corrcoef(la.ravel(), lb.ravel())[0, 1])
    return mean, corr


def find_dupes(rows, root, hash_dist=5, max_delta=10.0, min_corr=0.97):
    """Perceptual duplicates, in two stages.

    Stage one is a 64-bit dHash over all icons - cheap, but on its own useless
    here: 4100 items contain hundreds of swords with the same silhouette, and the
    pair count grows x2 per unit of hash distance with no knee anywhere.  So every
    candidate pair (distance <= 5, ~160 pairs on this set) is then *verified by
    pixels*: same art means a mean difference under 10/255 and a 32 px correlation
    above 0.97.  On `icons-256` that leaves exactly one pair out of 55 hash
    candidates - look-alikes such as two different swords score 18..67.
    """
    h = np.array([r['hash'] for r in rows], np.uint64)
    files = [r['file'] for r in rows]
    n = len(h)
    cand = []
    for i in range(0, n, 512):
        d = np.bitwise_count(h[i:i + 512, None] ^ h[None, :])
        xs, ys = np.nonzero((d <= hash_dist)
                            & (np.arange(n)[None, :] > (i + np.arange(len(d)))[:, None]))
        cand.extend((i + int(x), int(y)) for x, y in zip(xs.tolist(), ys.tolist()))

    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    kept = []
    for a, b in cand:
        mean, corr = pair_distance(root, files[a], files[b])
        if mean < max_delta and corr > min_corr:
            kept.append((files[a], files[b], mean, corr))
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra
    groups = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)
    out = [[files[i] for i in g] for g in groups.values() if len(g) > 1]
    return out, kept


def montage(root, files, path, cols=8, cell=110, title='', cap=32):
    files = files[:cap]
    if not files:
        return None
    rows = (len(files) + cols - 1) // cols
    pad, lab = 6, 20
    W = cols * (cell + pad) + pad
    H = 26 + rows * (cell + lab + pad) + pad
    cv = Image.new('RGB', (W, H), (22, 22, 26))
    dr = ImageDraw.Draw(cv)
    try:
        f = ImageFont.truetype(FONT_BOLD, 15)
    except OSError:
        f = ImageFont.load_default()
    dr.text((pad, 5), title, fill=(235, 235, 240), font=f)
    for k, rel in enumerate(files):
        im = Image.open(os.path.join(root, rel)).convert('RGB').resize((cell, cell), Image.LANCZOS)
        x = pad + (k % cols) * (cell + pad)
        y = 26 + (k // cols) * (cell + lab + pad)
        cv.paste(im, (x, y))
        dr.text((x, y + cell + 2), rel.split('/')[-1].replace('.webp', ''), fill=(150, 150, 160), font=f)
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    cv.save(path)
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--set', default='icons-256')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--jobs', type=int, default=os.cpu_count() or 1)
    ap.add_argument('--md', default='')
    ap.add_argument('--json', action='store_true')
    ap.add_argument('--rows', default='', help='dump per-icon metrics as CSV (for before/after)')
    ap.add_argument('--out', default='', help='dump the outlier file lists as JSON')
    ap.add_argument('--compare', default='', help='a --rows CSV of another set to diff against')
    ap.add_argument('--sheets', default='', help='write outlier montages into this dir')
    ap.add_argument('--dupes', action='store_true', help='also run the perceptual duplicate search')
    args = ap.parse_args(argv)

    src = os.path.join(ROOT, args.set)
    man = json.load(open(os.path.join(src, 'manifest.json')))
    rels = [e['file'] for e in man['icons']]
    if args.limit:
        rels = rels[:args.limit]

    jobs = [(r, src) for r in rels]
    if args.jobs > 1:
        from multiprocessing import Pool
        with Pool(args.jobs) as p:
            rows = p.map(_work, jobs, chunksize=16)
    else:
        rows = [_work(j) for j in jobs]

    keys = [('bg', 'подложка (яркость рамки)', ''), ('bgmax', 'подложка максимум', ''),
            ('fill', 'заполнение кадра', ''), ('bw', 'ширина предмета', ''),
            ('bh', 'высота предмета', ''), ('cx', 'центр по X', ''),
            ('cy', 'центр по Y', ''), ('touch', 'касание рамки', ''),
            ('expo', 'экспозиция предмета', ''), ('p95', 'света (p95)', ''),
            ('contrast', 'контраст (std)', ''), ('chroma', 'насыщенность', ''),
            ('hf64', 'деталь на 64 px', ''), ('micro', 'микроконтраст 64 px', '')]
    table = [spread([r[k] for r in rows], label) for k, label, _u in keys]
    outliers = find_outliers(rows)
    dupes = None
    if args.dupes:
        dupes, dupe_pairs = find_dupes(rows, src)

    report = {'set': args.set, 'count': len(rows), 'spread': table,
              'outliers': {k: len(v) for k, v in outliers.items()}}
    if dupes is not None:
        report['dupe_groups'] = len(dupes)
        report['dupe_files'] = sum(len(g) for g in dupes)
        report['dupe_pairs'] = [{'a': a, 'b': b, 'mean': round(m, 2), 'corr': round(c, 4)}
                                for a, b, m, c in dupe_pairs]

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print('audit %s: %d icons' % (args.set, len(rows)))
        print('%-22s %7s %7s %7s %7s %7s %8s' % ('метрика', 'min', 'p05', 'med', 'p95', 'max', 'std'))
        for t, (k, _l, _u) in zip(table, keys):
            print('%-22s %7.2f %7.2f %7.2f %7.2f %7.2f %8.2f' % (
                t['metric'], t['min'], t['p05'], t['median'], t['p95'], t['max'], t['std']))
        print()
        for k, v in outliers.items():
            print('%-11s %5d иконок' % (k, len(v)))
        if dupes is not None:
            print('dupes       %5d групп (%d файлов)' % (len(dupes), sum(len(g) for g in dupes)))
            for a, b, m, c in dupe_pairs:
                print('   Δ %.1f  corr %.3f  %s  ==  %s' % (m, c, a, b))

    if args.sheets:
        for k, v in outliers.items():
            files = [f for _s, f in v]
            montage(src, files, os.path.join(args.sheets, 'outliers-%s.png' % k),
                    title='%s  (%d)' % (k, len(v)))
        if dupes:
            flat = [f for g in dupes for f in g]
            montage(src, flat, os.path.join(args.sheets, 'outliers-dupes.png'),
                    title='dupes (%d файлов в %d группах)' % (len(flat), len(dupes)))

    if args.compare and os.path.exists(args.compare):
        import csv as _csv
        with open(args.compare) as fh:
            old = [r for r in _csv.DictReader(fh)]
        print('\nсравнение: %s -> %s' % (args.compare, args.set))
        print('%-22s %8s %8s %8s | %8s %8s %8s' % (
            'метрика', 'std было', 'std стало', 'std Δ', 'сред было', 'сред стало', 'сред Δ'))
        for k, label, _u in keys:
            a = np.array([float(r[k]) for r in old], np.float64)
            b = np.array([r[k] for r in rows], np.float64)
            if a.std() == 0:
                continue
            ch = (b.std() - a.std()) / a.std() * 100.0
            cm = (b.mean() - a.mean()) / max(abs(a.mean()), 1e-9) * 100.0
            print('%-22s %8.2f %8.2f %+7.1f%% | %8.2f %8.2f %+7.1f%%' % (
                label, a.std(), b.std(), ch, a.mean(), b.mean(), cm))
        print()
        for k in outliers:
            print('  %-11s %5d -> %5d' % (k, len(_count(old, k)), len(outliers[k])))

    if args.out:
        dump = {k: [f for _s, f in v] for k, v in outliers.items()}
        if dupes:
            dump['dupes'] = [f for g in dupes for f in g]
        with open(args.out, 'w') as fh:
            json.dump(dump, fh, ensure_ascii=False, indent=1)
        if not args.json:
            print('written', args.out)

    if args.rows:
        cols = ['file'] + [k for k, _l, _u in keys]
        with open(args.rows, 'w') as fh:
            fh.write(','.join(cols) + '\n')
            for r in rows:
                fh.write(','.join(str(r.get(c, '')) if c == 'file' else '%.6f' % r[c] for c in cols) + '\n')
        if not args.json:
            print('written', args.rows)

    if args.md:
        lines = ['# Аудит набора `%s`' % args.set, '',
                 'Иконок: %d. Замеры: `tools/audit_set.py --set %s --md ...`.' % (len(rows), args.set),
                 ' Важно не среднее, а **разброс**: в сетке лута иконка всегда соседствует с другими.', '',
                 '| метрика | min | p05 | медиана | p95 | max | std |', '|---|---|---|---|---|---|---|']
        for t in table:
            lines.append('| %s | %.2f | %.2f | %.2f | %.2f | %.2f | %.2f |' % (
                t['metric'], t['min'], t['p05'], t['median'], t['p95'], t['max'], t['std']))
        lines += ['', '## Выбросы (конкретные файлы)', '',
                  '| класс | сколько | что не так |', '|---|---|---|']
        why = {
            'bright-bg': 'подложка не чёрная — `quiet` до неё не дошёл',
            'blank': 'в кадре почти ничего нет',
            'clipped': 'предмет упирается в рамку (или в кадр попал сосед)',
            'dull-wash': 'на 64 px превращается в кашу: нет микроконтраста',
            'dark': 'предмет темнее %g/255 (p05 актуального набора)' % LIMITS['dark'],
            'bright': 'предмет светлее %g/255 (p95 актуального набора)' % LIMITS['bright'],
            'dupes': 'одинаковый арт: кандидат по хешу (≤ 5 из 64), подтверждённый '
                     'пиксельно (Δ < 10/255, корреляция > 0.97)',
        }
        for k, v in outliers.items():
            lines.append('| `%s` | %d | %s |' % (k, len(v), why.get(k, '')))
        if dupes is not None:
            lines.append('| `dupes` | %d групп / %d файлов | %s |' % (len(dupes), sum(len(g) for g in dupes), why['dupes']))
            for a, b, m, c in dupe_pairs:
                lines.append('')
                lines.append('  * `%s` == `%s` (Δ %.1f, корреляция %.3f)' % (a, b, m, c))
        os.makedirs(os.path.dirname(args.md) or '.', exist_ok=True)
        open(args.md, 'w').write('\n'.join(lines) + '\n')
        if not args.json:
            print('\nwritten', args.md)
    return 0


if __name__ == '__main__':
    sys.exit(main())
