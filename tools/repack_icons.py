"""Repack cut icons at full colour: enhance, upscale, re-encode, emit a manifest.

Why: `cut_icons.py` saves 255-colour palette PNGs, which throws away >97% of
the source gradients (median 9339 unique colours per crop), and the crops are
not normalised (21 sizes, 146-153 px, 40% not square).  This tool re-cuts the
sheets with the very same detector (`cut_icons.iter_cell_crops`, so numbering
and cell geometry match `cut_icons/` exactly) and re-encodes the crops:

    crop (full colour)  ->  enhance (tools/imgproc.py, see --raw to skip)
                        ->  resize to --size (Lanczos by default)
                        ->  encode (webp / avif / png)
                        ->  icons-<size>/<pack>/partN/icon_NNN.<ext>
                        ->  icons-<size>/manifest.json

The enhance pass attacks the four artefacts of the source chain (a ~148 px
JPEG crop stretched 3.46x), all of them measured on this set - see
`imgproc.py` and the "noise" section of the README:

  1. JPEG mosquito noise and 8 px blockiness in the flats -> guided denoise;
  2. ringing of the resampler around silhouettes on black -> anti-ringing
     clamp in `resample_ar` (a light object on black gets a halo up to 19/255
     with plain Lanczos, 0 with the clamp);
  3. pixel staircase of the small artwork -> steered blur along the contour;
  4. soft contours -> masked unsharp (threshold + clamp: no halo, no noise
     amplification - a plain unsharp was visible and is still not used).

Measured on the full set (every third icon, see the README table): flat-area
noise 0.112 -> 0.037 levels of 255 (median per icon, p90 0.285 -> 0.135), halo
around silhouettes 0.457 -> 0.237 levels at p99, fine-detail energy kept at
hf 0.948 of an honest resize, and at q92 the files are smaller than before.

Examples
--------
    python3 tools/repack_icons.py                       # 512px webp q92, enhanced
    python3 tools/repack_icons.py --raw                 # old pipeline, no enhance
    python3 tools/repack_icons.py --size 256 --quality 92
    python3 tools/repack_icons.py --format avif --quality 70   # ~40% smaller
    python3 tools/repack_icons.py --size 0              # native crop size
    python3 tools/repack_icons.py --sharpen 0 --steer 0 --denoise 0   # tune
    python3 tools/repack_icons.py --limit 1 --out /tmp/x   # smoke test
"""
import argparse
import glob
import hashlib
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
from PIL import Image

try:
    import cut_icons as C
    import imgproc as I
except ImportError:
    from tools import cut_icons as C
    from tools import imgproc as I

ROOT = C.ROOT
RESAMPLE = {'lanczos': Image.LANCZOS, 'bicubic': Image.BICUBIC, 'bilinear': Image.BILINEAR}
EXT = {'webp': 'webp', 'avif': 'avif', 'png': 'png'}

# enhanced-pipeline knobs exposed on the command line (None -> imgproc default)
ENHANCE_KEYS = ('denoise', 'black', 'steer', 'sharpen', 'sharpen_sigma',
                'sharpen_clamp', 'laplacian', 'stretch')


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--size', type=int, default=512,
                    help='output side in px, square; 0 = keep the native crop size (default 512)')
    ap.add_argument('--format', choices=sorted(EXT), default='webp')
    ap.add_argument('--quality', type=int, default=92, help='lossy quality, ignored with --lossless')
    ap.add_argument('--lossless', action='store_true', help='lossless encode (webp/avif/png)')
    ap.add_argument('--resample', choices=sorted(RESAMPLE), default='lanczos')
    ap.add_argument('--raw', action='store_true',
                    help='skip the enhance pass (reproduce the pre-2026 pipeline exactly)')
    ap.add_argument('--no-ar-clamp', action='store_true',
                    help='resample without the anti-ringing clamp (plain Lanczos)')
    ap.add_argument('--denoise', type=float, default=None,
                    help='source denoise strength, 0 disables (default 0.018)')
    ap.add_argument('--black', type=float, default=None,
                    help='luma below which the black background is flattened (default 0.010)')
    ap.add_argument('--steer', type=float, default=None,
                    help='along-contour blur sigma, 0 disables (default 2.2)')
    ap.add_argument('--sharpen', type=float, default=None,
                    help='masked unsharp amount, 0 disables (default 0.35)')
    ap.add_argument('--sharpen-sigma', type=float, default=None, dest='sharpen_sigma',
                    help='masked unsharp radius in px (default 1.6)')
    ap.add_argument('--laplacian', type=float, default=None,
                    help='mid-band detail amount, 0 disables (default 0)')
    ap.add_argument('--stretch', action='store_true',
                    help='true sinc reconstruction width instead of the Pillow convention')
    ap.add_argument('--out', default=None, help='output dir (default icons-<size>)')
    ap.add_argument('--manifest', default='manifest.json', help='manifest name inside --out')
    ap.add_argument('--only', action='append', default=None,
                    help='substring filter on the sheet path, repeatable')
    ap.add_argument('--jobs', type=int, default=0, help='encoder processes (0 = cpu count)')
    ap.add_argument('--limit', type=int, default=0, help='stop after N sheets (smoke test)')
    ap.add_argument('--force', action='store_true', help='re-encode icons that already exist')
    ap.add_argument('--dry-run', action='store_true',
                    help='measure only: encode the first 12 icons of each sheet, write nothing')
    return ap.parse_args(argv)


def enhance_cfg(args):
    """The imgproc.enhance() config for this run, or None with --raw."""
    if args.raw:
        return None
    cfg = dict(I.DEFAULTS)
    cfg['clamp'] = not args.no_ar_clamp
    if args.stretch:
        cfg['stretch'] = True
    for key in ENHANCE_KEYS:
        val = getattr(args, key, None)
        if val is not None:
            cfg[key] = val
    cfg['size'] = args.size or 0
    return cfg


def encode_one(task):
    """(array, size, fmt, quality, lossless, resample[, cfg]) -> encoded bytes."""
    arr, size, fmt, quality, lossless, resample = task[:6]
    cfg = task[6] if len(task) > 6 else None
    if cfg is not None:
        arr = I.enhance(arr, cfg)
    im = Image.fromarray(arr, 'RGB')
    if size and im.size != (size, size):
        im = im.resize((size, size), RESAMPLE[resample])
    if fmt == 'png':
        if lossless:
            import io
            buf = io.BytesIO()
            im.save(buf, 'PNG', optimize=True, compress_level=9)
            return buf.getvalue()
        q = im.quantize(colors=256, dither=Image.Dither.NONE)
        import io
        buf = io.BytesIO()
        q.save(buf, 'PNG', optimize=True, compress_level=9)
        return buf.getvalue()
    import io
    buf = io.BytesIO()
    if lossless:
        im.save(buf, fmt.upper(), lossless=True, method=6)
    else:
        im.save(buf, fmt.upper(), quality=quality, method=6)
    return buf.getvalue()


def sheets():
    return sorted(glob.glob(os.path.join(ROOT, 'RPG Loot Icons */*.jpg')))


def sheet_dirs(path):
    pack = os.path.basename(os.path.dirname(path))
    part = 'part' + os.path.basename(path).split('Part ')[-1].replace('.jpg', '')
    return pack, part


def main(argv=None):
    args = parse_args(argv)
    out_root = args.out or os.path.join(ROOT, 'icons-%d' % args.size if args.size else 'icons-native')
    out_root = os.path.abspath(out_root)
    jobs = args.jobs or (os.cpu_count() or 2)
    ext = EXT[args.format]
    ecfg = enhance_cfg(args)

    todo = sheets()
    if args.only:
        todo = [p for p in todo if any(o in p for o in args.only)]
    if args.limit:
        todo = todo[:args.limit]
    if not todo:
        print('no sheets matched')
        return 1

    print('sheets: %d | out: %s | %s %s %s | size=%s | enhance=%s | workers=%d'
          % (len(todo), os.path.relpath(out_root, ROOT), args.format.upper(),
             'lossless' if args.lossless else 'q%d' % args.quality,
             args.resample, args.size or 'native',
             'off (--raw)' if ecfg is None else
             'denoise=%.3g steer=%.3g sharpen=%.3g laplacian=%.3g ar-clamp=%s' %
             (ecfg['denoise'], ecfg['steer'], ecfg['sharpen'],
              ecfg.get('laplacian') or 0, ecfg['clamp']),
             jobs))

    pool = Pool(jobs)
    entries, total_bytes, skipped, failed = [], 0, 0, []
    modes = {}
    t0 = time.time()
    for si, path in enumerate(todo, 1):
        stats = {}
        prep = C.prepare_cells(path, stats)
        pack, part = sheet_dirs(path)
        if prep is None:
            failed.append((path, 'unreadable'))
            print('  !! %s: skipped (detector found no cells)' % os.path.relpath(path, ROOT))
            continue
        modes[stats.get('mode')] = modes.get(stats.get('mode'), 0) + 1
        crops = [(k, xy, crop) for k, xy, crop in C.iter_cell_crops(path, None)]
        out_dir = os.path.join(out_root, pack, part)
        if not args.dry_run:
            os.makedirs(out_dir, exist_ok=True)

        if args.dry_run:
            sample = crops[:12]
            tasks = [(c, args.size, args.format, args.quality, args.lossless, args.resample, ecfg)
                     for _k, _xy, c in sample]
            sizes = [len(b) for b in pool.map(encode_one, tasks)]
            avg = sum(sizes) / len(sizes)
            per_sheet = avg * len(crops)
            total_bytes += int(per_sheet)
            print('  %-44s %-5s icons=%2d avg=%5.1f KB  ~%5.1f MB/sheet  ~%5.0f MB total'
                  % (os.path.relpath(path, ROOT), stats.get('mode'), len(crops),
                     avg / 1024, per_sheet / 2**20, (total_bytes + 0) / 2**20))
            continue

        tasks, meta = [], []
        for k, xy, crop in crops:
            name = 'icon_%03d.%s' % (k, ext)
            fpath = os.path.join(out_dir, name)
            if os.path.exists(fpath) and not args.force:
                skipped += 1
                n = os.path.getsize(fpath)
                total_bytes += n
                entries.append(entry(path, pack, part, k, xy, crop.shape, name, n,
                                     digest=hashlib.md5(open(fpath, 'rb').read()).hexdigest()))
                continue
            tasks.append((crop, args.size, args.format, args.quality, args.lossless,
                          args.resample, ecfg))
            meta.append((k, xy, name, fpath, crop.shape))

        for (k, xy, name, fpath, native), data in zip(meta, pool.map(encode_one, tasks)):
            with open(fpath, 'wb') as fh:
                fh.write(data)
            total_bytes += len(data)
            entries.append(entry(path, pack, part, k, xy, native, name, len(data),
                                 digest=hashlib.md5(data).hexdigest()))

        print('  %-44s %-5s icons=%2d  [%d/%d] %5.0f MB total, %.0fs'
              % (os.path.relpath(path, ROOT), stats.get('mode'), len(crops), si, len(todo),
                 total_bytes / 2**20, time.time() - t0), flush=True)

    pool.close()
    pool.join()

    if not args.dry_run:
        manifest = {
            'generated': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
            'source_set': 'cut_icons/ (same detector, same numbering)',
            'crop': 'tools/cut_icons.py:iter_cell_crops',
            'enhance': enhance_report(ecfg, args),
            'format': args.format,
            'quality': None if args.lossless else args.quality,
            'lossless': bool(args.lossless),
            'size': args.size or 'native',
            'resample': args.resample,
            'sheet_modes': modes,
            'count': len(entries),
            'total_bytes': total_bytes,
            'icons': entries,
        }
        mpath = os.path.join(out_root, args.manifest)
        with open(mpath, 'w', encoding='utf-8') as fh:
            json.dump(manifest, fh, ensure_ascii=False, indent=1)
        print('manifest: %s (%d icons, %.1f MB total, skipped %d already present)'
              % (os.path.relpath(mpath, ROOT), len(entries), total_bytes / 2**20, skipped))

    if failed:
        print('FAILED sheets:')
        for p, why in failed:
            print('   %s  (%s)' % (os.path.relpath(p, ROOT), why))
    print('done in %.0fs' % (time.time() - t0))
    return 0


def enhance_report(ecfg, args):
    """What the enhance pass did, so a rebuild can be reproduced from the manifest."""
    if ecfg is None:
        return {'enabled': False, 'pipeline': 'raw (no enhance pass)'}
    cfg = {k: v for k, v in ecfg.items() if k != 'size'}
    return {'enabled': True, 'pipeline': 'tools/imgproc.py:enhance', 'config': cfg}


def entry(path, pack, part, index, xy, native, name, nbytes, digest):
    return {
        'file': '%s/%s/%s' % (pack, part, name),
        'pack': pack,
        'part': part,
        'index': index,
        'source': os.path.relpath(path, ROOT),
        'cell': [int(xy[0]), int(xy[1])],
        'native': [int(native[1]), int(native[0])],
        'bytes': int(nbytes),
        'md5': digest,
    }


if __name__ == '__main__':
    sys.exit(main())
