"""QA over an icon set: sizes, blanks, duplicates.

Pure post-processing check (no detection re-run), so it is fast enough to run
after every batch: ~2 s for 4096 icons.

    python3 tools/qa_icons.py                       # cut_icons/ (*.png)
    python3 tools/qa_icons.py icons-512 webp        # repacked set
    python3 tools/qa_icons.py -v                    # per-sheet table

Exit code 1 when something is reported.
"""
import collections
import glob
import hashlib
import os
import sys
import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main(verbose=False, root='cut_icons', ext='png'):
    packs = sorted(p for p in glob.glob(os.path.join(ROOT, root, '*')) if os.path.isdir(p))
    seen = {}
    problems = []
    total = 0
    for pack in packs:
        for part in sorted(glob.glob(os.path.join(pack, '*'))):
            files = sorted(glob.glob(os.path.join(part, '*.' + ext)))
            if not files:
                problems.append('%s/%s: empty' % (os.path.basename(pack), os.path.basename(part)))
                continue
            sizes = collections.Counter()
            for f in files:
                im = Image.open(f)
                sizes[im.size] += 1
                digest = hashlib.md5(open(f, 'rb').read()).hexdigest()
                if digest in seen:
                    problems.append('%s: duplicate of %s' % (os.path.relpath(f, ROOT),
                                                             os.path.relpath(seen[digest], ROOT)))
                seen[digest] = f
                a = np.asarray(im.convert('RGB'))
                if a.mean() < 3:
                    problems.append('%s: blank' % os.path.relpath(f, ROOT))
                elif a.shape[0] < 60 or a.shape[1] < 60:
                    problems.append('%s: tiny %dx%d' % (os.path.relpath(f, ROOT), a.shape[1], a.shape[0]))
            (mw, mh), _n = sizes.most_common(1)[0]
            odd = [s for s in sizes if abs(s[0] - mw) > 0.15 * mw or abs(s[1] - mh) > 0.15 * mh]
            if odd:
                problems.append('%s/%s: size outliers %s (modal %dx%d)'
                                % (os.path.basename(pack), os.path.basename(part), odd, mw, mh))
            total += len(files)
            if verbose:
                print('%-22s %-6s icons=%2d modal=%dx%d variants=%d spread=%dx%d'
                      % (os.path.basename(pack), os.path.basename(part), len(files), mw, mh,
                         len(sizes),
                         max(s[0] for s in sizes) - min(s[0] for s in sizes),
                         max(s[1] for s in sizes) - min(s[1] for s in sizes)))
    print('checked: %d icons in %d packs  (%s/*.%s)' % (total, len(packs), root, ext))
    if problems:
        print('problems (%d):' % len(problems))
        for p in problems[:60]:
            print(' ', p)
        return 1
    print('problems: none')
    return 0


if __name__ == '__main__':
    args = [a for a in sys.argv[1:] if not a.startswith('-')]
    sys.exit(main('-v' in sys.argv or '--verbose' in sys.argv,
                  args[0] if args else 'cut_icons',
                  args[1] if len(args) > 1 else 'png'))
