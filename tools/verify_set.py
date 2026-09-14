"""Verify an icon set against its manifest - the contract written by the tools.

`manifest.json` promises, for every icon, the file name, its size in bytes and its
md5; the set as a whole promises `count` entries and `total_bytes`.  This tool
checks all of it against the files on disk plus the WebP header, so a set that
left the toolchain by hand or half-copied fails loudly instead of silently.

    python3 tools/verify_set.py                    # icons-256 (actual set)
    python3 tools/verify_set.py icons-256-base     # clean base set
    python3 tools/verify_set.py icons-512 --no-md5 # quick pass, skip hashing
    python3 tools/verify_set.py icons-256 --json   # machine-readable summary

Exit code is 0 when the set is consistent, 1 otherwise (every problem is printed).
"""
import argparse
import hashlib
import json
import os
import struct
import sys
from collections import Counter


def webp_size(path):
    """(width, height) from the WebP header, or None if it is not a WebP."""
    with open(path, 'rb') as fh:
        head = fh.read(40)
    if len(head) < 30 or head[:4] != b'RIFF' or head[8:12] != b'WEBP':
        return None
    kind = head[12:16]
    if kind == b'VP8X':
        return (int.from_bytes(head[24:27], 'little') + 1,
                int.from_bytes(head[27:30], 'little') + 1)
    if kind == b'VP8 ':
        return (struct.unpack('<H', head[26:28])[0] & 0x3FFF,
                struct.unpack('<H', head[28:30])[0] & 0x3FFF)
    if kind == b'VP8L':
        bits = int.from_bytes(head[21:25], 'little')
        return (bits & 0x3FFF) + 1, ((bits >> 14) & 0x3FFF) + 1
    return None


def verify(set_dir, check_md5=True, verbose=True):
    """Returns (problems, summary dict)."""
    man_path = os.path.join(set_dir, 'manifest.json')
    problems = []
    if not os.path.exists(man_path):
        return ['no manifest.json in %s' % set_dir], {}
    man = json.load(open(man_path))
    icons = man.get('icons', [])
    role = man.get('role', 'canon')
    style = man.get('style') or {}
    if verbose:
        print('%s  role=%s%s' % (set_dir, role,
                                 '' if not style else '  style=%s' % '+'.join(style.get('presets', []))))
        print('  manifest: %d entries, count=%s, total_bytes=%.1f MB'
              % (len(icons), man.get('count'), man.get('total_bytes', 0) / 1e6))

    if man.get('count') != len(icons):
        problems.append('count=%s but %d entries' % (man.get('count'), len(icons)))

    listed = set()
    total = 0
    dup = Counter()
    size_field = man.get('size') or 0
    for e in icons:
        rel = e['file']
        if rel in listed:
            problems.append('duplicate manifest entry: %s' % rel)
        listed.add(rel)
        path = os.path.join(set_dir, rel)
        if not os.path.exists(path):
            problems.append('missing file: %s' % rel)
            continue
        raw = open(path, 'rb').read()
        total += len(raw)
        if len(raw) != e.get('bytes', -1):
            problems.append('bytes mismatch: %s (%d on disk, %d in manifest)'
                            % (rel, len(raw), e.get('bytes', -1)))
        if check_md5:
            got = hashlib.md5(raw).hexdigest()
            if got != e.get('md5'):
                problems.append('md5 mismatch: %s' % rel)
            dup[got] += 1
        if size_field:
            dim = webp_size(path)
            if dim != (size_field, size_field):
                problems.append('size mismatch: %s is %s, manifest says %dx%d'
                                % (rel, dim, size_field, size_field))

    extra = []
    for root, _dirs, files in os.walk(set_dir):
        for f in files:
            rel = os.path.relpath(os.path.join(root, f), set_dir)
            if rel != 'manifest.json' and rel not in listed:
                extra.append(rel)
    if extra:
        problems.append('%d file(s) on disk are not in the manifest, e.g. %s'
                        % (len(extra), ', '.join(sorted(extra)[:3])))
    if man.get('total_bytes') not in (None, total):
        problems.append('total_bytes=%s, sum on disk=%d' % (man.get('total_bytes'), total))
    dupes = [h for h, n in dup.items() if n > 1]
    if dupes:
        problems.append('%d duplicated md5 group(s)' % len(dupes))

    summary = dict(set=set_dir, role=role, icons=len(icons), bytes=total,
                   problems=len(problems),
                   dup_md5=len(dupes),
                   style=style.get('presets') or None,
                   md5_checked=bool(check_md5))
    if verbose:
        print('  checked: %d files, %.1f MB, md5 %s'
              % (len(listed), total / 1e6, 'yes' if check_md5 else 'skipped'))
        if problems:
            for p in problems[:20]:
                print('  PROBLEM: %s' % p)
            if len(problems) > 20:
                print('  ... and %d more' % (len(problems) - 20))
        else:
            print('  problems: none')
    return problems, summary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('set_dir', nargs='?', default='icons-256', help='set directory (default icons-256)')
    ap.add_argument('--no-md5', dest='md5', action='store_false', help='skip hashing (fast pass)')
    ap.add_argument('--json', action='store_true', help='print the summary as JSON')
    args = ap.parse_args(argv)

    problems, summary = verify(args.set_dir, check_md5=args.md5, verbose=not args.json)
    if args.json:
        print(json.dumps(dict(summary, problem_list=problems), indent=1))
    return 1 if problems else 0


if __name__ == '__main__':
    sys.exit(main())
