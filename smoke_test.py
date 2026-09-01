"""Exercise the app's data layer without streamlit, and check the default view.

    python3 smoke_test.py [--data DIR]

Checks that the default selection of the app reproduces the paper, that every
option list in meta drives real files, and that the tables build for a handful
of awkward selections.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dotrecon_data as D   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=D.DEFAULT_DATA)
    a = ap.parse_args()
    M = D.load_meta(a.data)
    fails = []

    print('meta: %d parcels, %d probes, %d head models, %d metrics, %d tiers, '
          '%d regularizations' % (len(M['parcels']), len(M['probes']),
                                  len(M['head_models']), len(M['metrics']),
                                  len(M['tiers']), len(M['regularizations'])))

    # 1. every option list points at something real
    print('\n1. option lists resolve to files')
    for p in M['probes']:
        probe = p['dir']
        got = D.available_regs(a.data, probe, M)
        want = [r['key'] for r in M['regularizations']]
        missing = [k for k in want if k not in got]
        print('   %-16s %2d of %2d regularizations present%s'
              % (p['paper'], len(got), len(want),
                 '  missing ' + ', '.join(missing) if missing else ''))
        src = D.load_sources(a.data, probe)
        for key in got:
            Mx, mask = D.load_metrics(a.data, probe, key)
            n = len(src['subject']) if mask is None else int(mask.sum())
            if Mx.shape != (n, len(M['head_models']), len(M['metrics'])):
                fails.append('%s/%s shape %s' % (probe, key, Mx.shape))
    print('   all metric arrays match sources x head models x metrics: %s'
          % ('yes' if not fails else 'NO'))

    # 2. the default view reproduces the paper
    print('\n2. default view, original head model, across subjects')
    want = {'sparse': 11.5, 'HD': 9.2, 'UHD': 8.2}
    for p in M['probes']:
        probe = p['dir']
        src = D.load_sources(a.data, probe)
        keep = D.source_filter(M, src, tier=M['defaults']['tier'])
        rows = D.summary_table(a.data, M, probe, [M['defaults']['reg']], 'LOCA',
                               keep, src, head_models=['HM1'], how='across')
        got = rows[0]['_sort']
        ok = abs(got - want[probe]) <= 0.05
        print('   %-16s %.2f mm   paper %.1f   %s'
              % (p['paper'], got, want[probe], 'OK' if ok else 'MISS'))
        if not ok:
            fails.append('default %s gives %.2f' % (probe, got))

    # 3. tables build for awkward selections
    print('\n3. tables build under awkward selections')
    probe = 'HD'
    src = D.load_sources(a.data, probe)
    cases = [
        ('all tiers, no ROI', dict(tier='all')),
        ('min tier, Limbic', dict(tier='min', net_level='7', net='Limbic')),
        ('max tier, one 17-network', dict(tier='max', net_level='17',
                                          net='DefaultC')),
        ('left hemisphere only', dict(tier='median', hemi='left')),
        ('deep sources only', dict(tier='median', depth_range=(35.0, 99.0))),
        ('an empty selection', dict(tier='median', depth_range=(99.0, 100.0))),
    ]
    for name, kw in cases:
        keep = D.source_filter(M, src, **kw)
        rows = D.summary_table(a.data, M, probe, [M['defaults']['reg']], 'ERES',
                               keep, src, how='across')
        prows = D.parcel_table(a.data, M, probe, M['defaults']['reg'], 'ERES',
                               keep, src, how='across', max_rows=5)
        print('   %-26s %6d sources   %2d summary rows   %d parcel rows'
              % (name, int(keep.sum()), len(rows), len(prows)))
        if int(keep.sum()) and not prows:
            fails.append('%s built no parcel rows' % name)

    # 4. every metric is selectable and produces finite numbers somewhere
    print('\n4. every metric produces a number')
    keep = D.source_filter(M, src, tier=M['defaults']['tier'])
    for m in M['metrics']:
        rows = D.summary_table(a.data, M, probe, [M['defaults']['reg']],
                               m['name'], keep, src, how='across')
        vals = [r['_sort'] for r in rows]
        finite = int(np.isfinite(vals).sum())
        print('   %-24s %d of %d head models finite   range %s'
              % (m['name'], finite, len(vals),
                 'all n/a' if finite == 0
                 else '%.2f to %.2f' % (np.nanmin(vals), np.nanmax(vals))))
        if finite == 0 and m['name'] != 'SNR':
            fails.append('%s is all n/a' % m['name'])

    # 5. unreachable parcels carry their explanation
    print('\n5. parcels with no sampled source')
    for p in M['probes']:
        probe = p['dir']
        src = D.load_sources(a.data, probe)
        keep = D.source_filter(M, src, tier='all')
        prows = D.parcel_table(a.data, M, probe, M['defaults']['reg'], 'LOCA',
                               keep, src, how='across', include_unsampled=True)
        un = [r for r in prows if r['Reachable'] != 'yes']
        with_ctx = [r for r in un if r['Median sensitivity'] != 'n/a']
        print('   %-16s %4d of %d parcels have no sampled source, %d of those '
              'still carry a sensitivity and a depth'
              % (p['paper'], len(un), len(prows), len(with_ctx)))
        if len(un) != len(with_ctx):
            fails.append('%s: %d unreachable parcels have no all-voxel context'
                         % (probe, len(un) - len(with_ctx)))

    print('\n%s' % ('all checks pass' if not fails else
                    'FAILURES:\n  ' + '\n  '.join(fails)))
    return 1 if fails else 0


if __name__ == '__main__':
    sys.exit(main())
