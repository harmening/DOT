"""Data layer for the DOT reference work. No streamlit, so it can be tested.

Everything the app offers comes out of meta.json. Nothing here hard-codes a
probe name, a head model, a metric, a source sample or a regularization level.
"""
import json
import os
import shutil
import tempfile
import urllib.error
import urllib.request

import numpy as np

# Where the bundle lives.
#
# DOTRECON_DATA   a local directory. Used as-is when it holds the file.
# DOTRECON_URL    base URL of the release assets. Files missing locally are
#                 fetched from here once and cached under DOTRECON_CACHE.
#
# Deployed, only DOTRECON_URL is set and the repository carries no data. In
# development, point DOTRECON_DATA at the built bundle and leave the URL unset.
DEFAULT_DATA = os.environ.get(
    'DOTRECON_DATA', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data'))
_URL = os.environ.get(
    'DOTRECON_URL',
    'https://github.com/harmening/DOT/releases/download/v6.0/').strip()
# An explicitly empty DOTRECON_URL means local files only, so fetching is off
# rather than pointed at '/'.
DATA_URL = (_URL.rstrip('/') + '/') if _URL else ''
CACHE_DIR = os.environ.get(
    'DOTRECON_CACHE', os.path.join(tempfile.gettempdir(), 'dotrecon-cache'))


class MissingAsset(Exception):
    pass


def asset(data_dir, name, required=True):
    """Local path for one bundle file, fetching it once if it is not there.

    Returns None for an optional file that exists in neither place, which is how
    a regularization absent at one probe drops out of the app's option list.
    """
    local = os.path.join(data_dir, name)
    if os.path.isfile(local):
        return local
    cached = os.path.join(CACHE_DIR, name)
    if os.path.isfile(cached):
        return cached
    if not DATA_URL:
        if required:
            raise MissingAsset('%s not found in %s and no DOTRECON_URL set'
                               % (name, data_dir))
        return None
    os.makedirs(CACHE_DIR, exist_ok=True)
    tmp = cached + '.part'
    try:
        with urllib.request.urlopen(DATA_URL + name, timeout=120) as r, \
                open(tmp, 'wb') as f:
            shutil.copyfileobj(r, f)
    except urllib.error.HTTPError as e:
        if os.path.exists(tmp):
            os.remove(tmp)
        if e.code == 404 and not required:
            return None
        raise MissingAsset('%s: HTTP %s from %s' % (name, e.code, DATA_URL))
    except Exception as e:
        if os.path.exists(tmp):
            os.remove(tmp)
        if required:
            raise MissingAsset('%s: %s' % (name, e))
        return None
    os.replace(tmp, cached)
    return cached


# ------------------------------------------------------------------ loading
PAPER_NAMES = {'ICBM-152 scaled': 'ICBM-152'}


def load_meta(data_dir=DEFAULT_DATA):
    with open(asset(data_dir, 'meta.json')) as f:
        meta = json.load(f)
    for h in meta['head_models']:
        h['name'] = PAPER_NAMES.get(h['name'], h['name'])
    return meta


def load_sources(data_dir, probe):
    z = np.load(asset(data_dir, 'sources_%s.npz' % probe))
    return dict((k, z[k]) for k in z.files)


def load_metrics(data_dir, probe, reg):
    """(n_sources, n_head_models, n_metrics) plus the row mask into sources."""
    path = asset(data_dir, 'metrics_%s_%s.npz' % (probe, reg), required=False)
    if path is None:
        return None, None
    z = np.load(path)
    mask = z['row_mask']
    return z['metrics'], (None if mask.size == 0 else mask)


def load_voxelstats(data_dir, probe):
    z = np.load(asset(data_dir, 'voxelstats_%s.npz' % probe))
    return dict((k, z[k]) for k in z.files)


def available_regs(data_dir, probe, meta):
    """Which regularizations this probe actually has.

    Local runs test the filesystem. Deployed, that would mean a network round
    trip per option on every rerun, so the answer comes from meta, which records
    what the build produced.
    """
    keys = [r['key'] for r in meta['regularizations']]
    if os.path.isdir(data_dir) and os.path.isfile(
            os.path.join(data_dir, 'meta.json')):
        return [k for k in keys
                if os.path.isfile(os.path.join(
                    data_dir, 'metrics_%s_%s.npz' % (probe, k)))]
    built = meta.get('built_shards', {}).get(probe)
    return list(built) if built else keys


# ----------------------------------------------------------------- helpers
def reg_label(r):
    """'lambda2 = 0.1, with noise' rather than a filename."""
    parts = []
    if r['lambda2'] is None:
        parts.append('no SVR')
    else:
        parts.append('λ₂ = %g' % r['lambda2'])
    parts.append('λ₁ = %g' % r['lambda1'])
    parts.append('with noise' if r['noise'] else 'noise-free')
    return ', '.join(parts)


def probe_labels(meta):
    return dict((p['paper'], p['dir']) for p in meta['probes'])


def source_filter(meta, src, tier=None, net_level='all', net=None, hemi=None,
                  depth_range=None, sens_range=None, subjects=None,
                  require_radial_ok=True):
    """Boolean mask over the rows of sources_<probe>.npz."""
    keep = np.ones(len(src['subject']), bool)
    if require_radial_ok:
        keep &= src['radial_ok']
    if tier not in (None, 'all'):
        keep &= src['tier_code'] == meta['tiers'].index(tier)
    if net_level == '7' and net not in (None, 'all'):
        keep &= src['net7_code'] == meta['net7_table'].index(net)
    elif net_level == '17' and net not in (None, 'all'):
        keep &= src['net17_code'] == meta['net17_table'].index(net)
    if hemi in ('left', 'right'):
        keep &= src['hemi_code'] == (0 if hemi == 'left' else 1)
    if depth_range is not None:
        keep &= (src['scalp_dist'] >= depth_range[0]) & \
                (src['scalp_dist'] <= depth_range[1])
    if sens_range is not None:
        keep &= (src['sensitivity'] >= sens_range[0]) & \
                (src['sensitivity'] <= sens_range[1])
    if subjects:
        keep &= np.isin(src['subject'], list(subjects))
    return keep


def _aggregate(values, subject, keep, how):
    """values is (n_rows,). Returns (centre, spread, n_rows, n_subjects)."""
    v = values[keep]
    s = subject[keep]
    n_sub = len(np.unique(s))
    if v.size == 0:
        return np.nan, np.nan, 0, 0
    if how == 'pooled':
        return float(np.nanmedian(v)), float(np.nanstd(v)), int(v.size), n_sub
    per = []
    for u in np.unique(s):
        w = v[s == u]
        if w.size and np.isfinite(w).any():
            per.append(np.nanmedian(w))
    if not per:
        return np.nan, np.nan, int(v.size), n_sub
    per = np.array(per)
    return float(np.median(per)), float(np.std(per)), int(v.size), len(per)


# ------------------------------------------------------------------ table 1
def summary_table(data_dir, meta, probe, regs, metric_name, keep_src, src,
                  head_models=None, how='across'):
    """One row per head model and regularization, as the v5 app had."""
    mi = [m['name'] for m in meta['metrics']].index(metric_name)
    unit = meta['metrics'][mi]['unit']
    hms = [h['id'] for h in meta['head_models']]
    sel = head_models or hms
    reg_by_key = dict((r['key'], r) for r in meta['regularizations'])
    paper = dict((p['dir'], p['paper']) for p in meta['probes'])
    rows = []
    for key in regs:
        M, mask = load_metrics(data_dir, probe, key)
        if M is None:
            continue
        keep = keep_src if mask is None else keep_src[mask]
        subj = src['subject'] if mask is None else src['subject'][mask]
        for h in sel:
            j = hms.index(h)
            c, sd, n, ns = _aggregate(M[:, j, mi], subj, keep, how)
            rows.append({
                'Probe': paper[probe],
                'Head model': dict((x['id'], x['name'])
                                   for x in meta['head_models'])[h],
                'Group': dict((x['id'], x['group'])
                              for x in meta['head_models'])[h],
                'Regularization': reg_label(reg_by_key[key]),
                'Sources': n,
                'Subjects': ns,
                '%s (%s)' % (metric_name, unit):
                    'n/a' if not np.isfinite(c) else '%.2f ± %.2f' % (c, sd),
                '_sort': c,
            })
    return rows


# ------------------------------------------------------------------ table 2
def parcel_table(data_dir, meta, probe, reg, metric_name, keep_src, src,
                 head_models=None, how='across', voxel_group=2,
                 include_unsampled=False, max_rows=None):
    """One row per parcel: the all-voxel context, then one column per model."""
    mi = [m['name'] for m in meta['metrics']].index(metric_name)
    hms = [h['id'] for h in meta['head_models']]
    hname = dict((h['id'], h['name']) for h in meta['head_models'])
    sel = head_models or hms
    parcels = meta['parcels']
    n7 = meta['net7_table']
    n17 = meta['net17_table']
    dec = meta['histogram']['deciles']
    mid = dec.index(50)

    vs = load_voxelstats(data_dir, probe)
    q_sens, q_depth = vs['q_sens'], vs['q_depth']
    q_n, q_frac = vs['q_n'], vs['q_frac']

    M, mask = load_metrics(data_dir, probe, reg)
    keep = keep_src if mask is None else keep_src[mask]
    subj = src['subject'] if mask is None else src['subject'][mask]
    pcode = src['parcel_code'] if mask is None else src['parcel_code'][mask]
    n7code = src['net7_code'] if mask is None else src['net7_code'][mask]
    n17code = src['net17_code'] if mask is None else src['net17_code'][mask]

    present = np.unique(pcode[keep]) if keep.any() else np.array([], np.uint16)
    codes = np.arange(1000) if include_unsampled else present
    unsampled = set(range(1000)) - set(present.tolist())

    rows = []
    for c in codes:
        sub = keep & (pcode == c)
        row = {
            'Parcel': parcels[c],
            '7 networks': n7[int(n7code[sub][0])] if sub.any() else '',
            '17 networks': n17[int(n17code[sub][0])] if sub.any() else '',
            'Sampled sources': int(sub.sum()),
            'Cortex voxels': int(q_n[c, voxel_group]),
            'Median sensitivity': ('n/a' if not np.isfinite(q_sens[c, voxel_group, mid])
                                   else '10^%.2f' % q_sens[c, voxel_group, mid]),
            'Voxels above 1e-3': ('n/a' if not np.isfinite(q_frac[c, voxel_group])
                                  else '%.0f %%' % (100 * q_frac[c, voxel_group])),
            'Median scalp distance (mm)':
                ('n/a' if not np.isfinite(q_depth[c, voxel_group, mid])
                 else '%.1f' % q_depth[c, voxel_group, mid]),
            'Reachable': 'no sampled source' if int(c) in unsampled else 'yes',
        }
        if not sub.any():
            for h in sel:
                row[hname[h]] = 'n/a'
        else:
            for h in sel:
                j = hms.index(h)
                val, sd, _n, _ns = _aggregate(M[:, j, mi], subj, sub, how)
                row[hname[h]] = ('n/a' if not np.isfinite(val)
                                 else '%.2f ± %.2f' % (val, sd))
        rows.append(row)
    rows.sort(key=lambda r: r['Parcel'])
    return rows[:max_rows] if max_rows else rows


# ------------------------------------------------- all-voxel context blurb
def coverage_note(meta, probe, voxel_group):
    av = meta['allvoxel']
    floored = av['floored_1e-4'].get(probe, [])
    absent = av['absent'].get(probe, [])
    if voxel_group == 0:
        return ('All-voxel columns use the %d subjects that are full-head at every '
                'probe: %s.' % (len(av['common_subjects']),
                                ', '.join(str(x) for x in av['common_subjects'])))
    bits = ['All-voxel columns use every subject with data at this probe.']
    if floored:
        bits.append('Subject%s %s contribute no voxels below 1e-4 here, so a parcel '
                    'whose median sits under that floor is drawn from the rest.'
                    % ('s' if len(floored) > 1 else '',
                       ', '.join(str(x) for x in floored)))
    if absent:
        bits.append('Subject%s %s ha%s no all-voxel data at this probe at all.'
                    % ('s' if len(absent) > 1 else '',
                       ', '.join(str(x) for x in absent),
                       've' if len(absent) > 1 else 's'))
    return ' '.join(bits)
