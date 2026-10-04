"""DOT reconstruction accuracy, the online reference work for IMAG-25-0479.

Minimal v6 update of the v5 app: same two tables, plus a metric selector, a
source-sample selector, the paper's probe names and the seven head models of the paper.
Every option list comes from meta.json, nothing is hard-coded here.
"""
import os

import pandas as pd
import streamlit as st

import dotrecon_data as D

DATA = os.environ.get('DOTRECON_DATA', D.DEFAULT_DATA)

st.set_page_config(page_title='DOT reconstruction accuracy', layout='wide')
st.title('DOT reconstruction accuracy')


@st.cache_data(show_spinner=False)
def meta():
    return D.load_meta(DATA)


@st.cache_data(show_spinner=False)
def sources(probe):
    return D.load_sources(DATA, probe)


M = meta()
st.caption('Companion to the head-model comparison study. Simulated sources in '
           '%d Schaefer2018 parcels, %d subjects, %d probe densities, %d head '
           'models, %d metrics, %d source samples. Built %s.'
           % (len(M['parcels']), 15, len(M['probes']), len(M['head_models']),
              len(M['metrics']), len(M['tiers']), M['built']))

# ----------------------------------------------------------------- sidebar
sb = st.sidebar
sb.header('Selection')

plabels = [p['paper'] for p in M['probes']]
pdir = dict((p['paper'], p['dir']) for p in M['probes'])
probe_label = sb.selectbox('Probe density', plabels,
                           index=plabels.index('medium-density')
                           if 'medium-density' in plabels else 0)
probe = pdir[probe_label]

mnames = [m['name'] for m in M['metrics']]
paper_first = [m['name'] for m in M['metrics'] if m['in_paper']] + \
              [m['name'] for m in M['metrics'] if not m['in_paper']]
metric = sb.selectbox('Metric', paper_first,
                      help='The first three are the ones the paper reports.')

tier = sb.selectbox('Source sample', M['tiers'] + ['all'],
                    index=M['tiers'].index(M['defaults']['tier']),
                    help='Which voxel of each parcel carries the simulated '
                         'source: its most, median or least sensitive one.')

regs_here = D.available_regs(DATA, probe, M)
rl = dict((r['key'], D.reg_label(r)) for r in M['regularizations'])
reg_labels = [rl[k] for k in regs_here]
key_of = dict(zip(reg_labels, regs_here))
default_reg = M['defaults']['reg']
reg = key_of[sb.selectbox(
    'Regularization', reg_labels,
    index=regs_here.index(default_reg) if default_reg in regs_here else 0)]
compare_regs = [key_of[x] for x in
                sb.multiselect('Compare against',
                               [rl[k] for k in regs_here if k != reg], default=[])]

level = sb.radio('Region of interest', ['all', '7 networks', '17 networks'],
                 horizontal=True)
net = None
if level == '7 networks':
    net = sb.selectbox('Network', ['all'] + M['net7_table'])
elif level == '17 networks':
    net = sb.selectbox('Network', ['all'] + M['net17_table'])
net_level = {'all': 'all', '7 networks': '7', '17 networks': '17'}[level]

hnames = [h['name'] for h in M['head_models']]
hid = dict((h['name'], h['id']) for h in M['head_models'])
picked = sb.multiselect('Head models', hnames, default=hnames)
head_models = [hid[n] for n in picked] or [h['id'] for h in M['head_models']]

hemi = sb.selectbox('Hemisphere', ['both', 'left', 'right'])

src = sources(probe)
dmin, dmax = float(src['scalp_dist'].min()), float(src['scalp_dist'].max())
depth_range = sb.slider('Source distance from the scalp (mm)', dmin, dmax,
                        (dmin, dmax))
smin, smax = float(src['sensitivity'].min()), float(src['sensitivity'].max())
sens_range = sb.slider('Source sensitivity', smin, smax, (smin, smax),
                       format='%.4f')

how = sb.radio('Aggregation', ['across subjects', 'pooled'], horizontal=True,
               help='The paper reduces each subject to one median first, then '
                    'takes the median over the subjects. Pooling treats every '
                    'source as independent.')
how = 'across' if how.startswith('across') else 'pooled'

vg_label = sb.radio('All-voxel columns', ['common subjects', 'all available'],
                    horizontal=True)
voxel_group = 0 if vg_label == 'common subjects' else 2

keep = D.source_filter(M, src, tier=tier, net_level=net_level, net=net,
                       hemi=None if hemi == 'both' else hemi,
                       depth_range=depth_range, sens_range=sens_range)

# ------------------------------------------------------------------ table 1
st.subheader('Summary')
rows = D.summary_table(DATA, M, probe, [reg] + compare_regs, metric, keep, src,
                       head_models=head_models, how=how)
if not rows:
    st.warning('Nothing matches this selection.')
else:
    df = pd.DataFrame(rows).drop(columns=['_sort'])
    st.dataframe(df, width='stretch', hide_index=True)
    st.caption('%s, %s. Centre and spread are the median and the standard '
               'deviation %s.'
               % (probe_label, rl[reg],
                  'of the per-subject medians' if how == 'across'
                  else 'over the pooled sources'))

# ------------------------------------------------------------------ table 2
st.subheader('Per parcel')
show_unsampled = st.checkbox(
    'Include parcels with no sampled source, to see why they are unreachable',
    value=False)
prows = D.parcel_table(DATA, M, probe, reg, metric, keep, src,
                       head_models=head_models, how=how,
                       voxel_group=voxel_group,
                       include_unsampled=show_unsampled)
if not prows:
    st.info('No parcel matches this selection.')
else:
    st.dataframe(pd.DataFrame(prows), width='stretch', hide_index=True)
    st.caption(D.coverage_note(M, probe, voxel_group))
    st.download_button('Download this table as CSV',
                       pd.DataFrame(prows).to_csv(index=False).encode(),
                       file_name='dotrecon_%s_%s_%s.csv' % (probe, reg, metric),
                       mime='text/csv')

# ------------------------------------------------------------------- notes
with st.expander('What these metrics are'):
    st.markdown(
        'Peak localization error (LOCA) is the distance from the simulated '
        'source to the peak of the reconstruction. Effective resolution (ERES) '
        'is the diameter of the smallest sphere around the true source that '
        'still contains every voxel reaching half the reconstruction maximum, '
        'so it grows both when the blob broadens and when it moves. Signed '
        'radial error is measured along the cortex inward normal, so a negative '
        'value means the reconstruction was pulled towards the scalp.\n\n'
        'The all-voxel columns describe every parcel-labelled cortex voxel of '
        'the ground-truth anatomy, not just the simulated sources. They are '
        'what says whether a parcel is reachable at all.\n\n'
        '**' + M['caveats']['fig8_selection'] + '**')
