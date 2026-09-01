# DOT reconstruction accuracy, the online reference work

Companion app for the head-model comparison study (IMAG-25-0479, v6). The paper
points readers here from sections 3.1, 3.2, 3.3 and 4.

## Layout

    app.py                 the streamlit UI, no data logic
    dotrecon_data.py       loading, fetching, filtering and the two tables
    smoke_test.py          exercises the data layer, checks the default view
    requirements.txt
    .gitignore

`app.py` holds no probe name, head model, metric, tier or regularization level.
Every option list comes out of `data/meta.json`, so adding a metric or a
regularization to the bundle adds it to the app with no code change.

## The data bundle

Built by `outputs/streamlit_v6/build_streamlit_data.py`. 37 files, 238 MB.

| file | what it is |
|---|---|
| `meta.json` | every option list, the coverage table, the caveats |
| `sources_<probe>.npz` | one row per simulated source, label arrays only |
| `metrics_<probe>_<reg>.npz` | `(n_sources, 7, 9)` float32, rows aligned to sources |
| `voxelstats_<probe>.npz` | per-parcel sensitivity and scalp distance for all 1000 parcels |

The bundle is **not** in this repository. It ships as release assets, and the
app fetches one shard at a time, so a cold start pulls `meta.json` plus one
sources file and one metrics file, about 8 MB, rather than the whole 238 MB.

Three environment variables decide where it comes from:

| variable | meaning |
|---|---|
| `DOTRECON_DATA` | a local directory. A file found here is used as-is. |
| `DOTRECON_URL` | base URL of the release assets. Anything missing locally is fetched once from here. |
| `DOTRECON_CACHE` | where fetched files are kept. Defaults to a temp directory. |

Deployed, only `DOTRECON_URL` is set. In development, point `DOTRECON_DATA` at
the built bundle and the app never touches the network.

## Publishing a rebuilt bundle

1. `python3 build_streamlit_data.py --phase all` in `outputs/streamlit_v6/`
2. `python3 verify_streamlit_data.py --data data`
3. `python3 app/smoke_test.py --data ../data`
4. Cut a release and upload all 37 files of `data/` as assets
5. Point `DOTRECON_URL` at that release tag

`meta.json` is part of the release, not of the repository, so the app and the
data cannot drift apart: whichever release the URL names supplies both the
numbers and the option lists.

## Running it

    pip install -r requirements.txt
    DOTRECON_DATA=../data streamlit run app.py      # local bundle
    streamlit run app.py                            # from the release assets

Before shipping a rebuilt bundle:

    python3 smoke_test.py --data ../data
    python3 ../verify_streamlit_data.py --data ../data

The first checks the app can build every table from every option in meta. The
second reproduces the paper's published numbers from the bundle alone.

## What the default view shows

The paper's own selection: median-sensitivity tier, λ₁ = 0.01, λ₂ = 0.1, with
measurement noise, medians taken per subject first and then across the 15. On
that selection the original head model reads 11.49 / 9.17 / 8.19 mm median peak
error at the sparse, medium- and high-density probe, against 11.5 / 9.2 / 8.2 in
the paper.

## Two things to keep honest

**Fig 8 style comparisons.** The best head model and the best λ₂ are chosen on
the same data that then supplies the significance test. Any p-value shown for
that comparison is descriptive, not a test of a pre-specified hypothesis, and it
is not corrected for the selection. The app says so wherever it shows one.

**All-voxel coverage.** The per-parcel sensitivity and scalp distance come from
every cortex voxel of the ground-truth anatomy. Eleven subjects have the full
range at every probe. Subjects 5 and 6, and subject 3 at medium density,
contribute no voxels below 1e-4, and subject 9 has no all-voxel data at high
density. The all-voxel columns default to the eleven and say which subjects are
missing when you switch to all available. The per-source columns always use all
fifteen.
