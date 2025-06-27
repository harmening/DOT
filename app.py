import streamlit as st
import pandas as pd
import numpy as np
import ast

st.title("🧠 DOT Image Recon Explorer")

def format_stat(median, std):
    return f"{median:.2f} ± {std:.2f} mm"

@st.cache_data
def load_data():
    df = pd.read_csv("datalist.tsv", sep="\t")

    def safe_parse(x):
        try:
            if isinstance(x, str) and x.startswith("("):
                return ast.literal_eval(x)
        except Exception:
            pass
        return (np.nan, np.nan)

    for col in df.columns:
        if col.endswith("_median_std"):
            df[col] = df[col].apply(safe_parse)
    return df

df = load_data()

HEAD_MODEL_MAP = {
    "original": "recon_original",
    "TwoSurfaceHeadmodel (TSHM)": "recon_TSHM",
    "sBrain": "recon_sBrain",
    "PCAwarp": "recon_PCAwarp",
    "wMNI-FEM": "recon_wMNi-FEM",
    "wMNI-4": "recon_wMNI-4",
    "ICBM-152": "recon_ICBM-152",
    "Colin27": "recon_Colin27",
}

# === Sidebar filters ===
st.sidebar.header("Filter Options")

probes = ["sparse", "HD", "UHD"]
selected_probe = st.sidebar.selectbox("Probe type", options=probes, index=1)

networks = ["ALL"] + sorted(df["17Networks"].dropna().unique())
selected_network = st.sidebar.selectbox("17Networks", options=networks, index=0)

head_models = ["ALL"] + list(HEAD_MODEL_MAP.keys())
selected_head_model = st.sidebar.selectbox("Head model", options=head_models, index=0)

svr_values = ["ALL", 0.001, 0.01, 0.1]
selected_svr = st.sidebar.selectbox("SVR parameter", options=svr_values, index=3)

# === Filter rows ===
if selected_network != "ALL":
    filtered_df = df[df["17Networks"] == selected_network]
else:
    filtered_df = df.copy()

st.subheader("Filter results by choosing your setup on the left")

def format_stat(median, std):
    return f"{median:.2f} ± {std:.2f} mm"

# === Handle ALL combinations ===
results = []

if selected_head_model == "ALL":
    head_model_keys = list(HEAD_MODEL_MAP.keys())
else:
    head_model_keys = [selected_head_model]

if selected_svr == "ALL":
    svr_keys = [0.001, 0.01, 0.1]
else:
    svr_keys = [selected_svr]

for model in head_model_keys:
    for svr in svr_keys:
        prefix = HEAD_MODEL_MAP[model]
        colname = f"{prefix}_svr{svr}_{selected_probe}_median_std"

        if colname not in df.columns:
            continue

        valid_values = filtered_df[colname].dropna()
        medians = [v[0] for v in valid_values if isinstance(v, tuple) and not np.isnan(v[0])]
        stds = [v[1] for v in valid_values if isinstance(v, tuple) and not np.isnan(v[1])]

        if len(medians) == 0 or len(stds) == 0:
            continue

        avg_median = np.nanmedian(medians)
        avg_std = np.nanstd(stds)

        results.append({
            "Head Model": model,
            "SVR": svr,
            "Probe": selected_probe,
            "Median ± Std": format_stat(avg_median, avg_std),
            "Column": colname
        })

# === Display results ===
if len(results) == 0:
    st.warning("No valid data for selected combination(s).")
else:
    results_df = pd.DataFrame(results)
    st.dataframe(results_df[["Head Model", "SVR", "Median ± Std"]])
    
    with st.expander("Show matching parcels"):
        display_df = filtered_df[["parcel", "num_vox"]].copy()

        if selected_head_model == "ALL" or selected_svr == "ALL":
            # Multiple columns case
            columns_to_display = []
            for head_model in [selected_head_model] if selected_head_model != "ALL" else [hm for hm in head_models if hm != "ALL"]:
                for svr in [selected_svr] if selected_svr != "ALL" else svr_values:
                    prefix = HEAD_MODEL_MAP[head_model]
                    col = f"{prefix}_svr{svr}_{selected_probe}_median_std"
                    if col in filtered_df.columns:
                        label = f"{head_model} / SVR {svr}"
                        display_df[label] = filtered_df[col].apply(
                            lambda val: format_stat(val[0], val[1]) if isinstance(val, tuple) and not np.isnan(val[0]) else "N/A"
                        )
                        columns_to_display.append(label)
            st.dataframe(display_df[["parcel", "num_vox"] + columns_to_display])

        else:
            # Single column case
            col = f"{HEAD_MODEL_MAP[selected_head_model]}_svr{selected_svr}_{selected_probe}_median_std"
            display_df["Median ± Std"] = filtered_df[col].apply(
                lambda val: format_stat(val[0], val[1]) if isinstance(val, tuple) and not np.isnan(val[0]) else "N/A"
            )
            st.dataframe(display_df[["parcel", "num_vox", "Median ± Std"]])
