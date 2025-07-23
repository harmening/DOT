import streamlit as st
import numpy as np
import pandas as pd
import glob

st.set_page_config(page_title="🧠 DOT Image Recon Explorer", layout="wide")
st.title("🧠 DOT Image Recon Explorer")

# === Load data ===
@st.cache_data
def load_data():
    # Load all split chunk files
    chunk_files = sorted(glob.glob("data_chunk_*.npz"))
    
    all_rows = []
    meta = None

    for file in chunk_files:
        chunk = np.load(file, allow_pickle=True)["data"].item()
        all_rows.extend(chunk["rows"])
        if meta is None:
            meta = chunk["meta"]  # Assume consistent metadata

    df = pd.DataFrame(all_rows)
    return df, meta

#@st.cache_data
#def load_data():
#    data = np.load("data_dict.npy", allow_pickle=True).item()
#    df = pd.DataFrame(data["rows"])
#    return df, data["meta"]

df, meta = load_data()

# === Sidebar filters ===
st.sidebar.header("Filter Options")

selected_probe = st.sidebar.selectbox("Probe type", options=["ALL"] + meta["probes"], index=2)
selected_network = st.sidebar.selectbox("17Networks", options=["ALL"] + sorted(df["17Networks"].dropna().unique()), index=0)
selected_head_model = st.sidebar.selectbox("Head model", options=["ALL"] + meta["head_models"], index=0)
selected_svr = st.sidebar.selectbox("SVR parameter", options=["ALL"] + [str(v) for v in meta["svr_values"]], index=3)

min_depth, max_depth = st.sidebar.slider("Depth range (mm)", 0.0, 40.0, (0.0, 40.0))
min_sens, max_sens = st.sidebar.slider("Sensitivity range", 0.0, 1.0, (0.0, 1.0))

# === Filter dataframe by network ===
if selected_network != "ALL":
    filtered_df = df[df["17Networks"] == selected_network]
else:
    filtered_df = df.copy()

st.subheader("Filtered Summary Results")

# === Generate results table ===
results = []

head_models = meta["head_models"] if selected_head_model == "ALL" else [selected_head_model]
svr_values = meta["svr_values"] if selected_svr == "ALL" else [float(selected_svr)]

for model in head_models:
    for svr in svr_values:
        key = f"recon_{selected_probe}_{model}_svr{svr}"
        if key not in df.columns:
            continue
        locerrors = []
        for val_list in filtered_df[key]:
            if not isinstance(val_list, list):
                continue
            for val in val_list:
                #if not isinstance(val, (list, tuple)) or len(val) != 3:
                #    continue
                locerror, sensitivity,depth = val
                if min_depth <= depth <= max_depth and min_sens <= sensitivity <= max_sens:
                    locerrors.append(locerror)

        if locerrors:
            median = np.median(locerrors)
            std = np.std(locerrors)
            results.append({
                "Probe": selected_probe,
                "Head Model": model,
                "SVR": svr,
                "Number of placed synth. HRFs": len(locerrors),
                "Median ± Std": f"{median:.2f} ± {std:.2f} mm",
            })

# === Display summary results ===
if not results:
    st.warning("No valid data for selected combination(s).")
else:
    results_df = pd.DataFrame(results)
    st.dataframe(results_df)

