import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import ast

# Load Data
@st.cache_data
def load_data():
    # Replace with actual URL or local path
    df = pd.read_csv("datalist.tsv", sep='\t')

    # Convert stringified tuples like "(13.3, 9.1)" into actual tuples
    for col in df.columns:
        if df[col].astype(str).str.startswith("(").any():
            df[col] = df[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith("(") else (np.nan, np.nan))
    
    return df

df = load_data()

st.title("🧠 Brain Parcel Explorer")

# Filter by parcel name
parcels = df["parcel"].unique()
selected_parcels = st.multiselect("Select parcels", parcels, default=parcels[:5])

filtered_df = df[df["parcel"].isin(selected_parcels)]

# Column selection
numeric_columns = [col for col in df.columns if df[col].dtype == object or isinstance(df[col].iloc[0], tuple)]
selected_metric = st.selectbox("Select metric column", numeric_columns)

# Extract tuple components for plotting
def extract_component(series, index):
    return series.apply(lambda x: x[index] if isinstance(x, tuple) else np.nan)

component = st.radio("Component to visualize", ["First", "Second"])
index = 0 if component == "First" else 1

filtered_df["metric_value"] = extract_component(filtered_df[selected_metric], index)

# Table view
st.subheader("📊 Filtered Data")
st.dataframe(filtered_df[["parcel", "metric_value"]])

# Plotting
st.subheader("📈 Plot")
fig = px.bar(filtered_df, x="parcel", y="metric_value", title=f"{selected_metric} ({component})")
fig.update_layout(xaxis_tickangle=-45)
st.plotly_chart(fig, use_container_width=True)
