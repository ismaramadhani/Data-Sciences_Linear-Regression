import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'modules'))

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt

from modules.data_loader import load_data
from modules.preprocessing import summary_statistics, handle_missing_values
from modules.visualization import histogram, boxplot, lineplot
from modules.export_utils import download_csv, export_data_pdf, export_fig_to_pdf

from pathlib import Path

# =====================================================
# PATH
# =====================================================
BASE_DIR = Path(__file__).resolve().parents[1]

# =====================================================
# SIDEBAR (KONSISTEN)
# =====================================================
with st.sidebar:
    st.markdown("## 🌾 CRISP-DM Framework")
    st.markdown("---")
    st.markdown("### Tahapan Analisis")
    st.markdown("""
    - Business Understanding  
    - Data Understanding  
    - **Data Preparation**  
    - Modeling  
    - Evaluation  
    """)
    st.markdown("---")
    st.caption("Proyek Akademik Data Science\nSkala Riset")

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Agricultural Production Analysis",
    layout="wide"
)

# =========================
# CSS — BLUE OUTLINE CONTAINER
# =========================
st.markdown("""
<style>
.stApp {
    background-color: #0d1117;
    color: #c9d1d9;
    font-family: 'Segoe UI', sans-serif;
}
section[data-testid="stContainer"] {
    background-color: #161b22;
    border: 2px solid #58a6ff;
    border-radius: 14px;
    padding: 22px;
    margin-bottom: 28px;
    box-shadow: 0 0 12px rgba(88,166,255,0.25);
}
h1, h2, h3 {
    color: #58a6ff;
    font-weight: 700;
}
[data-testid="stDataFrame"] {
    background-color: #0d1117;
    border-radius: 10px;
}
[data-testid="metric-container"] {
    background-color: #0d1117;
    border: 1px solid #30363d;
    padding: 16px;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# =========================
# TITLE
# =========================
st.title("📊 Agricultural Production Analysis")
st.caption("Data Preparation")

# =========================
# LOAD DATA
# =========================
df = load_data()

# HAPUS KOLOM IDENTITAS (TERMASUK Country_Code)
df_cleaned = df.drop(
    columns=['Country', 'Year', 'Country_Code'],
    errors='ignore'
)

# =========================
# 1️⃣ DATA OVERVIEW
# =========================
with st.container(border=True):
    st.subheader("1️⃣ Data Overview")
    st.dataframe(df_cleaned, use_container_width=True)

# =========================
# 2️⃣ LINE PLOT OVERVIEW
# =========================
with st.container(border=True):
    st.subheader("2️⃣ Line Plot Overview")

    numeric_cols = df_cleaned.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cols = st.columns(3)

    for i, col in enumerate(numeric_cols):
        with cols[i % 3]:
            st.pyplot(lineplot(df_cleaned, col, f"Line Plot of {col}"))

# =========================
# 3️⃣ DATA CLEANING
# =========================
with st.container(border=True):
    st.subheader("3️⃣ Data Cleaning")

    missing = df_cleaned.isnull().sum()
    missing = missing[missing > 0]

    if len(missing) > 0:
        st.dataframe(missing.rename("Missing Count"))
    else:
        st.info("No missing values found.")

    df_cleaned = handle_missing_values(df_cleaned, strategy='median')
    st.success("Missing values handled successfully.")

# =========================
# 4️⃣ DUPLICATE REMOVAL
# =========================
with st.container(border=True):
    st.subheader("4️⃣ Duplicate Value Removal")

    dup_before = int(df_cleaned.duplicated().sum())
    total_before = int(len(df_cleaned))

    c1, c2, c3 = st.columns(3)
    c1.metric("Rows (Before)", total_before)
    c2.metric("Duplicates Found", dup_before)
    c3.metric("Duplicate Rate", f"{(dup_before/total_before*100):.2f}%")

    df_cleaned = df_cleaned.drop_duplicates()
    st.metric("Rows (After)", int(len(df_cleaned)))

# =========================
# 5️⃣ OUTLIER CHECK (BEFORE LOG)
# =========================
with st.container(border=True):
    st.subheader("5️⃣ Outlier Checking (Before Handling)")

    selected_cols = st.multiselect(
        "Select numeric columns:",
        df_cleaned.select_dtypes(include='number').columns.tolist(),
        default=df_cleaned.select_dtypes(include='number').columns.tolist()
    )

    grid = st.columns(2)
    for i, col in enumerate(selected_cols):
        with grid[i % 2]:
            st.pyplot(boxplot(df_cleaned, col, f"Boxplot of {col} (Before Log)"))

# =========================
# 6️⃣ LOG TRANSFORMATION
# =========================
with st.container(border=True):
    st.subheader("6️⃣ Log Transformation (np.log1p)")

    log_cols = ['Production', 'Land', 'Labor', 'N', 'P', 'K', 'Pesticides', 'fert']
    log_cols = [c for c in log_cols if c in df_cleaned.columns]

    df_log = df_cleaned.copy()
    df_log[log_cols] = np.log1p(df_log[log_cols])

    st.dataframe(df_log[log_cols].head())

# =========================
# 7️⃣ OUTLIER CHECK (AFTER LOG)
# =========================
with st.container(border=True):
    st.subheader("7️⃣ Outlier Checking (After Handling)")

    grid = st.columns(2)
    for i, col in enumerate(selected_cols):
        if col in df_log.columns:
            with grid[i % 2]:
                st.pyplot(boxplot(df_log, col, f"Boxplot of {col} (After Log)"))


# =========================
#  CORRELATION HEATMAP
# =========================
with st.container(border=True):
    st.subheader("8️⃣ Correlation Heatmap")

    st.markdown("### 🔴 Before Log Transformation")
    fig1, ax1 = plt.subplots(figsize=(12, 7))
    sns.heatmap(
        df_cleaned.select_dtypes(include='number').corr(),
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        ax=ax1
    )
    st.pyplot(fig1)

    st.markdown("### 🟢 After Log Transformation")
    fig2, ax2 = plt.subplots(figsize=(12, 7))
    sns.heatmap(
        df_log.select_dtypes(include='number').corr(),
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        ax=ax2
    )
    st.pyplot(fig2)

# =========================
# 8️⃣ PCA
# =========================
with st.container(border=True):
    st.subheader("9️⃣ PCA (Principal Component Analysis)")

    X_pca = df_log.select_dtypes(include='number').dropna()

    pca = PCA()
    pca.fit(X_pca)

    var = pca.explained_variance_ratio_
    cum = np.cumsum(var)

    pca_df = pd.DataFrame({
        'PC': [f'PC{i+1}' for i in range(len(var))],
        'Explained Variance': var,
        'Cumulative Variance': cum
    })

    st.dataframe(pca_df)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(1, len(var) + 1), var, marker='o', label='Explained Variance')
    ax.plot(range(1, len(var) + 1), cum, marker='x', label='Cumulative Variance')
    ax.legend()
    st.pyplot(fig)

# =========================
# 1️⃣0️⃣ EDA
# =========================
with st.container(border=True):
    st.subheader("1️⃣0️⃣ EDA: Summary Statistics (After Log)")
    st.dataframe(summary_statistics(df_log))

# =========================
# 📥 EXPORT
# =========================
with st.container(border=True):
    st.subheader("📥 Download / Export")

    download_csv(df_log)

    stats = summary_statistics(df_log)
    export_data_pdf(df_log, stats, filename="report_agriculture.pdf")

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"

DATA_DIR.mkdir(exist_ok=True)

df_log.to_csv(DATA_DIR / "data_prep_log.csv", index=False)

st.success("Dataset hasil Data Preparation berhasil disimpan.")

# =====================================================
# FOOTER
# =====================================================
st.divider()
st.caption("Data Preparation | Proyek Data Science Akademik")
    