import streamlit as st
import pandas as pd
from pathlib import Path

from modules.modelling import modelling

# =============================
# PAGE CONFIG
# =============================
st.set_page_config(
    page_title="Modeling - Linear Regression",
    layout="wide"
)

# =====================================================
# PATH
# =====================================================
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"

# =====================================================
# SIDEBAR
# =====================================================
with st.sidebar:
    st.markdown("## 🌾 CRISP-DM Framework")
    st.markdown("---")
    st.markdown("""
    - Business Understanding  
    - Data Understanding  
    - Data Preparation  
    - **Modeling**  
    - Evaluation  
    """)
    st.markdown("---")
    st.caption("Proyek Akademik Data Science\nSkala Riset")

# =====================================================
# HEADER
# =====================================================
st.markdown("""
# 🤖 Modeling  
### Analisis Produksi Pertanian di Kawasan Asia
""")

st.markdown("""
Tahap **Modeling** bertujuan untuk membangun **model regresi linear**
guna memprediksi nilai **Produksi Pertanian (Production)** berdasarkan
variabel-variabel numerik yang telah melalui proses **Data Preparation**.
""")

st.divider()

# =====================================================
# 1. PEMUATAN DATA
# =====================================================
st.subheader("1️⃣ Pemuatan Dataset")

df = pd.read_csv(DATA_DIR / "data_prep_log.csv")
st.success("Dataset hasil Data Preparation berhasil dimuat.")

# Ringkasan dataset
c1, c2, c3 = st.columns(3)
c1.metric("Jumlah Data", df.shape[0])
c2.metric("Jumlah Fitur", df.shape[1] - 1)
c3.metric("Target", "Production")

st.divider()

# =====================================================
# 2–5. PROSES MODELING
# =====================================================
st.subheader("2️⃣–5️⃣ Proses Modeling")

model, X_train, X_test, y_train, y_test = modelling(df)

# Simpan ke session state
st.session_state["model"] = model
st.session_state["X_train"] = X_train
st.session_state["X_test"] = X_test
st.session_state["y_train"] = y_train
st.session_state["y_test"] = y_test

st.success("Model regresi linear berhasil dilatih menggunakan data training.")

st.divider()

# =====================================================
# 6. STRUKTUR MODEL
# =====================================================
st.subheader("6️⃣ Struktur Model")

st.markdown("**Variabel Target (y):**")
st.code("Production")

st.markdown("**Variabel Fitur (X):**")
st.code(", ".join(X_train.columns.tolist()))

st.markdown("**Pembagian Data:**")
st.write(f"- Data Latih : {len(X_train)} baris")
st.write(f"- Data Uji   : {len(X_test)} baris")
st.write("- Rasio       : 80% Training – 20% Testing")

st.divider()

# =====================================================
# 7. DATA LATIH & DATA UJI
# =====================================================
st.subheader("7️⃣ Data Latih dan Data Uji")

st.markdown("""
Berikut adalah **contoh data** yang digunakan dalam proses pelatihan dan
pengujian model regresi linear. Ditampilkan **5 baris pertama** untuk
menjaga performa aplikasi.
""")

# ---------- DATA TRAIN ----------
st.markdown("### 📘 Data Latih (Training Set)")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**X_train (Fitur)**")
    st.dataframe(X_train.head(), use_container_width=True)

with col2:
    st.markdown("**y_train (Target)**")
    st.dataframe(y_train.head(), use_container_width=True)

# ---------- DATA TEST ----------
st.markdown("### 📙 Data Uji (Testing Set)")

col3, col4 = st.columns(2)

with col3:
    st.markdown("**X_test (Fitur)**")
    st.dataframe(X_test.head(), use_container_width=True)

with col4:
    st.markdown("**y_test (Target)**")
    st.dataframe(y_test.head(), use_container_width=True)

# =====================================================
# FOOTER
# =====================================================
st.divider()
st.caption("Modeling | Proyek Data Science Akademik")
