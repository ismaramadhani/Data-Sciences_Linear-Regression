import streamlit as st
import pandas as pd
from pathlib import Path

from modules.modelling import modelling

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
    st.markdown("### Tahapan Analisis")
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
Tahap **Modeling** bertujuan untuk membangun model statistik
berdasarkan data yang telah dipersiapkan.
""")

st.divider()

# =====================================================
# 1. PEMUATAN DATA
# =====================================================
st.subheader("1. Pemuatan Data")

df_log = pd.read_csv(DATA_DIR / "data_prep_log.csv")
st.success("Dataset hasil Data Preparation berhasil dimuat.")

# =====================================================
# 2–5. PEMODELAN (via function)
# =====================================================
st.subheader("2–5. Proses Modeling")

model, X_train, X_test, y_train, y_test = modelling(df_log)

st.session_state["model"] = model
st.session_state["X_test"] = X_test
st.session_state["y_test"] = y_test
st.session_state["X_train"] = X_train

st.success("Model regresi linear berhasil dilatih.")

# =====================================================
# 6. TAMPILAN DATA LATIH
# =====================================================
st.subheader("6. Contoh Data Latih (X_train)")

st.markdown("""
Tabel berikut menampilkan **5 baris pertama data latih (X_train)**.
""")

st.dataframe(X_train.head(5), use_container_width=True)

# =====================================================
# FOOTER
# =====================================================
st.divider()
st.caption("Modeling | Proyek Data Science Akademik")


