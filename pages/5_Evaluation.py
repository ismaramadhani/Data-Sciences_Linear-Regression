import streamlit as st
import numpy as np

# =============================
# PAGE CONFIG (HARUS PALING ATAS)
# =============================
st.set_page_config(
    page_title="Evaluation - Linear Regression",
    layout="wide"
)

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from modules.modelling import modelling
from modules.Eval import hasil_eval, visual_eval, koefisien_regresi

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
    - Modeling  
    - **Evaluation**  
    """)
    st.markdown("---")
    st.caption("Proyek Akademik Data Science\nSkala Riset")

# =====================================================
# HEADER
# =====================================================
st.markdown("""
# 📈 Evaluation  
### Analisis Produksi Pertanian di Kawasan Asia
""")

st.markdown("""
Tahap **Evaluation** digunakan untuk menilai kinerja model regresi linear
menggunakan metrik error dan visualisasi hasil prediksi.
""")

st.divider()

# =====================================================
# LOAD MODEL & DATA FROM SESSION STATE
# =====================================================

if "model" in st.session_state:
    model = st.session_state["model"]
    X_test = st.session_state["X_test"]
    y_test = st.session_state["y_test"]
    X_train = st.session_state["X_train"]

else:
    st.error("Model belum dilatih di sesi ini. Jalankan halaman Modeling dulu.")
    st.stop()

eval_result = hasil_eval(model, X_test, y_test)

y_pred_log = model.predict(X_test)

y_test_real = np.expm1(y_test)
y_pred_real = np.expm1(y_pred_log)

# eval_result = hasil_eval(
#     y_test_real,
#     y_pred_real
# )

# =====================================================
# SECTION 1: METRIC
# =====================================================
st.subheader("📌 Evaluation Metrics")

c1, c2, c3, c4 = st.columns(4)

c1.metric("MAE", f"{eval_result['MAE']:.4f}")
c2.metric("MSE", f"{eval_result['MSE']:.4f}")
c3.metric("RMSE", f"{eval_result['RMSE']:.4f}")
c4.metric("R²", f"{eval_result['R2']:.4f}")

# =====================================================
# SECTION 2: VISUALISASI
# =====================================================
st.subheader("📊 Model Visualization")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Actual vs Predicted**")
    fig1 = visual_eval(y_test, eval_result["y_pred"])
    st.pyplot(fig1)

with col2:
    st.markdown("**Residual Plot**")
    residuals = y_test - eval_result["y_pred"]

    fig2, ax = plt.subplots()
    ax.scatter(eval_result["y_pred"], residuals)
    ax.axhline(0, linestyle="--")
    ax.set_xlabel("Predicted Value")
    ax.set_ylabel("Residual")
    ax.set_title("Residual Distribution")
    st.pyplot(fig2)

# =====================================================
# SECTION 3: KOEFISIEN REGRESI
# =====================================================
st.subheader("📐 Koefisien Regresi")

feature_names = X_train.columns.tolist()
coef_df, intercept = koefisien_regresi(model, feature_names)

st.dataframe(coef_df, use_container_width=True)
st.markdown(f"**Intercept:** `{intercept:.4f}`")

# =====================================================
# SECTION 5: PREDIKSI PRODUKSI (USER INPUT)
# =====================================================
st.subheader("🔮 Prediksi Hasil Produksi")

st.markdown("""
Masukkan nilai variabel input untuk memprediksi hasil produksi pertanian
menggunakan model regresi linear yang telah dilatih.
""")

with st.form("prediction_form"):
    input_data = {}

    for feature in feature_names:
        input_data[feature] = st.number_input(
            label=f"Input {feature}",
            value=0.0,
            format="%.2f"
        )

    submitted = st.form_submit_button("📈 Prediksi Produksi")


if submitted:
    input_df = pd.DataFrame([input_data])

    input_df = input_df[X_train.columns]

    input_df_log = np.log1p(input_df)

    prediction_log = model.predict(input_df_log)[0]

    prediction_real = np.expm1(prediction_log)

    st.success("✅ Prediksi berhasil dilakukan!")
    st.metric(
        label="📊 Prediksi Produksi (Skala Asli)",
        value=f"{prediction_real:.2f}"
    )

# =====================================================
# SECTION 4: INTERPRETASI
# =====================================================
st.subheader("🧠 Interpretation")

st.markdown("""
- **MAE dan RMSE rendah** menunjukkan kesalahan prediksi relatif kecil  
- **Nilai R² mendekati 1** menandakan model mampu menjelaskan variasi produksi pertanian  
- **Residual menyebar di sekitar nol** menunjukkan model tidak bias secara sistematis  

Berdasarkan hasil evaluasi, model regresi linear layak digunakan
sebagai pendekatan awal dalam analisis produksi pertanian di Asia.
""")

# =====================================================
# FOOTER
# =====================================================
st.divider()
st.caption("Evaluation | Proyek Data Science Akademik")
