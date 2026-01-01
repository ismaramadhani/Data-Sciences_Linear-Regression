import joblib
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from joblib import load
from sklearn.model_selection import train_test_split

from modules.Eval import hasil_eval, visual_eval, koefisien_regresi

# =============================
# PAGE CONFIG
# =============================
st.set_page_config(
    page_title="Evaluation - Linear Regression",
    layout="wide"
)

# =====================================================
# PATH
# =====================================================
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
CURRENT_DIR = Path(__file__).resolve().parent
MODEL_PATH = CURRENT_DIR / "linear_regression_model.joblib"

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
# LOAD MODEL (JOBLIB)
# =====================================================

bundle = joblib.load(MODEL_PATH)

model = bundle["model"]
X_test = bundle["X_test"]
y_test = bundle["y_test"]
X_train = bundle["X_train"]
y_train = bundle["y_train"]
feature_names = bundle["feature_names"]

# =====================================================
# EVALUATION
# =====================================================
eval_result = hasil_eval(model, X_test, y_test)

y_pred_log = model.predict(X_test)

# Konversi ke skala asli
y_test_real = np.expm1(y_test)
y_pred_real = np.expm1(y_pred_log)

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

coef_df, intercept = koefisien_regresi(model, feature_names)

st.dataframe(coef_df, use_container_width=True)
st.markdown(f"**Intercept:** `{intercept:.4f}`")

# =====================================================
# SECTION 4: PREDIKSI HASIL PRODUKSI
# =====================================================
st.subheader("🔮 Prediksi Hasil Produksi")

st.markdown("""
Masukkan nilai variabel input **dalam skala asli**.
Sistem akan melakukan transformasi logaritmik secara otomatis.
""")

with st.form("prediction_form"):
    input_data = {}

    for feature in feature_names:
        input_data[feature] = st.number_input(
            label=f"Input {feature}",
            min_value=0.0,
            value=0.0,
            format="%.2f"
        )

    submitted = st.form_submit_button("📈 Prediksi Produksi")

if submitted:

    input_df = pd.DataFrame([input_data])
    input_df = input_df[feature_names]

    input_df_log = np.log1p(input_df)

    prediction_log = model.predict(input_df_log)[0]

    prediction_real = np.expm1(prediction_log)

    st.success("✅ Prediksi berhasil dilakukan!")
    st.metric(
        label="📊 Prediksi Produksi (Skala Asli)",
        value=f"{prediction_real:.2f}"
    )

# =====================================================
# SECTION 5: INTERPRETASI
# =====================================================
st.subheader("🧠 Interpretation")

st.markdown("""
- **MAE dan RMSE rendah** menunjukkan kesalahan prediksi relatif kecil  
- **Nilai R² mendekati 1** menandakan model mampu menjelaskan variasi produksi  
- **Residual menyebar di sekitar nol** menunjukkan model tidak bias secara sistematis  

Berdasarkan hasil evaluasi, model regresi linear layak digunakan
sebagai pendekatan awal dalam analisis produksi pertanian di Asia.
""")

# =====================================================
# FOOTER
# =====================================================
st.divider()
st.caption("Evaluation | Proyek Data Science Akademik")
