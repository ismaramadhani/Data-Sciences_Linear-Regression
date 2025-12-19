import streamlit as st
from pathlib import Path

# =====================================================
# KONFIGURASI APLIKASI
# =====================================================
st.set_page_config(
    page_title="CRISP-DM | Analisis Produksi Pertanian Asia",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# PATH
# =====================================================
BASE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = BASE_DIR / "assets"

# =====================================================
# LOAD CSS
# =====================================================
css_file = ASSETS_DIR / "style.css"
if css_file.exists():
    with open(css_file, encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# =====================================================
# SIDEBAR (GLOBAL)
# =====================================================
with st.sidebar:
    st.markdown("## 🌾 CRISP-DM Framework")
    st.markdown("---")

    st.markdown("### Tahapan Analisis")
    st.markdown("""
    - **Business Understanding**
    - Data Understanding  
    - Data Preparation  
    - Modeling  
    - Evaluation  
    """)

    st.markdown("---")
    st.caption("Proyek Data Science Akademik\nSkala Riset")

# =====================================================
# HERO SECTION
# =====================================================
col1, col2 = st.columns([1.4, 1])

with col1:
    st.markdown("""
    # Analisis Produksi Pertanian di Asia  
    ### Pendekatan Data Science Berbasis CRISP-DM

    Aplikasi ini merupakan **dashboard penelitian data science**
    yang dikembangkan menggunakan framework **CRISP-DM**
    untuk menganalisis hubungan antara input dan output
    produksi pertanian di kawasan Asia.
    """)

    st.markdown("""
    Aplikasi ini berfungsi sebagai **wadah integrasi**
    dari seluruh tahapan analisis,
    mulai dari pemahaman masalah hingga evaluasi model.
    """)

with col2:
    hero_img = ASSETS_DIR / "hero_agriculture.jpg"
    if hero_img.exists():
        st.image(hero_img, use_container_width=True)

st.divider()

# =====================================================
# FRAMEWORK CRISP-DM (OVERVIEW)
# =====================================================
st.subheader("Kerangka Kerja CRISP-DM")

crisp_img = ASSETS_DIR / "crispdm_flow.png"
if crisp_img.exists():
    st.image(
        crisp_img,
        caption="Framework CRISP-DM sebagai alur analisis data",
        use_container_width=True
    )

st.markdown("""
CRISP-DM digunakan sebagai kerangka kerja utama
karena menyediakan alur analisis yang sistematis,
fleksibel, dan banyak digunakan dalam penelitian akademik
maupun industri.
""")

# =====================================================
# DATASET OVERVIEW (RINGKAS)
# =====================================================
st.subheader("Gambaran Umum Dataset")

st.markdown("""
Dataset yang digunakan bersumber dari **Mendeley Data**
dan mencakup data produksi pertanian dari beberapa negara di Asia.

Dataset ini terdiri dari variabel numerik yang merepresentasikan
input produksi pertanian dan satu variabel output utama
berupa produksi pertanian.
""")

# =====================================================
# SKENARIO PENELITIAN (RINGKAS)
# =====================================================
st.subheader("Ringkasan Skenario Penelitian")

st.markdown("""
Penelitian ini dirancang menggunakan **7 skenario analisis**
yang membandingkan berbagai metode penanganan outlier
dan korelasi fitur dalam pemodelan regresi linear.
""")

st.markdown("""
⭐ **Skenario Utama (Terpilih):**
- Transformasi logaritmik untuk handling outlier
- PCA untuk menangani korelasi fitur
- Regresi linear sebagai model utama

Detail lengkap setiap skenario
dijelaskan pada tahap **Business Understanding**.
""")

# =====================================================
# ARAHAN PENGGUNA
# =====================================================
st.info(
    "Silakan masuk ke menu **Business Understanding** "
    "untuk memahami permasalahan, tujuan, dan justifikasi metode penelitian."
)

# =====================================================
# FOOTER
# =====================================================
st.divider()
st.caption(
    "Dashboard CRISP-DM | Proyek Data Science Akademik"
)
