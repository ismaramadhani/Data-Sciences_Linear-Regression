import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# =====================================================
# SIDEBAR (KONSISTEN DENGAN app.py)
# =====================================================
with st.sidebar:
    st.markdown("## 🌾 CRISP-DM Framework")
    st.markdown("---")
    st.markdown("### Tahapan Analisis")
    st.markdown("""
    - Business Understanding
    - **Data Understanding**  
    - Data Preparation  
    - Modeling  
    - Evaluation  
    """)
    st.markdown("---")
    st.caption("Proyek Akademik Data Science\nSkala Riset")

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(
    page_title="Agricultural Production Analysis",
    layout="wide"
)

# -------------------------
# TITLE
# -------------------------
st.title("Agricultural Production Analysis")
st.subheader("Data Understanding")

st.caption(
    "Tahap Data Understanding bertujuan untuk memahami struktur, kualitas, "
    "serta karakteristik umum data sebelum dilakukan proses pembersihan dan pemodelan."
)

st.divider()

# -------------------------
# LOAD DATA
# -------------------------
CSV_URL = "https://docs.google.com/spreadsheets/d/14N_7TYa72pCxbJE5m3W9RwoZVGKqOP4r/export?format=csv&gid=970731575"
data = pd.read_csv(CSV_URL)

# =========================
# 1. DATA OVERVIEW
# =========================
st.header("1. Data Overview")
st.caption(
    "Bagian ini menampilkan keseluruhan dataset secara lengkap untuk memberikan "
    "gambaran menyeluruh mengenai isi, struktur, dan konsistensi data. "
    "Tabel disajikan dalam bentuk scroll agar seluruh data dapat ditelusuri "
    "tanpa membatasi jumlah baris yang ditampilkan."
)

# --- Dataset lengkap dengan scroll ---
st.subheader("Dataset Lengkap")
st.dataframe(
    data,
    use_container_width=True,
    height=350
)

# --- Ringkasan ukuran data ---
col1, col2 = st.columns(2)
with col1:
    st.metric("Jumlah Baris", data.shape[0])
with col2:
    st.metric("Jumlah Kolom", data.shape[1])

st.divider()


# -------------------------
# INTERPRETASI PER KELOMPOK KOLOM
# -------------------------
st.subheader("Interpretasi Statistik Berdasarkan Variabel")

st.markdown(
    """
    ### 🌍 Country_Code
    Variabel ini merupakan **kode numerik negara**. Statistik seperti mean dan standar deviasi
    tidak diinterpretasikan secara kuantitatif karena kolom ini hanya berfungsi sebagai
    **identifier**, bukan ukuran numerik.

    ### 📅 Year
    Data mencakup periode waktu **1990–2018**, dengan rata-rata tahun sekitar **2005**.
    Nilai standar deviasi yang relatif kecil menunjukkan bahwa data tersebar cukup merata
    sepanjang rentang waktu pengamatan.

    ### 🌾 Production
    Variabel produksi menunjukkan **variasi yang sangat tinggi**, terlihat dari perbedaan
    yang besar antara nilai minimum, median, dan maksimum.
    Hal ini menandakan adanya **ketimpangan produksi pertanian antar negara dan tahun**,
    serta potensi keberadaan outlier.

    ### 🌱 Land
    Luas lahan pertanian memiliki rentang nilai yang sangat lebar.
    Perbedaan yang signifikan antara mean dan median mengindikasikan bahwa distribusi data
    **tidak simetris** dan dipengaruhi oleh beberapa negara dengan luas lahan yang sangat besar.

    ### 👷 Labor
    Variabel tenaga kerja pertanian memiliki standar deviasi yang tinggi,
    menunjukkan **perbedaan struktur tenaga kerja antar wilayah**.
    Jumlah data valid yang sedikit lebih rendah juga mengindikasikan adanya nilai hilang.

    ### 🧪 N, P, K
    Ketiga variabel ini merepresentasikan **unsur hara pupuk**.
    Nilai minimum sebesar 0 menunjukkan bahwa tidak semua wilayah menggunakan pupuk kimia.
    Rentang nilai yang luas menandakan **perbedaan intensitas penggunaan input pertanian**.

    ### ☠️ Pesticides
    Penggunaan pestisida memiliki variasi yang besar dan nilai maksimum yang jauh dari median.
    Hal ini mengindikasikan **potensi outlier**, sehingga perlu dilakukan deteksi dan penanganan
    outlier pada tahap praproses data.

    ### 🧂 fert
    Variabel total pupuk menunjukkan distribusi yang tidak merata.
    Standar deviasi yang besar menunjukkan adanya **ketimpangan penggunaan pupuk** antar observasi.

    ### 🔄 Variabel Transformasi Logaritmik (lnprod, lnland, lnlabor, lnN, lnP, lnK, lnpest, lnfert)
    Variabel hasil transformasi logaritmik memiliki rentang nilai dan standar deviasi yang lebih kecil
    dibandingkan data aslinya. Hal ini menunjukkan bahwa transformasi logaritmik
    **berhasil menstabilkan varians dan mengurangi pengaruh outlier**,
    sehingga data menjadi lebih sesuai untuk analisis statistik dan pemodelan.
    """
)

st.divider()

# -------------------------
# STATISTIK DESKRIPTIF
# -------------------------
st.header("2. Statistik Deskriptif")

st.write(
    """
    Bagian ini menyajikan ringkasan statistik dari seluruh variabel numerik
    untuk memahami karakteristik, distribusi, dan variasi data sebelum
    dilakukan analisis lanjutan.
    """
)

desc = data.describe()
st.dataframe(desc, use_container_width=True)

st.divider()

# -------------------------
# PENJELASAN UMUM STATISTIK
# -------------------------
st.subheader("Penjelasan Umum Statistik")

st.markdown(
    """
    **Statistik deskriptif** memberikan gambaran awal mengenai pola dan sebaran data.
    Adapun arti dari masing-masing ukuran statistik adalah sebagai berikut:

    - **count** : jumlah data valid (tidak bernilai kosong)
    - **mean** : nilai rata-rata
    - **std (standard deviation)** : tingkat penyebaran data terhadap nilai rata-rata
    - **min** : nilai terkecil
    - **25% (Q1)** : kuartil bawah
    - **50% (median)** : nilai tengah
    - **75% (Q3)** : kuartil atas
    - **max** : nilai terbesar
    """
)

st.divider()

# =========================
# 3. STRUKTUR DATA
# =========================
st.header("3. Struktur Data")
st.caption(
    "Bagian ini menunjukkan struktur dataset berdasarkan nama kolom, "
    "tipe data, dan jumlah data yang tidak bernilai kosong."
)

info_df = pd.DataFrame({
    "Nama Kolom": data.columns,
    "Tipe Data": data.dtypes.astype(str),
    "Jumlah Data Non-Null": data.notnull().sum().values
})

st.dataframe(info_df, use_container_width=True)

st.divider()

# =========================
# 4. MISSING VALUES
# =========================
st.header("4. Missing Values")
st.caption(
    "Bagian ini digunakan untuk mengidentifikasi jumlah nilai yang hilang "
    "pada setiap kolom sebagai indikator kualitas data."
)

missing_df = pd.DataFrame({
    "Nama Kolom": data.columns,
    "Jumlah Missing": data.isnull().sum().values
})

st.dataframe(missing_df, use_container_width=True)

st.divider()

# =========================
# 5. DUPLICATE DATA
# =========================
st.header("5. Duplikasi Data")
st.caption(
    "Bagian ini bertujuan untuk mendeteksi adanya data duplikat "
    "yang berpotensi memengaruhi hasil analisis."
)

duplicate_count = data.duplicated().sum()
st.write(f"Jumlah baris duplikat: **{duplicate_count}**")

st.divider()

# =========================
# 6. OUTLIER DETECTION
# =========================
st.header("6. Deteksi Outlier (Boxplot)")
st.caption(
    "Bagian ini digunakan untuk melihat distribusi data numerik "
    "dan mengidentifikasi kemungkinan adanya outlier."
)
st.markdown(
    """
    **Catatan:**  
    Pada visualisasi boxplot, titik berbentuk bulat merepresentasikan *outlier*, 
    sedangkan garis pembatas menunjukkan nilai minimum dan maksimum data.
    """
)

exclude_cols = ["Country_Code", "Year"]
numeric_cols = [
    col for col in data.select_dtypes(include=np.number).columns
    if col not in exclude_cols
]

selected_col = st.selectbox("Pilih kolom numerik:", numeric_cols)

fig, ax = plt.subplots(figsize=(6, 2))
sns.boxplot(x=data[selected_col], ax=ax)
ax.set_title(f"Boxplot {selected_col}")

st.pyplot(fig)

st.divider()

# =========================
# 7. ANALISIS TREN
# =========================
st.header("7. Analisis Tren")
st.caption(
    "Bagian ini menampilkan pola atau tren data numerik "
    "berdasarkan urutan pengamatan."
)

trend_col = st.selectbox(
    "Pilih kolom untuk visualisasi tren:",
    numeric_cols
)

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(data[trend_col])
ax.set_title(f"Tren {trend_col}")
ax.set_xlabel("Index")
ax.set_ylabel(trend_col)

st.pyplot(fig)

# -------------------------
# DESKRIPSI INTERPRETASI TREN
# -------------------------
st.markdown(
    """
    **Interpretasi Umum Tren:**

    Visualisasi tren menunjukkan bahwa nilai variabel yang diamati mengalami
    fluktuasi yang cukup signifikan sepanjang urutan data.
    Terlihat adanya beberapa lonjakan nilai yang tinggi pada titik-titik tertentu,
    sementara sebagian besar observasi berada pada tingkat yang relatif lebih rendah.

    Pola ini mengindikasikan adanya **ketimpangan antar observasi**, yang dapat
    disebabkan oleh perbedaan karakteristik wilayah, waktu, atau intensitas input
    pertanian. Keberadaan lonjakan ekstrem juga memperkuat indikasi adanya
    **outlier**, sehingga analisis lanjutan dan transformasi data menjadi relevan
    pada tahap selanjutnya.
    """
)

st.divider()

# =========================
# 8. DATA YANG DIGUNAKAN
# =========================
st.header("8. Data yang Digunakan")
st.caption(
    "Bagian ini menampilkan dataset akhir yang digunakan dalam analisis lanjutan. "
    "Dataset ini merupakan data asli tanpa menyertakan variabel hasil transformasi "
    "logaritmik (ln). Transformasi log tetap tersedia pada data sumber, namun "
    "tidak digunakan pada tahap analisis ini."
)

# Daftar kolom logaritmik yang dihapus
ln_cols = [
    "lnprod", "lnland", "lnlabor",
    "lnN", "lnP", "lnK",
    "lnpest", "lnfert"
]

# Dataset tanpa kolom ln
data_used = data.drop(columns=ln_cols, errors="ignore")

# Tampilkan dataset dengan scroll
st.subheader("Dataset Tanpa Variabel Transformasi Logaritmik")
st.dataframe(
    data_used,
    use_container_width=True,
    height=350
)

# Ringkasan dataset yang digunakan
col1, col2 = st.columns(2)
with col1:
    st.metric("Jumlah Baris", data_used.shape[0])
with col2:
    st.metric("Jumlah Kolom", data_used.shape[1])

st.markdown(
    """
    **Penjelasan:**

    Dataset yang digunakan pada tahap ini hanya mencakup **variabel asli**,
    seperti produksi, luas lahan, tenaga kerja, penggunaan pupuk, dan pestisida.
    Variabel transformasi logaritmik (**ln\***) **tidak disertakan dalam tabel ini**
    meskipun tersedia pada data sumber.

    Penghapusan kolom ln dilakukan untuk:
    1. Menjaga **keterbacaan dan interpretasi data asli**
    2. Memisahkan secara jelas antara **data mentah** dan **data hasil transformasi**
    3. Memastikan konsistensi dataset sebelum tahap praproses dan pemodelan
    """
)

st.divider()

# -------------------------
# KESIMPULAN
# -------------------------
st.subheader("Kesimpulan Tahap Data Understanding")

st.markdown(
    """
    Berdasarkan hasil statistik deskriptif, dapat disimpulkan bahwa:
    1. Dataset memiliki **variasi tinggi** pada variabel produksi dan input pertanian.
    2. Terdapat indikasi kuat keberadaan **outlier**.
    3. Distribusi data cenderung **tidak simetris**, sehingga transformasi logaritmik relevan.
    4. Dataset telah memenuhi kebutuhan tahap **Data Understanding dalam CRISP-DM**
       dan siap dilanjutkan ke tahap praproses dan pemodelan.
    """
)

# -------------------------
# FOOTER
# -------------------------
st.caption("Tahap Data Understanding — Metodologi CRISP-DM")