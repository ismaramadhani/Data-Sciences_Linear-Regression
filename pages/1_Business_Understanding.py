import streamlit as st
from pathlib import Path
import pandas as pd

# =====================================================
# PATH & ASSETS
# =====================================================
BASE_DIR = Path(__file__).resolve().parents[1]
ASSETS_DIR = BASE_DIR / "assets"
DATA_DIR = BASE_DIR / "data"

# =====================================================
# SIDEBAR (KONSISTEN DENGAN app.py)
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
    st.caption("Proyek Akademik Data Science\nSkala Riset")

# =====================================================
# HEADER
# =====================================================
st.markdown("""
# 🧠 Business Understanding  
### Analisis Produksi Pertanian di Kawasan Asia
""")

st.markdown("""
Tahap **Business Understanding** merupakan fondasi utama
dalam metodologi **CRISP-DM (Cross Industry Standard Process for Data Mining)**.
Pada tahap ini, fokus utama adalah memahami konteks permasalahan,
karakteristik data, tujuan analisis, serta perancangan pendekatan metodologis,
sebelum memasuki tahapan teknis seperti pengolahan data dan pemodelan statistik.
""")

st.divider()

# =====================================================
# 1. LATAR BELAKANG
# =====================================================
st.subheader("1. Latar Belakang")

st.markdown("""
Sektor pertanian memiliki peran strategis dalam pembangunan ekonomi
di kawasan Asia, baik sebagai penyedia pangan, penyerap tenaga kerja,
maupun penopang stabilitas sosial dan ekonomi.

Namun, tingkat produksi pertanian antar negara dan antar waktu
menunjukkan variasi yang cukup besar.
Variasi tersebut dipengaruhi oleh berbagai faktor input produksi,
seperti luas lahan, tenaga kerja, serta penggunaan pupuk dan pestisida.

Hubungan antara input produksi dan output pertanian
tidak selalu bersifat sederhana.
Data produksi pertanian sering mengandung nilai ekstrem (*outlier*),
korelasi tinggi antar variabel input,
serta perbedaan skala antar variabel.

Oleh karena itu, diperlukan pendekatan **data-driven**
yang sistematis untuk memahami hubungan input–output pertanian,
mengelola karakteristik data yang kompleks,
dan membangun model statistik yang dapat diinterpretasikan secara ekonomi.
""")

# =====================================================
# 2. SUMBER DATASET
# =====================================================
st.subheader("2. Sumber Dataset")

st.markdown("""
Dataset yang digunakan dalam penelitian ini berasal dari **Mendeley Data**
dengan judul:

**“Technical and Environmental Efficiency in Agriculture: A Case in Asia”**
""")

st.markdown("""
**Informasi Dataset:**
- Platform: Mendeley Data  
- Penulis: Jagath Edirisinghe  
- Tahun Publikasi: 2022  
- DOI: 10.17632/m54rkgw7f3.1  
- Lisensi: CC BY 4.0  
""")

# =====================================================
# 3. DESKRIPSI UMUM DATASET
# =====================================================
st.subheader("3. Deskripsi Umum Dataset")

df = pd.read_csv(DATA_DIR / "data_asia.csv")

st.markdown(f"""
Dataset **data_asia.csv** memiliki karakteristik sebagai berikut:

- **Jumlah observasi:** {df.shape[0]} baris  
- **Jumlah variabel:** {df.shape[1]} kolom  
- **Jenis data:** Data panel (lintas negara dan lintas waktu)  
- **Cakupan wilayah:** Negara-negara di kawasan Asia  
- **Rentang waktu:** {df['Year'].min()} – {df['Year'].max()}  
""")

st.markdown("""
Bentuk data panel memungkinkan analisis yang lebih kaya,
karena mampu menangkap variasi antar negara
serta perubahan kondisi produksi dari waktu ke waktu.
""")

# =====================================================
# 4. CAKUPAN NEGARA
# =====================================================
st.subheader("4. Cakupan Negara")

negara = sorted(df["Country"].unique())

st.markdown(f"""
Dataset ini mencakup **{len(negara)} negara** di kawasan Asia.
Daftar negara disajikan secara ringkas sebagai berikut:
""")

for i, n in enumerate(negara, start=1):
    st.markdown(f"{i}. {n}")

# =====================================================
# 5. DESKRIPSI VARIABEL DATASET (19 KOLOM)
# =====================================================
st.subheader("5. Deskripsi Variabel Dataset")

st.markdown("""
Dataset terdiri dari **19 variabel** yang dapat dikelompokkan menjadi:
- Variabel identitas
- Variabel input produksi
- Variabel output
- Variabel hasil transformasi logaritmik
""")

kolom_deskripsi = pd.DataFrame({
    "No": range(1, 20),
    "Nama Kolom": df.columns,
    "Penjelasan": [
        "Nama negara",
        "Kode negara",
        "Tahun pengamatan",
        "Total output produksi pertanian",
        "Luas lahan pertanian",
        "Jumlah tenaga kerja pertanian",
        "Penggunaan pupuk Nitrogen (N)",
        "Penggunaan pupuk Fosfor (P)",
        "Penggunaan pupuk Kalium (K)",
        "Penggunaan pestisida",
        "Total agregat pupuk",
        "Produksi pertanian (transformasi log)",
        "Luas lahan (transformasi log)",
        "Tenaga kerja (transformasi log)",
        "Nitrogen (transformasi log)",
        "Fosfor (transformasi log)",
        "Kalium (transformasi log)",
        "Pestisida (transformasi log)",
        "Agregat pupuk (transformasi log)"
    ]
})

st.table(kolom_deskripsi)

st.markdown("""
Transformasi logaritmik digunakan untuk menstabilkan varians data,
mengurangi pengaruh nilai ekstrem,
serta meningkatkan kesesuaian data
dengan asumsi model regresi linear.
""")

# =====================================================
# 6. POTENSI ANALISIS
# =====================================================
st.subheader("6. Potensi Analisis dari Dataset")

st.markdown("""
Berdasarkan karakteristik dataset,
analisis yang dapat dilakukan meliputi:
- Analisis hubungan input–output produksi pertanian
- Deteksi dan penanganan outlier
- Analisis korelasi dan multikolinearitas
- Reduksi dimensi menggunakan PCA
- Pemodelan regresi linear untuk interpretasi ekonomi
""")

# =====================================================
# 7. PERUMUSAN MASALAH
# =====================================================
st.subheader("7. Perumusan Masalah")

st.markdown("""
Permasalahan penelitian dirumuskan sebagai berikut:

1. Bagaimana hubungan antara faktor input produksi
   dan output pertanian di negara-negara Asia?
2. Bagaimana pengaruh outlier terhadap performa model regresi?
3. Metode penanganan outlier dan korelasi fitur apa
   yang menghasilkan model paling stabil dan informatif?
""")

# =====================================================
# 8. TUJUAN PENELITIAN
# =====================================================
st.subheader("8. Tujuan Penelitian")

st.markdown("""
Tujuan penelitian ini adalah:
- Memahami karakteristik data produksi pertanian Asia
- Menganalisis pengaruh input terhadap output produksi
- Membandingkan berbagai metode penanganan outlier
- Menentukan pendekatan terbaik untuk regresi linear
""")

# =====================================================
# 9. SKENARIO ANALISIS (DENGAN PENJELASAN AWAM)
# =====================================================
st.subheader("9. Skenario Analisis")

st.markdown("""
Penelitian ini menggunakan **7 skenario analisis**
untuk membandingkan berbagai pendekatan penanganan outlier
dan korelasi fitur. Penjelasan berikut disusun
agar dapat dipahami oleh pembaca non-teknis.
""")

st.markdown("""
### **Skenario 1 – Transformasi Log + PCA + Regresi Linear (Isma)**  
Pendekatan statistik klasik.  
- **Transformasi log** digunakan untuk mengecilkan pengaruh nilai ekstrem
  tanpa menghapus data.
- **PCA (Principal Component Analysis)** digunakan untuk menggabungkan
  variabel input yang saling berkorelasi menjadi komponen baru.
- **Regresi linear** dipilih karena mudah diinterpretasikan
  dalam konteks ekonomi pertanian.

---

### **Skenario 2 – DBSCAN + PCA + Regresi Linear (Daus)**  
Pendekatan berbasis klaster.  
- **DBSCAN** digunakan untuk mendeteksi data yang sangat berbeda
  dari pola umum (outlier ekstrem).
- PCA dan regresi linear digunakan setelah data ekstrem diidentifikasi.

---

### **Skenario 3 – Metode Statistik Alternatif + Regresi Linear (Rapi)**  
Pendekatan eksploratif.  
- Menggunakan teknik statistik non-klaster
  untuk membandingkan hasil dengan pendekatan lain.

---

### **Skenario 4 – Capping / Winsorizing + Regresi Linear (Dylan)**  
Pendekatan konservatif.  
- Nilai ekstrem dibatasi pada ambang tertentu
  agar tidak mendistorsi model.

---

### **Skenario 5 – Isolation Forest + Seleksi Korelasi + Regresi Linear (Andika)**  
Pendekatan machine learning.  
- **Isolation Forest** mendeteksi outlier secara non-linear.
- Fitur yang sangat berkorelasi dihapus untuk menjaga stabilitas model.

---

### **Skenario 6 – MAD (Modified Z-Score) + Seleksi Korelasi + Regresi Linear (Razan)**  
Pendekatan robust statistics.  
- **MAD** tahan terhadap distribusi data tidak normal
  dan efektif untuk data ekonomi.

---

### **Skenario 7 – IQR + VIF + Regresi Linear (Angga)**  
Pendekatan ekonometrika klasik.  
- **IQR** digunakan untuk mendeteksi outlier.
- **VIF** digunakan untuk mengukur dan mengurangi multikolinearitas.
""")

# =====================================================
# 10. JUSTIFIKASI SKENARIO UTAMA
# =====================================================
st.subheader("10. Justifikasi Pemilihan Skenario Utama")

st.markdown("""
Skenario 1 dipilih sebagai skenario utama karena:
- Tidak menghilangkan observasi data
- Stabil terhadap outlier
- Mengatasi multikolinearitas
- Mudah diinterpretasikan secara ekonomi
- Cocok untuk penelitian akademik
""")



# =====================================================
# 11. TINJAUAN PUSTAKA / REFERENSI JURNAL
# =====================================================
st.subheader("11. Tinjauan Pustaka (Referensi Jurnal)")

st.markdown("""
Penelitian ini mengacu pada literatur ilmiah
yang membahas penggunaan **transformasi logaritmik**
dan implikasinya dalam analisis data yang memiliki
distribusi tidak normal dan keberadaan outlier.

Salah satu referensi utama yang digunakan adalah:
""")

st.markdown("""
**Feng, C., Wang, H., Lu, N., Chen, T., He, H., Lu, Y., & Tu, X. M. (2014)**  
*Log-transformation and its implications for data analysis*  
Shanghai Archives of Psychiatry, Vol. 26, No. 2  
DOI: 10.3969/j.issn.1002-0829.2014.02.009
""")

st.markdown("""
Jurnal tersebut menyoroti bahwa meskipun transformasi log
sering digunakan untuk mengurangi skewness dan pengaruh outlier,
pendekatan ini **tidak selalu menjamin distribusi data menjadi normal**
dan dalam beberapa kasus justru dapat meningkatkan skewness
atau menyulitkan interpretasi hasil model.

Temuan ini menjadi dasar metodologis penting dalam penelitian ini,
khususnya dalam:
- Pemilihan transformasi log sebagai pendekatan awal
- Kehati-hatian dalam interpretasi hasil regresi
- Perbandingan dengan metode alternatif penanganan outlier
  seperti metode robust statistics dan machine learning
""")

st.markdown("""
Dengan merujuk pada literatur tersebut,
penelitian ini tidak hanya mengikuti praktik umum,
tetapi juga mempertimbangkan **keterbatasan teoretis**
dari transformasi logaritmik dalam konteks analisis ekonomi pertanian.
""")




# =====================================================
# 12. KESIMPULAN
# =====================================================
st.subheader("12. Kesimpulan Tahap Business Understanding")

st.markdown("""
Tahap Business Understanding telah berhasil:
- Mendefinisikan masalah penelitian
- Menjelaskan karakteristik dataset secara rinci
- Merancang skenario analisis yang komprehensif
- Menentukan pendekatan utama untuk tahap selanjutnya
""")

st.info(
    "Tahap selanjutnya adalah **Data Understanding**, "
    "yang akan membahas karakteristik data secara teknis."
)

st.divider()
st.caption("Business Understanding | Proyek Data Science Akademik")


