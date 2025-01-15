import streamlit as st
import pickle
import pandas as pd

with open('graduation_prediction_model.pkl', 'rb') as file:
    model = pickle.load(file)
    
# Sidebar navigation
st.sidebar.title("Aplikasi Prediksi Kelulusan Mahasiswa")
page = st.sidebar.radio("Pilih Halaman:", ["Prediksi Kelulusan", "Penjelasan Aplikasi"])

if page == "Prediksi Kelulusan":
    st.header("Masukkan Detail Berikut:")

    EconomicStatus = st.selectbox("Bagaimana Status Ekonomi Mahasiswa?", ['Kurang Mampu', 'Mampu', 'Sangat Mampu'])
    GPA = st.number_input("Berapa IPK Terakhir?", min_value=0.0, max_value=4.0, step=0.01)
    CreditsTaken = st.number_input("Berapa Jumlah Total SKS Mahasiswa?", min_value=0, max_value=144, step=1)
    SemesterCount = st.number_input("Anda Semester Berapa Saat Ini?", min_value=0, max_value=6, step=1)
    AttendancePercentage = st.number_input("Berapa Rata-Rata Kehadiran Mahasiswa?", min_value=0.0, max_value=100.0, step=0.1)
    RepeatingCourses = st.number_input("Berapa Jumlah Mata kuliah Yang Diulang?", min_value=0, max_value=20, step=1)
    FamilySupport = st.selectbox("Bagaimana Support Keluarga Mahasiswa?", ['Rendah', 'Menengah', 'Tinggi'])
    AcademicGuidance = st.selectbox("Apakah mahasiswa menerima bimbingan akademik?", ['Tidak', 'Ya'])
    StressLevel = st.selectbox("Bagaimana Tingkat Stress Mahasiswa?", ['Rendah', 'Menengah', 'Tinggi'])
    FinancialSupport = st.selectbox("Bagaimana Dukungan Keuangan Mahasiswa?", ['Tidak Memadai', 'Memadai'])
    CampusFacilitiesAccess = st.selectbox("Bagaimana Akses ke Kampus Bagi Mahasiswa?", ['Tidak Memadai', 'Memadai', 'Sangat Memadai'])

    if st.button("Prediksi"):
     if EconomicStatus == "" or GPA == 0.0 or CreditsTaken == 0 or SemesterCount == 0 or \
       AttendancePercentage == 0.0 or FamilySupport == "" or \
       AcademicGuidance == "" or StressLevel == "" or FinancialSupport == "" or CampusFacilitiesAccess == "":
        st.error("Harap isi semua kolom sebelum melakukan prediksi!")
     else:
        input_data = pd.DataFrame({
            'EconomicStatus': [EconomicStatus],
            'GPA': [GPA],
            'CreditsTaken': [CreditsTaken],
            'SemesterCount': [SemesterCount],
            'AttendancePercentage': [AttendancePercentage],
            'RepeatingCourses': [RepeatingCourses],
            'FamilySupport': [FamilySupport],
            'AcademicGuidance': [AcademicGuidance],
            'StressLevel': [StressLevel],
            'FinancialSupport': [FinancialSupport],
            'CampusFacilitiesAccess': [CampusFacilitiesAccess]
        })

        input_data_encoded = pd.get_dummies(input_data)

        input_data_encoded = input_data_encoded.reindex(columns=model.feature_names_in_, fill_value=0)

        prediction = model.predict(input_data_encoded)[0]
        prediction_probability = model.predict_proba(input_data_encoded)[0][1]

        # Mapping: 0 = Tepat Waktu, 1 = Terlambat
        # Display the result
        if prediction == 0:
            st.success("Mahasiswa akan lulus tepat waktu")
        else:
            st.error("Mahasiswa akan lulus terlambat")

elif page == "Penjelasan Aplikasi":
    # Explanation and instructions page
    st.title("Penjelasan Aplikasi")
    st.write("""
    **Aplikasi Prediksi Kelulusan Mahasiswa** dirancang untuk membantu memprediksi apakah seorang mahasiswa akan lulus tepat waktu atau terlambat berdasarkan berbagai faktor, seperti:
    - Status ekonomi : Status Ekonomi Mahasiswa
    - IPK terakhir : Indeks Prestasi Keberhasilan Terakhir Mahasiswa
    - Jumlah SKS yang telah diambil : Total jumlah SKS yang sudah diambil oleh Mahasiswa
    - Semester saat ini : Semester terakhir yang diambil oleh mahasiswa
    - Kehadiran rata-rata: Rata-rata Persentase kehadiran Mahasiswa
    - Jumlah mata kuliah yang diulang : jumlah mata kuliah yang diulang oleh mahasiswa
    - Dukungan keluarga : Dukungan orang tua bagi mahasiswa dalam kuliah
    - Bimbingan Akademik : Mahasiswa mendapatkan bimbingan akademik seperti bimbingan terhadap karir dimasa depan
    - Tingkat Stress: Tingkat stress Mahasiswa hal ini dapat diketahui melalui bimbingan ke psikolog
    - Dukungan keuangan mahasiswa: Dukungan keuangan mahasiswa entah dari beasiswa atau biaya orang tua
    - Akses Ke kampus : Dapat dipengaruhi banyak faktor seperti jarak rumah Mahasiswa Ke kampus dan sebagainya

    **Cara Menggunakan Aplikasi**:
    1. Pilih halaman **Prediksi Kelulusan** di menu navigasi di sisi kiri.
    2. Masukkan data mahasiswa pada kolom yang tersedia.
    3. Klik tombol **Prediksi** untuk melihat hasil prediksi.
    4. Jika mahasiswa diprediksi **lulus tepat waktu**, aplikasi akan menampilkan pesan sukses. Jika tidak, pesan peringatan akan ditampilkan.

    **Tujuan Aplikasi**:
    - Membantu pihak universitas atau mahasiswa memahami faktor-faktor yang memengaruhi kelulusan.
    - Memberikan wawasan untuk perbaikan proses belajar dan bimbingan akademik.

    **Catatan**: Pastikan data yang dimasukkan akurat agar hasil prediksi lebih optimal.
    """)
