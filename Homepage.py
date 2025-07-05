import streamlit as st

# Set page configuration
st.set_page_config(
    page_title="Sentiment Analysis BYOND & BSI Mobile",
    page_icon="📱",
    layout="wide"
)

st.title("📊 Analisis Sentimen Aplikasi M-Banking")
st.markdown("""
Selamat datang di aplikasi analisis sentimen untuk review pengguna aplikasi **BYOND** dan **BSI Mobile**.

Di aplikasi ini, kamu bisa:
- Mengambil data review dari Google Play Store
- Melakukan preprocessing (pembersihan teks)
- Memprediksi menggunakan model Ensemble Stacking Random forest dan Linear Regression yang telah dilatih
- Melihat Peforma Model
- Melihat visualisasi seperti Word Cloud, Top Words, dan Distribusi Sentimen

---

📍 Gunakan **sidebar kiri** untuk berpindah antar halaman.
""")
