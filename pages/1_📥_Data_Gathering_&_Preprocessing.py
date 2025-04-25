import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from functools import lru_cache
from google_play_scraper import Sort, reviews_all
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory, ArrayDictionary
from Sastrawi.StopWordRemover.StopWordRemover import StopWordRemover

st.title("📥 Data Gathering & Preprocessing")

# --- Pilih Aplikasi dan Sorting ---
app_choice = st.selectbox("Pilih Mbanking yang ingin dianalisis:", ["BYOND", "BSI Mobile"])
app_id = "co.id.bankbsi.superapp" if app_choice == "BYOND" else "com.bsm.activity2"

sort_choice = st.selectbox("Urutkan Review:", ["Most Relevant", "Newest"])
sort_review = Sort.MOST_RELEVANT if sort_choice == "Most Relevant" else Sort.NEWEST

# --- Filter Tanggal ---
col1, col2 = st.columns(2)
with col1:
    start_date = st.date_input("Mulai Dari:", value=datetime(2025, 1, 1))
with col2:
    end_date = st.date_input("Sampai Dengan:", value=datetime.today())

# --- Fungsi Label Sentimen ---
def label_sentiment(score):
    if score <= 2:
        return 'negatif'
    elif score == 3:
        return 'netral'
    else:
        return 'positif'

# Inisialisasi session state
st.session_state.setdefault("df_raw", None)
st.session_state.setdefault("df_clean", None)
st.session_state.setdefault("df", None)

# --- Ambil Review ---
if st.button("Ambil Review"):
    with st.spinner("Mengambil data dari Google Play Store..."):
        result = reviews_all(app_id, lang='id', country='id', sort=sort_review)
        df = pd.DataFrame(result)
        df['at'] = pd.to_datetime(df['at'])
        df = df[(df['at'] >= pd.Timestamp(start_date)) & (df['at'] <= pd.Timestamp(end_date))]
        df['sentimen'] = df['score'].apply(label_sentiment)
        st.session_state.df_raw = df[['userName','score','at','content','sentimen']]
        st.success(f"Berhasil mengambil {len(df)} review.")
        st.dataframe(st.session_state.df_raw)

# --- Data Cleaning ---
if st.session_state.df_raw is not None:
    st.subheader("Data Mentah (Raw)")
    st.dataframe(st.session_state.df_raw)

    if st.button("Lakukan Data Cleaning"):
        with st.spinner("Membersihkan data..."):
            df_clean = st.session_state.df_raw.drop_duplicates().fillna('')
            st.session_state.df_clean = df_clean
            st.success("Data sudah dibersihkan dari duplikasi dan NaN.")
            st.dataframe(df_clean)

# --- Load NLP Processors ---
@st.cache_resource
def load_text_processors():
    stemmer = StemmerFactory().create_stemmer()
    stop_words = StopWordRemoverFactory().get_stop_words()
    stopword_remover = StopWordRemover(ArrayDictionary(stop_words))
    return stemmer, stopword_remover

# --- Normalisasi Kata ---
norm = {
    "bgt": "banget", "brp": "berapa", "blm": "belum", "lbh": "lebih", "tp": "tapi",
    "ngga": "tidak", "nggak": "tidak", "gak": "tidak", "dpt": "dapat", "lg": "lagi",
    "krn": "karena", "jgn": "jangan", "transaksi": "transaksi", "trf": "transfer",
    "saldo": "saldo", "rek": "rekening", "cekrek": "cek rekening", "tarik": "tarik tunai",
    "setor": "setor tunai", "topup": "isi ulang", "tf": "transfer",
    "dompet": "dompet digital", "profisional": "profesional", "mulu": "terus", "payah": "jelek",
    "identivikasi": "identifikasi", "apk": "aplikasi"
}

# --- Fungsi Caching per Teks ---
@lru_cache(maxsize=None)
def cached_process(text):
    text = text.lower()
    for w, r in norm.items():
        text = text.replace(w, r)
    text = stopword_remover.remove(text)
    return " ".join(stemmer.stem(tkn) for tkn in text.split())

# --- Lakukan Transformasi ---
if st.session_state.df_clean is not None:
    stemmer, stopword_remover = load_text_processors()

    if st.button("Lakukan Preprocessing Teks"):
        df2 = st.session_state.df_clean.copy()
        df2 = df2[df2['content'].astype(str).apply(lambda x: 3 <= len(x.split()) <= 50)]

        # Progress bar
        progress = st.progress(0)
        processed = []
        total = len(df2)
        for i, text in enumerate(df2['content'], 1):
            processed.append(cached_process(text))
            progress.progress(i / total)

        df2['content'] = processed
        df2['sentimen'] = df2['score'].apply(label_sentiment)
        st.session_state.df = df2
        st.success("Preprocessing selesai, sentimen dilabeli, dan data disimpan ke session_state.")
        st.subheader("Data Setelah Preprocessing")
        st.dataframe(df2)

# Informasi lanjut
if st.session_state.df is None:
    st.info("Lakukan langkah data gathering dan preprocessing terlebih dahulu untuk lanjut ke halaman Model & Visualisasi.")
