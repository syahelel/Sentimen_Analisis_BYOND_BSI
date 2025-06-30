import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from huggingface_hub import hf_hub_download, login
import joblib
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_extraction.text import CountVectorizer

st.title("🛠️ Predict & Visualisasi")

# 1️⃣ Autentikasi Hugging Face
api_token = st.secrets["huggingface"]["api_token"]
login(token=api_token)

# 2️⃣ Load Model dari Hugging Face (cache)
@st.cache_resource(show_spinner="🔄 Memuat...")
def load_model_from_huggingface():
    repo_id = "Syahelel/byond-sentimen-rf-lr-stacking-model"
    filename = "stacking_sentiment_model.pkl"
    model_path = hf_hub_download(repo_id=repo_id, filename=filename)
    model = joblib.load(model_path)
    return model

model = load_model_from_huggingface()
st.success("✅ Model berhasil dimuat!")

# 3️⃣ Dataset Pilihan
st.subheader("📂 Pilih Sumber Dataset")
data_option = st.radio(
    "Gunakan data dari:",
    ("Page Data Gathering", "Upload file dataset (.csv)")
)

df_new = None

if data_option == "Page Data Gathering":
    if "df" in st.session_state and st.session_state.df is not None:
        df_new = st.session_state.df.copy()
        st.success("✅ Data berhasil diambil dari Page Data Gathering")
    else:
        st.warning("⚠️ Data dari Page Data Gathering belum tersedia.")
else:
    uploaded_file = st.file_uploader("Upload file CSV", type=["csv"])
    if uploaded_file is not None:
        try:
            df_new = pd.read_csv(uploaded_file)
            if 'content' not in df_new.columns:
                st.error("File harus memiliki kolom 'content'.")
                df_new = None
            else:
                st.success("✅ File berhasil diunggah.")
        except Exception as e:
            st.error(f"Gagal membaca file: {e}")
            df_new = None

# 4️⃣ Proses Predict & Visualisasi
if df_new is not None:
    st.subheader("🔍 Preview Data")
    st.dataframe(df_new.head())

    if st.button("🔮 Predict & Visualisasikan"):
        X_new = df_new['content'].astype(str)
        df_new['predicted_sentiment'] = model.predict(X_new)
        
        st.success("✅ Prediksi selesai!")

        # Visualisasi Sebaran Sentimen
        st.subheader("📊 Sebaran Sentimen Prediksi")
        fig1, ax1 = plt.subplots()
        palette_colors = {'negatif': 'red', 'positif': 'green', 'netral': 'blue'}
        sns.countplot(x='predicted_sentiment', data=df_new, palette=palette_colors, ax=ax1)
        ax1.set_title("Sebaran Prediksi Sentimen")
        st.pyplot(fig1)

        # 🔎 Evaluasi Kinerja jika label tersedia
        if 'sentimen' in df_new.columns:
            st.subheader("🧪 Evaluasi Model")
            y_true = df_new['sentimen']
            y_pred = df_new['predicted_sentiment']

            # Accuracy
            acc = accuracy_score(y_true, y_pred)
            st.markdown(f"**✅ Akurasi: {acc:.2f}**")

            # Classification Report
            st.text("📋 Classification Report:")
            st.text(classification_report(y_true, y_pred))

            # Confusion Matrix
            st.subheader("🧩 Confusion Matrix")
            cm = confusion_matrix(y_true, y_pred, labels=model.classes_)
            fig_cm, ax_cm = plt.subplots()
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=model.classes_,
                yticklabels=model.classes_,
                ax=ax_cm
            )
            ax_cm.set_xlabel("Predicted")
            ax_cm.set_ylabel("Actual")
            ax_cm.set_title("Confusion Matrix")
            st.pyplot(fig_cm)
        else:
            st.info("⚠️ Kolom 'sentimen' tidak ditemukan, evaluasi model tidak dapat dilakukan.")

        # WordCloud per Sentimen
        st.subheader("☁️ WordCloud per Sentimen")
        for label in df_new['predicted_sentiment'].unique():
            text = " ".join(df_new[df_new['predicted_sentiment'] == label]['content'].astype(str))
            cmap = {
                'positif': 'Greens',
                'negatif': 'Reds',
                'netral': 'Blues'
            }.get(label.lower(), "viridis")
            wordcloud = WordCloud(
                width=800,
                height=400,
                background_color='white',
                colormap=cmap
            ).generate(text)
            st.markdown(f"**Sentimen: {label}**")
            fig_wc, ax_wc = plt.subplots(figsize=(10, 4))
            ax_wc.imshow(wordcloud, interpolation='bilinear')
            ax_wc.axis("off")
            st.pyplot(fig_wc)

        # Top 10 Words Setiap Sentimen
        st.subheader("🔝 Kata yang sering muncul tiap pada tiap sentimen")
        vectorizer = CountVectorizer(stop_words='english')
        top_words_dict = {}

        for label in df_new['predicted_sentiment'].unique():
            text_data = df_new[df_new['predicted_sentiment'] == label]['content'].astype(str)
            if text_data.empty:
                continue
            X_vec = vectorizer.fit_transform(text_data)
            sum_words = X_vec.sum(axis=0)
            words_freq = [(word, sum_words[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
            words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)[:10]
            top_words_dict[label] = words_freq

            words = [w for w, _ in words_freq]
            counts = [c for _, c in words_freq]

            fig2, ax2 = plt.subplots()
            sns.barplot(y=words, x=counts, ax=ax2,
                        color=palette_colors.get(label.lower(), 'gray'))
            ax2.set_title(f"Kata yang sering muncul pada : - {label}")
            st.pyplot(fig2)

        # 🔥 Tampilkan daftar Top 10 secara eksplisit juga (di luar grafik)
        st.subheader("📋 Kata yang sering muncul pada : Positif, Negatif, Netral")
        for label in ['positif', 'negatif', 'netral']:
            st.markdown(f"**{label.capitalize()}**")
            if label in top_words_dict:
                top_list = top_words_dict[label]
                st.write(pd.DataFrame(top_list, columns=["Kata", "Jumlah"]))
            else:
                st.info(f"Tidak ada data untuk label {label}.")
