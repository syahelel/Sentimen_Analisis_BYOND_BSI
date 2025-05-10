import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from wordcloud import WordCloud
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

st.title("📚 Prediksi & Visualisasi ")

# 1️⃣ Pilihan sumber data
st.subheader("📂 Pilih Sumber Dataset")

data_option = st.radio(
    "Gunakan data dari:",
    ("Page 1 (preprocessing)", "Upload file dataset (.csv)")
)

df = None
label_column = "sentimen"

if data_option == "Page 1 (preprocessing)":
    if "df" in st.session_state and st.session_state.df is not None:
        df = st.session_state.df.copy()
        st.success("✅ Data berhasil diambil dari Page 1")
    else:
        st.warning("⚠️ Data dari Page 1 belum tersedia. Silakan preprocessing dulu.")
else:
    uploaded_file = st.file_uploader("Upload file CSV", type=["csv"])
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            if 'content' not in df.columns:
                st.error("File harus memiliki kolom 'content'.")
                df = None
            else:
                st.success("✅ File berhasil diunggah.")
        except Exception as e:
            st.error(f"Gagal membaca file: {e}")
            df = None

# 2️⃣ Jika data tersedia, lanjutkan
if df is not None:
    st.subheader("🔍 Preview Data")
    st.dataframe(df.head())

    # WordCloud per Sentimen (jika ada kolom label)
    if label_column in df.columns:
        st.subheader("☁️ WordCloud per Sentimen")
        colormap_map = {
            "positif": "Greens",
            "negatif": "Reds",
            "netral": "Blues"
        }
        unique_labels = df[label_column].unique()
        for label in unique_labels:
            text = " ".join(df[df[label_column] == label]['content'].astype(str))
            cmap = colormap_map.get(str(label).lower(), "viridis")
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

    # 🔥 Tombol Prediksi
    st.subheader("🤖 Prediksi dengan Model yang Sudah Dilatih")

    if st.button("Mulai Prediksi"):
        with st.spinner("🔄 Sedang memuat model dan memproses prediksi..."):
            try:
                # 1️⃣ Load model
                with open('/workspaces/Sentimen_Analisis_BYOND_BSI/stacking_sentiment_model.pkl', 'rb') as f:
                    model = pickle.load(f)

                # 2️⃣ Prediksi
                X_new = df['content'].astype(str)
                y_pred = model.predict(X_new)

                # Tambahkan hasil prediksi ke dataframe
                df['prediksi'] = y_pred
                st.success("✅ Prediksi selesai!")

                # Tampilkan hasil
                st.subheader("📝 Hasil Prediksi")
                st.dataframe(df[['content', 'prediksi']].head(20))

                # Visualisasi distribusi prediksi
                st.subheader("📊 Distribusi Hasil Prediksi")
                pred_counts = df['prediksi'].value_counts()
                fig_pred, ax_pred = plt.subplots()
                sns.barplot(
                    x=pred_counts.index,
                    y=pred_counts.values,
                    palette={'negatif': 'red', 'positif': 'green', 'netral': 'blue'},
                    ax=ax_pred
                )
                ax_pred.set_title("Jumlah Prediksi per Sentimen")
                st.pyplot(fig_pred)

                # Jika ada kolom 'sentimen' ➔ tampilkan evaluasi
                if label_column in df.columns:
                    st.subheader("🧩 Confusion Matrix & Evaluasi")
                    cm = confusion_matrix(df[label_column], df['prediksi'], labels=model.classes_)
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

                    acc = accuracy_score(df[label_column], df['prediksi'])
                    st.markdown(f"**🎯 Akurasi: {acc:.2f}**")
                    st.text("📋 Classification Report:")
                    st.text(classification_report(df[label_column], df['prediksi']))
                else:
                    st.info("Kolom 'sentimen' tidak ada di dataset, jadi evaluasi model tidak ditampilkan.")

            except Exception as e:
                st.error(f"Terjadi error saat prediksi: {e}")
