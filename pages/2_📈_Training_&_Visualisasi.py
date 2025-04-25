import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

st.title("📚 Training & Visualisasi")

# 1. Pilihan sumber data
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
            if 'content' not in df.columns or label_column not in df.columns:
                st.error(f"File harus memiliki kolom 'content' dan '{label_column}'.")
                df = None
            else:
                st.success("✅ File berhasil diunggah.")
        except Exception as e:
            st.error(f"Gagal membaca file: {e}")
            df = None

# 2. Jika data tersedia, lanjutkan
if df is not None:
    st.subheader("🔍 Preview Data")
    st.dataframe(df.head())

    # 3. Visualisasi Distribusi Sentimen
    st.subheader("📊 Distribusi Sentimen")
    label_counts = df[label_column].value_counts()
    fig1, ax1 = plt.subplots()
    sns.barplot(x=label_counts.index, y=label_counts.values, ax=ax1)
    ax1.set_title("Jumlah data per sentimen")
    st.pyplot(fig1)

    # 4. WordCloud per Sentimen (optional)
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

    # 5. Training Model
    st.subheader("🤖 Training Random Forest")

    if st.button("Mulai Training"):
        X = df['content'].astype(str)
        y = df[label_column]

        vectorizer = TfidfVectorizer()
        X_vec = vectorizer.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X_vec, y, test_size=0.2, random_state=42
        )

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        st.success("🎉 Model berhasil dilatih!")

        st.markdown(f"📈 Akurasi: {accuracy_score(y_test, y_pred):.2f}")
        st.text("📋 Classification Report:")
        st.text(classification_report(y_test, y_pred))

        # Confusion Matrix
        st.subheader("🧩 Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
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
