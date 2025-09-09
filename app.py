import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------
# Load model
# ------------------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("best_cnn_model.h5")
    return model

model = load_model()

# ------------------------
# Preprocessing
# ------------------------
IMG_SIZE = (128, 128)  # ganti sesuai ukuran input model

def preprocess_image(img: Image.Image):
    img = img.resize(IMG_SIZE)
    arr = np.asarray(img).astype("float32") / 255.0
    if arr.ndim == 2:  # grayscale → RGB
        arr = np.stack([arr] * 3, axis=-1)
    if arr.shape[-1] == 4:  # RGBA → RGB
        arr = arr[..., :3]
    arr = np.expand_dims(arr, axis=0)
    return arr

# ------------------------
# Streamlit UI
# ------------------------
st.set_page_config(page_title="🐾 Smart Pet Classifier", page_icon="🐶")

st.title("🐱🐶 Smart Pet Classifier")
st.markdown(
    """
    ### Selamat datang di aplikasi *Deep Learning* ini!  
    Unggah foto hewan peliharaanmu, dan biarkan model CNN menebak apakah itu **Kucing 🐱** atau **Anjing 🐶**.  
    """
)

st.sidebar.header("⚙️ Tentang Aplikasi")
st.sidebar.info(
    """
    - **Model:** Custom CNN (H5)  
    - **Input size:** 128 × 128  
    - **Output:** 2 kelas (Cat/Dog)  
    - **Dibuat dengan:** TensorFlow + Streamlit  
    """
)
st.sidebar.write("✨ Dibuat agar prediksi terasa interaktif & menyenangkan!")

# ------------------------
# Upload
# ------------------------
uploaded_files = st.file_uploader(
    "📤 Upload gambar (JPG/PNG). Bisa lebih dari satu file:",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

if uploaded_files:
    results = []
    cols = st.columns(3)

    for i, f in enumerate(uploaded_files):
        img = Image.open(f).convert("RGB")
        arr = preprocess_image(img)
        pred = model.predict(arr, verbose=0)

        # Jika output 2 kelas (softmax), ambil argmax
        if pred.shape[1] == 2:
            class_idx = np.argmax(pred, axis=1)[0]
            confidence = float(np.max(pred))
        else:  # jika output 1 neuron sigmoid
            class_idx = int(pred[0] > 0.5)
            confidence = float(pred[0]) if class_idx == 1 else 1 - float(pred[0])

        # Label (ubah sesuai dataset training kamu)
        labels = {0: "🐱 Cat", 1: "🐶 Dog"}
        label = labels.get(class_idx, str(class_idx))

        results.append({
            "filename": f.name,
            "predicted_label": label,
            "confidence": confidence
        })

        with cols[i % 3]:
            st.image(img, caption=f"{label} ({confidence:.2%})", use_column_width=True)

    # ------------------------
    # Show table results
    # ------------------------
    st.subheader("📊 Ringkasan Hasil Prediksi")
    df = pd.DataFrame(results)
    st.dataframe(df)

    # ------------------------
    # Download CSV
    # ------------------------
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "💾 Download hasil prediksi (CSV)",
        csv,
        "predictions.csv",
        "text/csv"
    )

    # ------------------------
    # Visualisasi distribusi prediksi
    # ------------------------
    st.subheader("📈 Distribusi Prediksi")
    fig, ax = plt.subplots()
    df["predicted_label"].value_counts().plot(kind="bar", ax=ax, color=["#FF9999","#66B3FF"])
    ax.set_ylabel("Jumlah Gambar")
    ax.set_xlabel("Kelas")
    st.pyplot(fig)

else:
    st.info("👆 Upload satu atau lebih gambar untuk mulai prediksi.")
