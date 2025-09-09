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
# Preprocessing function
# ------------------------
IMG_SIZE = (128, 128)  # ganti sesuai input model saat training

def preprocess_image(img: Image.Image):
    img = img.resize(IMG_SIZE)
    arr = np.asarray(img).astype("float32") / 255.0
    if arr.ndim == 2:  # grayscale ke RGB
        arr = np.stack([arr] * 3, axis=-1)
    if arr.shape[-1] == 4:  # RGBA → RGB
        arr = arr[..., :3]
    arr = np.expand_dims(arr, axis=0)
    return arr

# ------------------------
# Streamlit UI
# ------------------------
st.set_page_config(page_title="CNN Image Classifier", page_icon="🖼️")

st.title("🖼️ CNN Image Classifier (best_cnn_model.h5)")
st.write("Upload gambar untuk diklasifikasikan oleh model CNN kamu.")

uploaded_files = st.file_uploader(
    "Upload gambar (jpg/png), bisa lebih dari satu",
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

        # Jika output 2 kelas: ambil argmax
        if pred.shape[1] == 2:
            class_idx = np.argmax(pred, axis=1)[0]
            confidence = float(np.max(pred))
        else:
            class_idx = int(pred[0] > 0.5)
            confidence = float(pred[0])

        # Label manual (ubah sesuai training)
        labels = {0: "Class 0", 1: "Class 1"}
        label = labels.get(class_idx, str(class_idx))

        results.append({
            "filename": f.name,
            "predicted_label": label,
            "confidence": confidence
        })

        with cols[i % 3]:
            st.image(img, caption=f"{f.name} → {label} ({confidence:.2f})", use_column_width=True)

    # Tampilkan tabel hasil
    st.subheader("Hasil Prediksi")
    df = pd.DataFrame(results)
    st.dataframe(df)

    # Download CSV hasil
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download hasil CSV", csv, "predictions.csv", "text/csv")
