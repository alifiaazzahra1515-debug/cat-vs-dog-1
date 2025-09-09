# app.py
import streamlit as st
from PIL import Image
import numpy as np
import io
import requests
import json
import os
import matplotlib.pyplot as plt
import tensorflow as tf

st.set_page_config(page_title="Cat vs Dog - Demo", page_icon="🐱🐶", layout="centered")

# -----------------------
# CONFIG / HUGGINGFACE URLS
# -----------------------
MODEL_URL = "https://huggingface.co/alifia1/catvsdog/resolve/main/model_mobilenetv2.keras"
CLASS_INDICES_URL = "https://huggingface.co/alifia1/catvsdog/resolve/main/class_indices.json"

# Local cache paths
MODEL_CACHE_DIR = "models_cache"
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
MODEL_LOCAL_PATH = os.path.join(MODEL_CACHE_DIR, "model_mobilenetv2.keras")
CLASS_LOCAL_PATH = os.path.join(MODEL_CACHE_DIR, "class_indices.json")

# -----------------------
# UTIL: download helper
# -----------------------
def download_file(url, local_path, desc=None):
    """
    Download url to local_path if not exist. Return local_path.
    """
    if os.path.exists(local_path):
        return local_path
    try:
        with st.spinner(f"Downloading {desc or 'file'} from remote..."):
            resp = requests.get(url, stream=True, timeout=60)
            resp.raise_for_status()
            with open(local_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        return local_path
    except Exception as e:
        # if download failed, remove possibly partial file
        if os.path.exists(local_path):
            try:
                os.remove(local_path)
            except:
                pass
        raise RuntimeError(f"Download failed for {url}: {e}")

# -----------------------
# CACHE: load model & classes
# -----------------------
@st.cache_resource(show_spinner=False)
def load_model_and_classes(model_url=MODEL_URL, class_url=CLASS_INDICES_URL):
    # 1) download model file (if needed)
    try:
        model_path = download_file(model_url, MODEL_LOCAL_PATH, desc="model")
    except Exception as e:
        raise RuntimeError(f"Could not download model: {e}")

    # 2) load keras model
    try:
        model = tf.keras.models.load_model(model_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load model from {model_path}: {e}")

    # 3) download class indices json
    try:
        class_path = download_file(class_url, CLASS_LOCAL_PATH, desc="class indices")
    except Exception as e:
        # continue but warn
        st.warning(f"Gagal mengunduh class_indices.json: {e}. Akan coba baca lokal jika ada.")
        class_path = CLASS_LOCAL_PATH

    # 4) load class mapping
    labels_map = None
    if os.path.exists(class_path):
        try:
            with open(class_path, "r") as f:
                class_json = json.load(f)
            # class_json expected like {"Cat":0,"Dog":1} or {"0":"Cat","1":"Dog"}
            # Normalize to index->label map
            if all(isinstance(v, int) for v in class_json.values()):
                # name->index mapping -> invert
                labels_map = {int(v): k for k, v in class_json.items()}
            else:
                # maybe index->name already
                labels_map = {int(k): v for k, v in class_json.items()}
        except Exception as e:
            st.warning(f"Gagal membaca class_indices.json: {e}")
            labels_map = {0: "Cat", 1: "Dog"}  # fallback
    else:
        labels_map = {0: "Cat", 1: "Dog"}  # fallback

    return model, labels_map

# Try loading model (on startup)
try:
    model, labels_map = load_model_and_classes()
except Exception as e:
    st.error(f"Error saat memuat model atau class indices: {e}")
    st.stop()

# -----------------------
# APP UI
# -----------------------
st.title("Cat vs Dog Classifier")
st.markdown(
    """
Aplikasi ini menggunakan **MobileNetV2 (pretrained)** yang sudah di-fine-tune untuk membedakan kucing dan anjing.  
Model dan class mapping disimpan di Hugging Face.  
**Preprocessing** mengikuti pipeline training: resize ke **128×128**, ubah ke array, dan normalisasi `pixel / 255.0`.
"""
)

with st.expander("Cara kerja singkat (untuk catatan)"):
    st.write(
        """
- Masukkan gambar (upload atau URL) berformat JPG/PNG.
- Gambar diresize ke 128×128 dan distandarisasi (/255).
- Model menghasilkan probabilitas, threshold 0.5 -> Dog, <0.5 -> Cat.
- Model diunduh sekali dan disimpan secara lokal di folder `models_cache/`.
"""
    )

st.sidebar.header("Pengaturan & Info")
st.sidebar.write("Model: MobileNetV2 (fine-tuned)\nTarget image size: 128x128")
st.sidebar.write("Model source: Hugging Face (downloaded)")

# -----------------------
# Input: image via upload or URL or sample
# -----------------------
input_mode = st.radio("Pilih sumber gambar:", ("Upload file", "Image URL", "Contoh sample"))

image_bytes = None
image_pil = None

if input_mode == "Upload file":
    uploaded_file = st.file_uploader("Upload file gambar (JPG/PNG)", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        try:
            image_bytes = uploaded_file.read()
            image_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as e:
            st.error(f"Gagal membaca file: {e}")

elif input_mode == "Image URL":
    img_url = st.text_input("Masukkan URL gambar (public):")
    if img_url:
        try:
            with st.spinner("Mengunduh gambar..."):
                resp = requests.get(img_url, timeout=30)
                resp.raise_for_status()
                image_bytes = resp.content
                image_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as e:
            st.error(f"Gagal mengunduh atau membaca gambar dari URL: {e}")

else:  # sample
    st.write("Gunakan contoh dari dataset publik atau upload sendiri.")
    sample_col1, sample_col2 = st.columns(2)
    with sample_col1:
        sample_cat = st.button("Contoh: Kucing (sample online)")
    with sample_col2:
        sample_dog = st.button("Contoh: Anjing (sample online)")

    if sample_cat:
        # small public sample image; if not accessible, user can upload
        sample_link = "https://raw.githubusercontent.com/mdavids1990/cats-vs-dogs-samples/main/cat.jpg"
        try:
            resp = requests.get(sample_link, timeout=20)
            resp.raise_for_status()
            image_bytes = resp.content
            image_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as e:
            st.warning("Tidak dapat mengambil sample; silakan upload manual.")
    if sample_dog:
        sample_link = "https://raw.githubusercontent.com/mdavids1990/cats-vs-dogs-samples/main/dog.jpg"
        try:
            resp = requests.get(sample_link, timeout=20)
            resp.raise_for_status()
            image_bytes = resp.content
            image_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as e:
            st.warning("Tidak dapat mengambil sample; silakan upload manual.")

# Show image preview
if image_pil is not None:
    st.subheader("Preview Gambar")
    st.image(image_pil, use_column_width=True)

    # Preprocess
    IMG_SIZE = 128
    def preprocess_pil(img: Image.Image, target_size=(IMG_SIZE, IMG_SIZE)):
        img_resized = img.resize(target_size)
        arr = np.asarray(img_resized).astype("float32") / 255.0
        if arr.ndim == 2:  # grayscale -> to RGB
            arr = np.stack([arr]*3, axis=-1)
        if arr.shape[-1] == 4:  # RGBA -> RGB
            arr = arr[..., :3]
        arr = np.expand_dims(arr, axis=0)
        return arr

    X = preprocess_pil(image_pil)

    # Predict
    with st.spinner("Melakukan prediksi..."):
        try:
            pred = model.predict(X)
            # pred shape could be (1,1) or (1,) depending on model
            prob = float(pred.flatten()[0])
            # since training uses sigmoid binary classification:
            label_idx = 1 if prob > 0.5 else 0
            label_name = labels_map.get(label_idx, str(label_idx))
            confidence = prob if label_idx == 1 else 1 - prob  # confidence for predicted class
        except Exception as e:
            st.error(f"Gagal prediksi: {e}")
            st.stop()

    # Display results
    st.markdown("### Hasil Prediksi")
    st.write(f"**Prediksi:** `{label_name}`")
    st.write(f"**Probabilitas (as Dog probability):** {prob:.4f}")
    st.write(f"**Confidence pada kelas terpilih:** {confidence:.4f}")

    # Probability bar
    fig, ax = plt.subplots(figsize=(6,1.2))
    ax.barh([0], [prob], height=0.5)
    ax.set_xlim(0,1)
    ax.set_xlabel("Probabilitas (Dog)")
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    st.pyplot(fig)

    st.success(f"Label: {label_name} (confidence {confidence:.2%})")

    st.markdown("---")
    st.caption("Notes: Model threshold 0.5. Jika model dilatih berbeda, threshold sebaiknya disesuaikan.")

else:
    st.info("Belum ada gambar. Upload file atau masukkan URL gambar untuk mulai prediksi.")

# -----------------------
# Footer / Usage
# -----------------------
st.markdown("---")
st.markdown("### Cara deploy")
st.markdown(
    """
1. Simpan `app.py` dan `requirements.txt` di satu folder.  
2. Jalankan di lokal:
