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
import pandas as pd
import zipfile
from pathlib import Path
from tqdm import tqdm
import tempfile

# ---------------------------
# Config halaman & theme UI
# ---------------------------
st.set_page_config(page_title="Cat vs Dog - Pro App", page_icon="🐱🐶", layout="wide")
# Minimal style polish
st.markdown(
    """
<style>
.header { font-size:28px; font-weight:700; }
.small-desc { color: #666; font-size:14px; }
.card { background: #fff; border-radius: 12px; padding: 16px; box-shadow: 0 2px 6px rgba(0,0,0,0.06); }
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------
# Constants - Hugging Face links
# ---------------------------
MODEL_URL = "https://huggingface.co/alifia1/catvsdog/resolve/main/model_mobilenetv2.keras"
CLASS_INDICES_URL = "https://huggingface.co/alifia1/catvsdog/resolve/main/class_indices.json"

MODEL_CACHE_DIR = "models_cache"
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
MODEL_LOCAL_PATH = os.path.join(MODEL_CACHE_DIR, "model_mobilenetv2.keras")
CLASS_LOCAL_PATH = os.path.join(MODEL_CACHE_DIR, "class_indices.json")

IMG_SIZE = 128

# ---------------------------
# Helpers
# ---------------------------
def get_hf_headers():
    # Support Hugging Face private repos via HF token in env var HF_TOKEN or Streamlit secrets
    token = None
    # Prefer st.secrets if available (Streamlit Cloud)
    try:
        token = st.secrets.get("HF_TOKEN")
    except Exception:
        token = None
    if not token:
        token = os.environ.get("HF_TOKEN")
    if token:
        return {"Authorization": f"Bearer {token}"}
    return {}

def download_file(url, local_path, desc=None):
    if os.path.exists(local_path):
        return local_path
    headers = get_hf_headers()
    try:
        with st.spinner(f"Downloading {desc or 'file'} from remote..."):
            resp = requests.get(url, stream=True, headers=headers, timeout=60)
            resp.raise_for_status()
            with open(local_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        return local_path
    except Exception as e:
        if os.path.exists(local_path):
            try:
                os.remove(local_path)
            except:
                pass
        raise RuntimeError(f"Download failed for {url}: {e}")

@st.cache_resource(show_spinner=False)
def load_model_and_classes():
    # download model and class json (if public) or use local copy if provided manually
    try:
        model_path = download_file(MODEL_URL, MODEL_LOCAL_PATH, desc="model")
    except Exception as e:
        raise RuntimeError(f"Could not download model: {e}")
    try:
        model = tf.keras.models.load_model(model_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load model from {model_path}: {e}")

    # classes
    labels_map = {0: "Cat", 1: "Dog"}  # fallback
    try:
        class_path = download_file(CLASS_INDICES_URL, CLASS_LOCAL_PATH, desc="class indices")
        with open(class_path, "r") as f:
            class_json = json.load(f)
        if all(isinstance(v, int) for v in class_json.values()):
            labels_map = {int(v): k for k, v in class_json.items()}
        else:
            labels_map = {int(k): v for k, v in class_json.items()}
    except Exception:
        # ignore - fallback used
        pass

    return model, labels_map

def preprocess_pil(img: Image.Image, target_size=(IMG_SIZE, IMG_SIZE)):
    img_resized = img.resize(target_size)
    arr = np.asarray(img_resized).astype("float32") / 255.0
    if arr.ndim == 2:
        arr = np.stack([arr]*3, axis=-1)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    arr = np.expand_dims(arr, axis=0)
    return arr

def predict_image(model, img_pil):
    X = preprocess_pil(img_pil)
    pred = model.predict(X)
    prob = float(pred.flatten()[0])
    label_idx = 1 if prob > 0.5 else 0
    return label_idx, prob

def extract_zip_to_temp(zip_bytes):
    tmpdir = tempfile.mkdtemp()
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
        z.extractall(tmpdir)
    return tmpdir

# ---------------------------
# Load model (cached)
# ---------------------------
try:
    model, labels_map = load_model_and_classes()
except Exception as e:
    st.error("Error memuat model — lihat pesan berikut:")
    st.exception(e)
    st.stop()

# ---------------------------
# Layout - Sidebar
# ---------------------------
with st.sidebar:
    st.header("Settings")
    st.write("Model: MobileNetV2 (fine-tuned)")
    st.write(f"Input size: {IMG_SIZE}×{IMG_SIZE}")
    st.markdown("---")
    st.subheader("Hugging Face Token (opsional)")
    st.info("Untuk repo privat: simpan HF token sebagai `HF_TOKEN` di Secrets (Streamlit) atau env var.")
    if "HF_TOKEN" in os.environ:
        st.success("HF_TOKEN ditemukan di environment.")
    st.markdown("---")
    st.subheader("Quick actions")
    if st.button("Clear cached model"):
        # clear cache by deleting file + resetting cache resource
        try:
            if os.path.exists(MODEL_LOCAL_PATH):
                os.remove(MODEL_LOCAL_PATH)
            if os.path.exists(CLASS_LOCAL_PATH):
                os.remove(CLASS_LOCAL_PATH)
            st.cache_resource.clear()
            st.success("Cache model dihapus. Reload aplikasi.")
        except Exception as e:
            st.error(f"Gagal menghapus cache: {e}")

st.title("Cat vs Dog — Professional Streamlit App")
st.markdown('<div class="small-desc">Upload gambar tunggal / multiple / zip, lihat preview, lakukan prediksi batch & download hasil CSV.</div>', unsafe_allow_html=True)
st.markdown("---")

# ---------------------------
# Main columns: left (controls) right (results)
# ---------------------------
left, right = st.columns([1, 1.3])

with left:
    st.subheader("Input gambar")
    input_mode = st.radio("Pilih mode input:", ["Single upload", "Multiple files", "Upload ZIP", "Image URL", "Sample gallery"])

    uploaded_files = []
    zip_tmpdir = None
    single_image = None

    if input_mode == "Single upload":
        f = st.file_uploader("Upload 1 gambar (jpg/png)", type=["jpg", "jpeg", "png"], key="single")
        if f:
            try:
                single_image = Image.open(f).convert("RGB")
            except Exception as e:
                st.error(f"Gagal membaca file: {e}")

    elif input_mode == "Multiple files":
        uploaded_files = st.file_uploader("Upload beberapa gambar (Ctrl+click) (jpg/png)", type=["jpg","jpeg","png"], accept_multiple_files=True, key="multi")
        if uploaded_files:
            st.info(f"{len(uploaded_files)} file ter-upload")

    elif input_mode == "Upload ZIP":
        z = st.file_uploader("Upload ZIP berisi gambar (jpg/png)", type=["zip"], key="zip")
        if z:
            try:
                zip_bytes = z.read()
                zip_tmpdir = extract_zip_to_temp(zip_bytes)
                st.success(f"ZIP diekstrak ke: {zip_tmpdir}")
            except Exception as e:
                st.error(f"Gagal ekstrak ZIP: {e}")

    elif input_mode == "Image URL":
        img_url = st.text_input("Masukkan URL gambar publik:")
        if img_url:
            try:
                resp = requests.get(img_url, timeout=20)
                resp.raise_for_status()
                single_image = Image.open(io.BytesIO(resp.content)).convert("RGB")
            except Exception as e:
                st.error(f"Gagal ambil gambar dari URL: {e}")

    else:  # Sample gallery
        st.write("Pilih contoh gambar untuk testing cepat")
        col1, col2, col3 = st.columns(3)
        sample_urls = [
            ("Cat 1", "https://raw.githubusercontent.com/mdavids1990/cats-vs-dogs-samples/main/cat.jpg"),
            ("Dog 1", "https://raw.githubusercontent.com/mdavids1990/cats-vs-dogs-samples/main/dog.jpg"),
            # add more public sample links if desired
        ]
        with col1:
            if st.button("Sample: Cat"):
                try:
                    resp = requests.get(sample_urls[0][1], timeout=20); resp.raise_for_status()
                    single_image = Image.open(io.BytesIO(resp.content)).convert("RGB")
                except Exception as e:
                    st.warning("Gagal ambil sample.")
        with col2:
            if st.button("Sample: Dog"):
                try:
                    resp = requests.get(sample_urls[1][1], timeout=20); resp.raise_for_status()
                    single_image = Image.open(io.BytesIO(resp.content)).convert("RGB")
                except Exception as e:
                    st.warning("Gagal ambil sample.")

    st.markdown("---")
    st.subheader("Options")
    thresh = st.slider("Threshold (0.0 - 1.0) untuk klasifikasi Dog", 0.0, 1.0, 0.5, 0.01)
    show_prob_plot = st.checkbox("Tampilkan bar probabilitas", True)
    run_button = st.button("Run Prediction", type="primary")

with right:
    st.subheader("Preview & Results")
    results = []  # list of dicts: {filename, label, prob}

    # Prepare list of image sources depending on selected input mode
    imgs_for_pred = []  # tuples (name, PIL.Image)
    if single_image:
        imgs_for_pred.append(("uploaded_image", single_image))
    if uploaded_files:
        for up in uploaded_files:
            try:
                pil = Image.open(up).convert("RGB")
                imgs_for_pred.append((up.name, pil))
            except Exception as e:
                st.warning(f"Gagal baca {up.name}: {e}")
    if zip_tmpdir:
        # find images in extracted folder
        p = Path(zip_tmpdir)
        image_paths = list(p.rglob("*"))
        image_paths = [p for p in image_paths if p.suffix.lower() in [".jpg",".jpeg",".png"]]
        st.info(f"{len(image_paths)} gambar ditemukan di ZIP")
        for ip in image_paths:
            try:
                pil = Image.open(ip).convert("RGB")
                imgs_for_pred.append((ip.name, pil))
            except Exception as e:
                st.warning(f"Gagal baca {ip}: {e}")

    # Show thumbnails
    if imgs_for_pred:
        cols = st.columns(3)
        for i, (name, pil) in enumerate(imgs_for_pred):
            with cols[i % 3]:
                st.image(pil.resize((160,160)), caption=str(name), use_column_width=False)

    if run_button:
        if not imgs_for_pred:
            st.warning("Belum ada gambar untuk diprediksi. Upload atau pilih sample dulu.")
        else:
            progress_bar = st.progress(0)
            total = len(imgs_for_pred)
            for idx, (name, pil) in enumerate(imgs_for_pred):
                try:
                    label_idx, prob = predict_image(model, pil)
                    # use user threshold for deciding label
                    pred_label_idx = 1 if prob >= thresh else 0
                    pred_label = labels_map.get(pred_label_idx, str(pred_label_idx))
                    confidence = prob if pred_label_idx == 1 else 1 - prob
                    results.append({
                        "filename": str(name),
                        "pred_label": pred_label,
                        "prob_dog": float(prob),
                        "confidence": float(confidence)
                    })
                except Exception as e:
                    results.append({
                        "filename": str(name),
                        "pred_label": "ERROR",
                        "prob_dog": None,
                        "confidence": None,
                        "error": str(e)
                    })
                progress_bar.progress((idx+1)/total)
            st.success("Selesai prediksi batch ✅")

            # Display results table
            df_res = pd.DataFrame(results)
            st.dataframe(df_res)

            # Probability bar visualization per image
            if show_prob_plot:
                st.markdown("#### Probabilities (Dog score)")
                fig, ax = plt.subplots(figsize=(8, max(2, total*0.3)))
                ax.barh(df_res["filename"], df_res["prob_dog"].fillna(0))
                ax.set_xlim(0,1)
                ax.set_xlabel("Probability (Dog)")
                st.pyplot(fig)

            # Provide CSV download
            csv_bytes = df_res.to_csv(index=False).encode("utf-8")
            st.download_button("Download results (CSV)", data=csv_bytes, file_name="predictions.csv", mime="text/csv")

            # OPTION: allow images + CSV bundle download (zip)
            if st.checkbox("Download images + CSV as ZIP"):
                import zipfile, io
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, "w") as zf:
                    # add CSV
                    zf.writestr("predictions.csv", csv_bytes)
                    # add each image
                    for name, pil in imgs_for_pred:
                        buf = io.BytesIO()
                        pil.save(buf, format="PNG")
                        zf.writestr(f"images/{name}.png", buf.getvalue())
                zip_buffer.seek(0)
                st.download_button("Download ZIP (images + csv)", data=zip_buffer, file_name="predictions_bundle.zip", mime="application/zip")

# Footer
st.markdown("---")
st.markdown(
    """
**Notes & Tips**  
- Threshold dapat disesuaikan sesuai trade-off false-positive/false-negative.  
- Untuk repo privat Hugging Face, simpan token di Secrets (Streamlit) atau environment variable `HF_TOKEN`.  
- Untuk performa lebih baik saat batch besar, jalankan di server dengan GPU (jika model butuh).
"""
)
