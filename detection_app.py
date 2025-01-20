import streamlit as st
import cv2
import torch
from pathlib import Path

# Fungsi untuk memuat model YOLOv5
@st.cache_resource
def load_model():
    model_path = "best.pt"  # Ganti dengan path model Anda
    return torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)

# Fungsi untuk prediksi menggunakan YOLOv5
def predict_image(model, frame):
    results = model(frame)
    # Render hasil deteksi langsung ke frame
    annotated_frame = results.render()[0]
    return annotated_frame

# Inisialisasi Streamlit
st.title("Aplikasi Deteksi Objek dengan YOLOv5")

# Tombol kontrol
if 'deteksi_dimulai' not in st.session_state:
    st.session_state.deteksi_dimulai = False

model = None  # Placeholder untuk model YOLOv5

if not st.session_state.deteksi_dimulai:
    if st.button("Mulai Deteksi", key="start_button"):
        st.session_state.deteksi_dimulai = True
        model = load_model()
else:
    if st.button("Stop Deteksi", key="stop_button"):
        st.session_state.deteksi_dimulai = False

# Loop deteksi objek
if st.session_state.deteksi_dimulai:
    st.text("Deteksi dimulai... Klik 'Stop Deteksi' untuk menghentikan.")
    cap = cv2.VideoCapture(0)  # Membuka kamera
    st_frame = st.empty()  # Placeholder untuk menampilkan frame

    while st.session_state.deteksi_dimulai:
        ret, frame = cap.read()
        if not ret:
            st.warning("Tidak dapat membaca frame dari kamera.")
            break

        # Lakukan prediksi pada frame
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Konversi ke RGB untuk Streamlit
        frame = predict_image(model, frame)
        st_frame.image(frame, channels="RGB", use_column_width=True)

    cap.release()
    st.text("Deteksi dihentikan.")

st.text("Aplikasi siap untuk deteksi objek.")
