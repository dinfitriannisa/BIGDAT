import streamlit as st
from streamlit_option_menu import option_menu
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import time

# ==========================
# Load Models
# ==========================
@st.cache_resource
def load_models():
    # YOLO Detection
    yolo_path = r"C:\SEMESTER 7\BIG DATA\UTS\uts.1\model\Din Fitri Annisa_Laporan 4.pt"
    yolo_model = YOLO(yolo_path)

    # H5 Classifier
    h5_path = r"C:\SEMESTER 7\BIG DATA\UTS\uts.1\model\Din Fitri Annisa_Laporan 2.h5.h5"
    classifier = tf.keras.models.load_model(h5_path)

    return yolo_model, classifier

yolo_model, classifier = load_models()

# ==========================
# WC Logo
# ==========================
male_wc_url = "https://tse3.mm.bing.net/th/id/OIP.VC23B05clfgXGC2eErSCRgHaHa?cb=12&rs=1&pid=ImgDetMain&o=7&rm=3"
female_wc_url = "https://vectordad.com/wp-content/uploads/2023/01/Women-Restroom-Sign-1024x858.png"

# ==========================
# Sidebar
# ==========================
if "page" not in st.session_state:
    st.session_state.page = "Home"

with st.sidebar:
    selected = option_menu(
        menu_title="Navigation",
        options=["Home", "Classification", "Detection"],
        icons=["house", "image", "eye"],
        menu_icon="cast",
        default_index=0
    )
    st.session_state.page = selected

# ==========================
# Home Page
# ==========================
if st.session_state.page == "Home":
    st.image(
        "https://tse2.mm.bing.net/th/id/OIP.kA2kMOzGD95g9evKDh5JsgAAAA?cb=12&rs=1&pid=ImgDetMain&o=7&rm=3",
        width=150,
    )
    st.title("Universitas Syiah Kuala")
    st.subheader("Praktikum Big Data")

    st.write("**Nama:** Din Fitri Annisa")
    st.write("**NPM:** 2208108010088")

    st.markdown("""
    Selamat datang👋  
    - 🧩 **Smart Character Gate (Labubu vs Lafufu)**  
    - 🚻 **Smart Restroom Detector (Female vs Male)**  
    """)

    st.markdown("""
    ---
    ### 🧭 Cara Menggunakan:
    1. Pilih menu di **sidebar** kiri:
        - **Classification** untuk mengenali karakter Labubu atau Lafufu.  
        - **Detection** untuk mendeteksi apakah wajah termasuk Female atau Male.  
    2. **Unggah gambar** (format JPG/PNG).  
    3. Tunggu proses analisis dari model.  
    4. Lihat hasil dan pesan interaktif yang muncul.  
    5. Gunakan tombol **Kembali ke Beranda** untuk mencoba fitur lain.  
    ---
    """)

    st.info("💡 Tips: Gunakan gambar dengan pencahayaan jelas agar hasil lebih akurat.")

# ==========================
# Classification
# ==========================
elif st.session_state.page == "Classification":
    st.markdown("<h2 style='text-align:center;'>🎭 Smart Character Gate</h2>", unsafe_allow_html=True)
    st.write("Unggah gambar karakter untuk mengenali apakah itu **Labubu** atau **Lafufu**.")

    uploaded_file = st.file_uploader("Unggah gambar karakter", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, caption="Gambar yang diunggah", use_container_width=True)
        st.write("🔍 Sedang memproses...")

        time.sleep(1)

        target_size = classifier.input_shape[1:3]
        img_resized = img.resize(target_size)
        img_array = image.img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        prediction = classifier.predict(img_array)
        class_index = np.argmax(prediction)
        confidence = float(np.max(prediction))

        st.progress(int(confidence * 100))
        st.write(f"**Confidence:** {confidence:.2%}")

        if class_index == 0:
            st.markdown("""
                <div style='background: linear-gradient(135deg, #b3e5fc, #e1f5fe); padding:25px; border-radius:20px; text-align:center; box-shadow:0 4px 10px rgba(0,0,0,0.1)'>
                <h3>🧸 Selamat datang di Dunia Labubu!</h3>
                <p>Langit biru bersinar, dunia penuh imajinasi menantimu!</p>
                <p>✨ Kamu bisa menemukan figure Labubu di <b>Pop Mart</b> terdekat!</p>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
                <div style='background: linear-gradient(135deg, #f8bbd0, #fce4ec); padding:25px; border-radius:20px; text-align:center; box-shadow:0 4px 10px rgba(0,0,0,0.1)'>
                <h3>🌸 Selamat datang di Dunia Lafufu!</h3>
                <p>Bunga bermekaran dan dunia penuh warna menyambutmu!</p>
                <p>🌷 Koleksi Lafufu juga bisa ditemukan di <b>Pop Mart</b>.</p>
                </div>
            """, unsafe_allow_html=True)

        st.balloons()
        st.success("✨ Analisis selesai!")

    if st.button("🏠 Kembali ke Beranda"):
        st.session_state.page = "Home"

# ==========================
# Detection Page
# ==========================
elif st.session_state.page == "Detection":
    st.markdown("<h2 style='text-align:center;'>🚻 Smart Restroom Detector</h2>", unsafe_allow_html=True)
    st.write("Unggah gambar wajah atau ambil langsung dari kamera untuk mendeteksi **Female** atau **Male**.")

    # Pilihan: Upload atau Kamera
    option = st.radio("Pilih sumber gambar:", ["Unggah dari File", "Ambil dari Kamera"])

    if option == "Unggah dari File":
        uploaded_face = st.file_uploader("Unggah gambar wajah", type=["jpg", "jpeg", "png"])
        if uploaded_face:
            img_face = Image.open(uploaded_face)
        else:
            img_face = None
    else:
        camera_photo = st.camera_input("Ambil foto wajah menggunakan kamera")
        if camera_photo:
            img_face = Image.open(camera_photo)
        else:
            img_face = None

    if img_face:
        st.image(img_face, caption="Gambar yang digunakan", use_container_width=True)
        st.write("🔎 Sedang mendeteksi...")
        time.sleep(1)

        results = yolo_model(img_face)
        result_img = results[0].plot()
        st.image(result_img, caption="Hasil Deteksi", use_container_width=True)

        detected_gender = "Unknown"
        if len(results[0].boxes) > 0:
            cls_id = int(results[0].boxes.cls[0])
            detected_gender = "Female" if cls_id == 0 else "Male"

        if detected_gender == "Female":
            st.markdown(f"""
                <div style='background: linear-gradient(135deg, #fce4ec, #f8bbd0); padding:25px; border-radius:20px; text-align:center'>
                🚺 <h3>Silakan masuk kamar mandi perempuan.</h3>
                <img src="{female_wc_url}" width="120">
                <p>💖 Jangan lupa menjaga kebersihan!</p>
                </div>
            """, unsafe_allow_html=True)
        elif detected_gender == "Male":
            st.markdown(f"""
                <div style='background: linear-gradient(135deg, #bbdefb, #e3f2fd); padding:25px; border-radius:20px; text-align:center'>
                🚹 <h3>Silakan masuk kamar mandi laki-laki.</h3>
                <img src="{male_wc_url}" width="120">
                <p>💧 Jangan lupa menjaga kebersihan!</p>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("⚠️ Tidak terdeteksi gender dengan jelas. Silakan coba ulangi.")

        st.success("✅ Deteksi selesai!")

    if st.button("🏠 Kembali ke Beranda"):
        st.session_state.page = "Home"
