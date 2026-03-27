import streamlit as st
import cv2
import numpy as np
import mediapipe as mp

# ================= CONFIG =================
st.set_page_config(page_title="AI Emotion Music System", layout="wide")

st.markdown("<h1 style='text-align:center;'>🎭 AI Emotion Music Recommender</h1>", unsafe_allow_html=True)

# ================= MEDIAPIPE =================
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh()

# ================= PLAYLIST LINKS =================
playlist_links = {
    "Happy": "https://www.youtube.com/embed/videoseries?list=YOUR_HAPPY_ID",
    "Sad": "https://www.youtube.com/embed/videoseries?list=YOUR_SAD_ID",
    "Angry": "https://www.youtube.com/embed/videoseries?list=YOUR_ANGRY_ID",
    "Neutral": "https://www.youtube.com/embed/videoseries?list=YOUR_NEUTRAL_ID"
}

# ================= UI LAYOUT =================
col1, col2 = st.columns(2)

# ===== LEFT SIDE (IMAGE UPLOAD) =====
with col1:
    st.subheader("📷 Upload Face Image")

    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png"])

    emotion = ""

    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)

        st.image(img, channels="BGR")

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb)

        if result.multi_face_landmarks:
            # 👉 Simple demo logic (you can improve later)
            emotion = "Happy"
        else:
            emotion = "No Face"

        st.success(f"Detected Emotion: {emotion}")

# ===== RIGHT SIDE (PLAYLIST) =====
with col2:
    st.subheader("🎵 Recommended Playlist")

    if emotion in playlist_links:
        st.markdown(
            f"""
            <iframe width="100%" height="300"
            src="{playlist_links[emotion]}"
            frameborder="0" allowfullscreen></iframe>
            """,
            unsafe_allow_html=True
        )
    else:
        st.info("👉 Upload image to get playlist")

# ================= FOOTER =================
st.markdown("---")
st.markdown("💡 Emotion-based music recommendation using AI")
