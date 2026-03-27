import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from deepface import DeepFace
import os

# ================= CONFIG =================
st.set_page_config(page_title="AI Emotion Music System", layout="wide")

st.markdown("<h1 style='text-align:center;'>🎭 AI Emotion Music Recommender</h1>", unsafe_allow_html=True)

# ================= LOAD MODEL =================
model = load_model("model.h5")
labels = np.load("labels.npy")

# ================= MEDIAPIPE =================
mp_holistic = mp.solutions.holistic
holis = mp_holistic.Holistic()

# ================= PLAYLIST LINKS =================
playlist_links = {
    "Happy": "https://www.youtube.com/embed/videoseries?list=YOUR_HAPPY_ID",
    "Sad": "https://www.youtube.com/embed/videoseries?list=YOUR_SAD_ID",
    "Angry": "https://www.youtube.com/embed/videoseries?list=YOUR_ANGRY_ID",
    "Neutral": "https://www.youtube.com/embed/videoseries?list=YOUR_NEUTRAL_ID"
}

# ================= SESSION =================
if "emotion" not in st.session_state:
    st.session_state["emotion"] = ""

if "person" not in st.session_state:
    st.session_state["person"] = ""

if "identified" not in st.session_state:
    st.session_state["identified"] = False

# ================= VIDEO PROCESSOR =================
class EmotionProcessor:
    def recv(self, frame):
        frm = frame.to_ndarray(format="bgr24")
        frm = cv2.flip(frm, 1)

        rgb = cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)
        res = holis.process(rgb)

        if res.face_landmarks:
            lst = []

            for i in res.face_landmarks.landmark:
                lst.append(i.x - res.face_landmarks.landmark[1].x)
                lst.append(i.y - res.face_landmarks.landmark[1].y)

            lst = np.array(lst).reshape(1, -1)

            # ===== Emotion Detection =====
            pred = labels[np.argmax(model.predict(lst, verbose=0))]
            st.session_state["emotion"] = pred

            # ===== Face Recognition =====
            if not st.session_state["identified"]:
                try:
                    result = DeepFace.find(
                        img_path=frm,
                        db_path="database",
                        enforce_detection=False,
                        detector_backend="opencv"
                    )

                    if len(result[0]) > 0:
                        name = result[0].iloc[0]['identity'].split(os.sep)[-2]
                    else:
                        name = "Unknown"

                except:
                    name = "Unknown"

                st.session_state["person"] = name
                st.session_state["identified"] = True

            cv2.putText(frm, f"{st.session_state['person']} - {pred}",
                        (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2)

        else:
            cv2.putText(frm, "No Face Detected",
                        (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 255), 2)

        return av.VideoFrame.from_ndarray(frm, format="bgr24")

# ================= UI LAYOUT =================
col1, col2 = st.columns([1, 1])

# ===== CAMERA =====
with col1:
    st.subheader("📷 Live Emotion Detection")
    webrtc_streamer(
        key="camera",
        video_processor_factory=EmotionProcessor
    )

# ===== PLAYLIST =====
with col2:
    st.subheader("🎵 Your Playlist")

    emotion = st.session_state["emotion"]

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
        st.info("👉 Detect your emotion to load playlist")

# ================= RESULT =================
st.markdown("---")

if st.session_state["emotion"]:
    st.success(
        f"👤 {st.session_state['person']} | 😊 {st.session_state['emotion']}"
    )
