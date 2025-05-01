import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import cv2
from fer import FER
import numpy as np
import av
import os

# تعطيل تحذيرات TensorFlow غير الضرورية
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# إعداد RTC Configuration مع STUN و TURN server
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {
            "urls": ["turn:openrelay.metered.ca:80", "turn:openrelay.metered.ca:443"],
            "username": "openrelayproject",
            "credential": "openrelayproject"
        }
    ]}
)

# إنشاء كائن الكشف عن المشاعر
detector = FER(mtcnn=False)

# تعريف فئة لمعالجة الفيديو
class EmotionDetector(VideoProcessorBase):
    def __init__(self):
        self.detector = detector

    def recv(self, frame):
        try:
            # تحويل الإطار إلى صورة OpenCV
            img = frame.to_ndarray(format="bgr24")
            img = cv2.resize(img, (640, 480))

            # الكشف عن المشاعر
            result = self.detector.detect_emotions(img)

            # معالجة النتائج
            for face in result:
                x, y, w, h = face['box']
                emotions = face['emotions']
                # رسم مستطيل حول الوجه
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                # طباعة أكثر شعور بارز
                dominant_emotion = max(emotions, key=emotions.get)
                cv2.putText(img, dominant_emotion, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            return av.VideoFrame.from_ndarray(img, format="bgr24")

        except Exception as e:
            print(f"Error during frame processing: {e}")
            return frame

# واجهة Streamlit
st.title("🎭 Emotion Detection App")
st.write("ابدأ الكاميرا وسنحاول التعرف على مشاعرك")

# تشغيل الكاميرا ومعالجة الفيديو
webrtc_streamer(
    key="emotion-detection",
    video_processor_factory=EmotionDetector,
    rtc_configuration=RTC_CONFIGURATION
)
