import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import cv2
from fer import FER
import numpy as np
import av
import os

# تعطيل تحذيرات TensorFlow غير الضرورية
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# إعداد RTC Configuration مع خوادم STUN وTURN متعددة
RTC_CONFIGURATION = RTCConfiguration(
    {
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
            {"urls": ["stun:stun1.l.google.com:19302"]}
        ]
    }
)

# إنشاء كائن الكشف عن المشاعر
detector = FER(mtcnn=False)  # استخدام كاشف الوجوه الافتراضي

# تعريف فئة لمعالجة الإطارات
class EmotionDetector(VideoProcessorBase):
    def __init__(self):
        self.detector = detector

    def recv(self, frame):
        # تحويل الإطار إلى صيغة OpenCV (BGR)
        img = frame.to_ndarray(format="bgr24")
        img = cv2.resize(img, (640, 480))  # تقليل الدقة للأداء

        # الكشف عن المشاعر
        result = self.detector.detect_emotions(img)

        # معالجة النتائج
        for face in result:
            # استخراج إحداثيات الوجه
            x, y, w, h = face['box']
            
            # رسم مستطيل حول الوجه
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # استخراج المشاعر
            emotions = face['emotions']
            dominant_emotion = max(emotions, key=emotions.get)
            emotion_score = emotions[dominant_emotion]
            
            # عرض المشاعر على الإطار
            text = f"{dominant_emotion}: {emotion_score:.2f}"
            cv2.putText(img, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # إرجاع الإطار المعالج
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# إعداد واجهة Streamlit
st.title("Real-Time Emotion Detection")
st.write("This app detects emotions in real-time using your webcam.")

# إضافة مكون WebRTC
webrtc_streamer(
    key="emotion-detection",
    mode="sendrecv",
    rtc_configuration=RTC_CONFIGURATION,
    video_processor_factory=EmotionDetector,
    media_stream_constraints={"video": {"frameRate": 15}, "audio": False},
    async_processing=True,
    timeout=30
)
