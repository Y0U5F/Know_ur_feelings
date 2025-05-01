import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import cv2
from fer import FER
import numpy as np
import av
import queue
from typing import Union, Optional

# تعطيل تحذيرات TensorFlow غير الضرورية
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# إعداد RTC Configuration مع خوادم STUN متعددة
RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
        {"urls": ["stun:stun2.l.google.com:19302"]}
    ],
    "iceTransportPolicy": "all"  # يمكن تغييرها إلى "relay" إذا كنت خلف NAT صارم
})

# تعريف فئة لمعالجة الإطارات مع تحسينات الأداء
class EmotionDetector(VideoProcessorBase):
    def __init__(self):
        super().__init__()
        self.detector = FER(mtcnn=False)
        self.emotion_queue = queue.Queue()
        self.frame_count = 0
        self.skip_frames = 2  # معالجة إطار واحد من كل 3 إطارات للأداء

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        self.frame_count += 1
        if self.frame_count % (self.skip_frames + 1) != 0:
            return frame

        img = frame.to_ndarray(format="bgr24")
        
        # تحسين الأداء بتقليل حجم الإطار
        img = cv2.resize(img, (640, 480))
        
        try:
            # الكشف عن المشاعر
            result = self.detector.detect_emotions(img)
            
            for face in result:
                x, y, w, h = face['box']
                emotions = face['emotions']
                dominant_emotion = max(emotions.items(), key=lambda x: x[1])
                
                # رسم المستطيل والنص
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                text = f"{dominant_emotion[0]}: {dominant_emotion[1]:.2f}"
                cv2.putText(img, text, (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # إرسال النتائج للواجهة
                self.emotion_queue.put({
                    'emotion': dominant_emotion[0],
                    'score': dominant_emotion[1],
                    'box': face['box']
                })
                
        except Exception as e:
            st.error(f"Error in emotion detection: {str(e)}")

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# إعداد واجهة Streamlit
def main():
    st.set_page_config(
        page_title="Real-Time Emotion Detection",
        page_icon="😊",
        layout="wide"
    )
    
    st.title("Real-Time Emotion Detection")
    st.markdown("""
    <style>
    .st-emotion-box {
        border: 2px solid #4CAF50;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.write("This app detects emotions in real-time using your webcam.")
    
    # إضافة معلومات جانبية
    with st.sidebar:
        st.header("Settings")
        detect_emotions = st.checkbox("Enable Emotion Detection", True)
        show_fps = st.checkbox("Show FPS", False)
        st.markdown("---")
        st.info("Make sure your face is clearly visible in the camera.")
    
    # تشغيل كاميرا الويب
    ctx = webrtc_streamer(
        key="emotion-detection",
        video_processor_factory=EmotionDetector if detect_emotions else None,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={
            "video": {
                "width": {"ideal": 640},
                "height": {"ideal": 480},
                "frameRate": {"ideal": 15}
            },
            "audio": False
        },
        async_processing=True,
        desired_playing_state=True
    )
    
    # عرض النتائج
    if ctx.video_processor:
        result_placeholder = st.empty()
        fps_placeholder = st.empty() if show_fps else None
        last_time = time.time()
        frame_count = 0
        
        while True:
            try:
                if detect_emotions:
                    result = ctx.video_processor.emotion_queue.get(timeout=1.0)
                    with result_placeholder.container():
                        st.markdown(f"""
                        <div class="st-emotion-box">
                            <h3>Detected Emotion: <span style="color:#4CAF50">{result['emotion']}</span></h3>
                            <p>Confidence: <strong>{result['score']:.2f}</strong></p>
                        </div>
                        """, unsafe_allow_html=True)
                
                if show_fps:
                    frame_count += 1
                    current_time = time.time()
                    if current_time - last_time >= 1.0:
                        fps = frame_count / (current_time - last_time)
                        fps_placeholder.write(f"FPS: {fps:.1f}")
                        frame_count = 0
                        last_time = current_time
                        
            except queue.Empty:
                continue
            except Exception as e:
                st.error(f"Error: {str(e)}")
                break

if __name__ == "__main__":
    import time
    main()
