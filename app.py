import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import cv2
from fer import FER
import numpy as np
import av
import queue
import time
import os

# إعدادات البيئة
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# تكوين WebRTC
RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
        {"urls": ["stun:stun2.l.google.com:19302"]}
    ]
})

class EmotionDetector(VideoProcessorBase):
    def __init__(self):
        super().__init__()
        self.detector = FER(mtcnn=True)  # استخدام MTCNN بدقة أعلى
        self.emotion_queue = queue.Queue()
        self.frame_skip = 2  # معالجة إطار من كل 3

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        img = cv2.resize(img, (640, 480))
        
        try:
            results = self.detector.detect_emotions(img)
            for result in results:
                x, y, w, h = result["box"]
                emotions = result["emotions"]
                dominant_emotion = max(emotions.items(), key=lambda x: x[1])
                
                # رسم النتائج على الإطار
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(
                    img, 
                    f"{dominant_emotion[0]}: {dominant_emotion[1]:.2f}", 
                    (x, y-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, (0, 255, 0), 2
                )
                
                self.emotion_queue.put({
                    "emotion": dominant_emotion[0],
                    "score": dominant_emotion[1],
                    "box": result["box"]
                })
                
        except Exception as e:
            st.error(f"Detection error: {str(e)}")
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")

def main():
    st.set_page_config(
        page_title="Real-Time Emotion Detection",
        page_icon="😊",
        layout="centered"
    )
    
    st.title("🎭 Real-Time Emotion Detection")
    st.caption("Powered by Streamlit, FER and OpenCV")
    
    with st.sidebar:
        st.header("⚙️ Settings")
        enable_detection = st.checkbox("Enable Detection", True)
        show_stats = st.checkbox("Show Performance Stats", False)
        st.markdown("---")
        st.info("For best results:")
        st.info("- Ensure good lighting")
        st.info("- Face the camera directly")
        st.info("- Remove glasses if possible")
    
    # منطقة عرض النتائج
    result_container = st.empty()
    stats_container = st.empty()
    
    ctx = webrtc_streamer(
        key="emotion-detector",
        video_processor_factory=EmotionDetector if enable_detection else None,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={
            "video": {"width": 640, "height": 480, "frameRate": 15},
            "audio": False
        },
        async_processing=True
    )
    
    # معالجة النتائج
    if ctx.video_processor:
        last_update = time.time()
        fps = 0
        frame_count = 0
        
        while True:
            try:
                # عرض النتائج
                if enable_detection:
                    try:
                        result = ctx.video_processor.emotion_queue.get(timeout=1.0)
                        with result_container.container():
                            st.success(f"**Detected Emotion:** {result['emotion']}")
                            st.metric("Confidence", f"{result['score']*100:.1f}%")
                    except queue.Empty:
                        pass
                
                # عرض إحصائيات الأداء
                if show_stats:
                    frame_count += 1
                    if time.time() - last_update >= 1.0:
                        fps = frame_count / (time.time() - last_update)
                        with stats_container.container():
                            st.caption(f"**Performance:** {fps:.1f} FPS")
                        frame_count = 0
                        last_update = time.time()
                        
            except Exception as e:
                st.error(f"Application error: {str(e)}")
                break

if __name__ == "__main__":
    main()
