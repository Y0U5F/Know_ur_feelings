import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import cv2
import numpy as np
import torch
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import av
import time
import os

# تعطيل تحذيرات TensorFlow
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# تعيين الجهاز
device = torch.device('cpu')

# إعداد صفحة Streamlit
st.set_page_config(
    page_title="كشف المشاعر في الوقت الفعلي",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# تحميل CSS
with open("static/css/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# تحميل HTML
with open("templates/index.html") as f:
    html_content = f.read()

# شريط جانبي
with st.sidebar:
    st.title("معلومات التطبيق")
    st.markdown("""
    ### تعليمات الاستخدام:
    1. انقر على زر "بدء الكشف عن المشاعر" لبدء الكشف
    2. انتظر تحميل النموذج (قد يستغرق بضع دقائق)
    3. انقر على زر "إيقاف الكشف عن المشاعر" لإيقاف الكشف
    4. تأكد من أن وجهك واضح في الكاميرا

    ### ملاحظات مهمة:
    - تأكد من السماح للكاميرا في المتصفح
    - استخدم كاميرا أمامية للهاتف
    - تأكد من وجود إضاءة جيدة
    - حافظ على مسافة مناسبة من الكاميرا
    - تحميل النموذج يتم مرة واحدة فقط
    """)

# عنوان التطبيق
st.title("كشف المشاعر في الوقت الفعلي")

# إدارة حالة التطبيق
if 'is_running' not in st.session_state:
    st.session_state.is_running = False
    st.session_state.model = None
    st.session_state.face_cascade = None
    st.session_state.last_emotion = None
    st.session_state.last_score = None
    st.session_state.model_loaded = False

# زر البدء
if st.button("بدء الكشف عن المشاعر" if not st.session_state.is_running else "إيقاف الكشف عن المشاعر"):
    st.session_state.is_running = not st.session_state.is_running
    if not st.session_state.is_running:
        st.session_state.model = None
        st.session_state.face_cascade = None
        st.session_state.last_emotion = None
        st.session_state.last_score = None
        st.session_state.model_loaded = False

# منطقة عرض النتائج
result_placeholder = st.empty()
loading_placeholder = st.empty()

# معالج الفيديو لـ WebRTC
class EmotionProcessor(VideoProcessorBase):
    def __init__(self):
        self.model = None
        self.face_cascade = None
        self.emotion_labels = ['غاضب', 'مشمئز', 'خائف', 'سعيد', 'حزين', 'مندهش', 'محايد']
        self.model_loaded = False

    def load_model(self):
        if not self.model_loaded:
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            self.model = load_model('emotion_model.h5')
            self.model_loaded = True

    def recv(self, frame):
        if not self.model_loaded:
            self.load_model()

        img = frame.to_ndarray(format="bgr")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) > 0:
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                roi_gray = gray[y:y+h, x:x+w]
                roi_gray = cv2.resize(roi_gray, (48, 48))
                roi = roi_gray.astype('float') / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)

                predictions = self.model.predict(roi)[0]
                emotion = self.emotion_labels[np.argmax(predictions)]
                score = np.max(predictions)

                st.session_state.last_emotion = emotion
                st.session_state.last_score = score

                text = f"{emotion}: {score:.2f}"
                cv2.putText(img, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# تشغيل WebRTC
if st.session_state.is_running:
    try:
        if not st.session_state.model_loaded:
            with loading_placeholder.container():
                st.markdown(html_content.split("</body>")[0] + "</body></html>", unsafe_allow_html=True)
                progress_bar = st.progress(0)
                for i in range(100):
                    time.sleep(0.05)
                    progress_bar.progress(i + 1)
                st.session_state.model_loaded = True
                loading_placeholder.empty()
                st.success("تم تحميل النموذج بنجاح!")

        webrtc_streamer(
            key="emotion-detection",
            video_processor_factory=EmotionProcessor,
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            media_stream_constraints={"video": True, "audio": False}
        )

        if st.session_state.last_emotion and st.session_state.last_score:
            with result_placeholder.container():
                st.markdown(
                    html_content.replace(
                        "{emotion}", st.session_state.last_emotion
                    ).replace(
                        "{score}", f"{st.session_state.last_score * 100:.2f}"
                    ),
                    unsafe_allow_html=True
                )
        else:
            result_placeholder.warning("لم يتم اكتشاف أي وجه. تأكد من أن وجهك واضح في الكاميرا.")

    except Exception as e:
        st.error(f"خطأ في معالجة الفيديو: {str(e)}")
        st.session_state.is_running = False
        st.session_state.model = None
        st.session_state.model_loaded = False
else:
    result_placeholder.empty()
    loading_placeholder.empty()
    st.markdown(html_content, unsafe_allow_html=True)
