import streamlit as st
import cv2
from fer import FER
import numpy as np
import torch
import matplotlib.pyplot as plt
import io
import base64
from PIL import Image

# تعيين الجهاز المستخدم (CPU أو GPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# إعداد صفحة Streamlit
st.set_page_config(
    page_title="كشف المشاعر في الوقت الفعلي",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# تنسيق CSS مخصص
st.markdown("""
    <style>
    .stApp {
        max-width: 100%;
        margin: 0 auto;
    }
    .video-container {
        position: relative;
        width: 100%;
        max-width: 800px;
        margin: 0 auto;
    }
    .emotion-info {
        position: absolute;
        top: 10px;
        right: 10px;
        background-color: rgba(0, 0, 0, 0.7);
        color: white;
        padding: 10px;
        border-radius: 5px;
        font-size: 14px;
    }
    .start-button {
        background-color: #4CAF50;
        color: white;
        padding: 15px 30px;
        border: none;
        border-radius: 5px;
        font-size: 18px;
        cursor: pointer;
        transition: all 0.3s ease;
        margin-bottom: 20px;
        width: 100%;
        max-width: 300px;
    }
    .start-button:hover {
        background-color: #45a049;
    }
    .start-button.active {
        background-color: #f44336;
    }
    @media (max-width: 600px) {
        .video-container {
            width: 100%;
        }
        .start-button {
            width: 100%;
            padding: 12px 24px;
            font-size: 16px;
        }
        .stButton > button {
            width: 100%;
        }
    }
    </style>
""", unsafe_allow_html=True)

# إضافة شريط جانبي للمعلومات
with st.sidebar:
    st.title("معلومات التطبيق")
    st.markdown("""
    ### تعليمات الاستخدام:
    1. انقر على زر "بدء الكشف عن المشاعر" لبدء الكشف
    2. انقر على زر "إيقاف الكشف عن المشاعر" لإيقاف الكشف
    3. تأكد من أن وجهك واضح في الكاميرا
    4. يمكنك استخدام التطبيق على الهاتف المحمول

    ### ملاحظات مهمة:
    - تأكد من السماح للكاميرا في المتصفح
    - استخدم كاميرا أمامية للهاتف
    - تأكد من وجود إضاءة جيدة
    """)

# عنوان التطبيق
st.title("كشف المشاعر في الوقت الفعلي")

# زر البدء
if 'is_running' not in st.session_state:
    st.session_state.is_running = False
    st.session_state.detector = None

if st.button("بدء الكشف عن المشاعر" if not st.session_state.is_running else "إيقاف الكشف عن المشاعر"):
    st.session_state.is_running = not st.session_state.is_running
    if not st.session_state.is_running:
        st.session_state.detector = None

# منطقة عرض الفيديو
video_placeholder = st.empty()

# فتح الكاميرا
try:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.warning("""
        ⚠️ لا يمكن الوصول إلى الكاميرا. يرجى التأكد من:
        1. السماح للكاميرا في المتصفح
        2. استخدام متصفح يدعم الوصول إلى الكاميرا
        3. التأكد من أن الكاميرا متصلة وتعمل بشكل صحيح
        
        يمكنك تجربة التطبيق على جهازك المحلي أو استخدام هاتفك المحمول.
        """)
        st.session_state.is_running = False
except Exception as e:
    st.error(f"خطأ في الوصول إلى الكاميرا: {str(e)}")
    st.session_state.is_running = False

if st.session_state.is_running:
    try:
        # إنشاء كائن الكشف عن المشاعر فقط عند بدء التشغيل
        if st.session_state.detector is None:
            st.session_state.detector = FER(mtcnn=True)
            
        while st.session_state.is_running:
            ret, frame = cap.read()
            if not ret:
                st.error("خطأ: لا يمكن قراءة الإطار من الكاميرا")
                st.session_state.is_running = False
                break

            try:
                # تحويل الإطار إلى مصفوفة NumPy
                frame_np = np.array(frame, dtype=np.uint8)
                
                # الكشف عن المشاعر
                result = st.session_state.detector.detect_emotions(frame_np)
                
                # معالجة النتائج
                for face in result:
                    x, y, w, h = face['box']
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    emotions = face['emotions']
                    dominant_emotion = max(emotions, key=emotions.get)
                    emotion_score = emotions[dominant_emotion]
                    text = f"{dominant_emotion}: {emotion_score:.2f}"
                    cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                # تحويل الإطار إلى صيغة RGB لعرضه في Streamlit
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

            except Exception as e:
                st.error(f"خطأ في معالجة الإطار: {str(e)}")
                continue

    except Exception as e:
        st.error(f"خطأ في تهيئة نموذج الكشف عن المشاعر: {str(e)}")
        st.session_state.is_running = False
        st.session_state.detector = None
    finally:
        if 'cap' in locals() and cap.isOpened():
            cap.release()
else:
    video_placeholder.empty()
    if 'cap' in locals() and cap.isOpened():
        cap.release()

# إضافة معلومات إضافية في الأسفل
st.markdown("""
    ### دعم الهواتف المحمولة:
    - يدعم التطبيق جميع المتصفحات الحديثة على الهواتف
    - متوافق مع كاميرات الهواتف الأمامية والخلفية
    - واجهة مستخدم متجاوبة مع جميع أحجام الشاشات
    - يعمل على iOS و Android
    
    ### ملاحظة مهمة:
    - للتجربة على Streamlit Cloud، يرجى استخدام هاتفك المحمول
    - تأكد من السماح للكاميرا في متصفح هاتفك
    - استخدم كاميرا الهاتف الأمامية للحصول على أفضل النتائج
""")
