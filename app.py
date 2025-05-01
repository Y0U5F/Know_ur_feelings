import streamlit as st
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import io
import base64
from PIL import Image
import time
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# تعطيل تحذيرات TensorFlow
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# تعيين الجهاز المستخدم (CPU)
device = torch.device('cpu')

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
    .emotion-display {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 5px;
        margin-top: 10px;
        text-align: center;
    }
    .emotion-text {
        font-size: 24px;
        font-weight: bold;
        color: #333;
    }
    .emotion-score {
        font-size: 18px;
        color: #666;
    }
    .loading-container {
        text-align: center;
        padding: 20px;
        background-color: #f8f9fa;
        border-radius: 5px;
        margin: 20px 0;
    }
    .loading-text {
        font-size: 18px;
        color: #333;
        margin-bottom: 10px;
    }
    .loading-subtext {
        font-size: 14px;
        color: #666;
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
        .emotion-text {
            font-size: 20px;
        }
        .emotion-score {
            font-size: 16px;
        }
        .loading-text {
            font-size: 16px;
        }
        .loading-subtext {
            font-size: 12px;
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
    2. انتظر حتى يتم تحميل النموذج (قد يستغرق بضع دقائق)
    3. انقر على زر "إيقاف الكشف عن المشاعر" لإيقاف الكشف
    4. تأكد من أن وجهك واضح في الكاميرا

    ### ملاحظات مهمة:
    - تأكد من السماح للكاميرا في المتصفح
    - استخدم كاميرا أمامية للهاتف
    - تأكد من وجود إضاءة جيدة
    - حافظ على مسافة مناسبة من الكاميرا
    - تحميل النموذج يتم مرة واحدة فقط عند بدء التشغيل
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

# منطقة عرض الفيديو والنتائج
video_placeholder = st.empty()
result_placeholder = st.empty()
loading_placeholder = st.empty()

if st.session_state.is_running:
    try:
        # تحميل النموذج فقط عند بدء التشغيل
        if st.session_state.model is None:
            with loading_placeholder.container():
                st.markdown("""
                    <div class="loading-container">
                        <div class="loading-text">جاري تحميل نموذج الكشف عن المشاعر...</div>
                        <div class="loading-subtext">قد يستغرق هذا بضع دقائق. يرجى الانتظار.</div>
                    </div>
                """, unsafe_allow_html=True)
                
                # شريط التقدم
                progress_bar = st.progress(0)
                
                # محاكاة التقدم
                for i in range(100):
                    time.sleep(0.05)  # محاكاة وقت التحميل
                    progress_bar.progress(i + 1)
                
                # تحميل نموذج الكشف عن الوجوه
                st.session_state.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                
                # تحميل نموذج المشاعر
                st.session_state.model = load_model('emotion_model.h5')
                st.session_state.model_loaded = True
                
                # إخفاء شريط التقدم
                loading_placeholder.empty()
                st.success("تم تحميل النموذج بنجاح!")
        
        if st.session_state.model_loaded:
            # استخدام ميزة الكاميرا المدمجة في Streamlit
            img_file_buffer = st.camera_input("كاميرا الكشف عن المشاعر")
            
            if img_file_buffer is not None:
                # قراءة الصورة من المخزن المؤقت
                bytes_data = img_file_buffer.getvalue()
                cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
                
                try:
                    # تحويل الصورة إلى تدرجات الرمادي
                    gray = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)
                    
                    # الكشف عن الوجوه
                    faces = st.session_state.face_cascade.detectMultiScale(gray, 1.3, 5)
                    
                    if len(faces) > 0:
                        for (x, y, w, h) in faces:
                            # رسم مستطيل حول الوجه
                            cv2.rectangle(cv2_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                            
                            # استخراج منطقة الوجه
                            roi_gray = gray[y:y+h, x:x+w]
                            roi_gray = cv2.resize(roi_gray, (48, 48))
                            roi = roi_gray.astype('float') / 255.0
                            roi = img_to_array(roi)
                            roi = np.expand_dims(roi, axis=0)
                            
                            # التنبؤ بالمشاعر
                            predictions = st.session_state.model.predict(roi)[0]
                            emotion_labels = ['غاضب', 'مشمئز', 'خائف', 'سعيد', 'حزين', 'مندهش', 'محايد']
                            emotion = emotion_labels[np.argmax(predictions)]
                            score = np.max(predictions)
                            
                            # تحديث حالة المشاعر
                            st.session_state.last_emotion = emotion
                            st.session_state.last_score = score
                            
                            # عرض النص على الصورة
                            text = f"{emotion}: {score:.2f}"
                            cv2.putText(cv2_img, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                        # تحويل الإطار إلى صيغة RGB لعرضه في Streamlit
                        frame_rgb = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
                        video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
                        
                        # عرض النتائج في منطقة منفصلة
                        with result_placeholder.container():
                            st.markdown("""
                                <div class="emotion-display">
                                    <div class="emotion-text">المشاعر المكتشفة</div>
                                    <div class="emotion-score">
                                        المشاعر السائدة: {} (ثقة: {:.2f}%)
                                    </div>
                                </div>
                            """.format(
                                st.session_state.last_emotion,
                                st.session_state.last_score * 100
                            ), unsafe_allow_html=True)
                    else:
                        result_placeholder.warning("لم يتم اكتشاف أي وجه في الصورة. يرجى التأكد من أن وجهك واضح في الكاميرا.")

                except Exception as e:
                    st.error(f"خطأ في معالجة الإطار: {str(e)}")
                    result_placeholder.empty()

    except Exception as e:
        st.error(f"خطأ في تهيئة نموذج الكشف عن المشاعر: {str(e)}")
        st.session_state.is_running = False
        st.session_state.model = None
        st.session_state.face_cascade = None
        st.session_state.model_loaded = False
else:
    video_placeholder.empty()
    result_placeholder.empty()
    loading_placeholder.empty()

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
    - حافظ على مسافة مناسبة من الكاميرا (حوالي 30-50 سم)
    - تحميل النموذج يتم مرة واحدة فقط عند بدء التشغيل
""")
