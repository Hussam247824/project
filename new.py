import streamlit as st
import os
import requests
import tempfile
import cv2
from PIL import Image, ImageDraw
from ultralytics import YOLO
import torch

# إعداد مسار حفظ الملفات
UPLOAD_FOLDER = 'uploads/'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# تحميل النماذج مرة واحدة فقط
if 'models' not in st.session_state:
    model_paths = [
        'https://github.com/Hussam247824/project/blob/master/best(3).pt',
        'https://github.com/Hussam247824/project/blob/master/best100epochs.pt',
        'https://github.com/Hussam247824/project/blob/master/best_v11_100epoch.pt'
    ]

    models = []
    for model_path in model_paths:
        try:
            model_name = model_path.split('/')[-1]
            model_local_path = os.path.join(UPLOAD_FOLDER, model_name)

            # تنزيل النموذج إذا لم يكن موجودًا محليًا
            if not os.path.exists(model_local_path):
                response = requests.get(model_path)
                if response.status_code == 200:
                    with open(model_local_path, 'wb') as f:
                        f.write(response.content)

            # تحميل النموذج
            model = YOLO(model_local_path).to('cuda' if torch.cuda.is_available() else 'cpu')
            models.append(model)
        except Exception as e:
            st.error(f"فشل في تحميل النموذج {model_path}: {e}")
            st.stop()

    # تخزين النماذج في session_state لتجنب تحميلها مرة أخرى
    st.session_state.models = models

# الصفحة الرئيسية لرفع الصور والفيديوهات وتحليلها
st.title("رفع صورة أو مقطع فيديو لتحليله باستخدام نماذج YOLOv8")

# زر لتشغيل الكاميرا
if st.button('تشغيل الكاميرا الأمامية'):
    # HTML5 و JavaScript لفتح الكاميرا وعرض الفيديو في الوقت الفعلي
    camera_html = """
    <html>
    <head>
        <style>
            video {
                width: 100%;
                border: 2px solid black;
            }
        </style>
    </head>
    <body>
        <h2>كاميرا الويب</h2>
        <video id="video" autoplay></video>
        <script>
            const video = document.getElementById('video');
            navigator.mediaDevices.getUserMedia({ video: true })
                .then(stream => {
                    video.srcObject = stream;
                    // هنا يمكنك إضافة الكود لتمرير الفيديو عبر النماذج بعد التقاطه
                })
                .catch(err => {
                    console.log("حدث خطأ: " + err);
                    alert("فشل في الوصول إلى الكاميرا.");
                });
        </script>
    </body>
    </html>
    """
    st.components.v1.html(camera_html, height=500)
