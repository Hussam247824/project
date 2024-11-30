import streamlit as st
import os
import torch
import tempfile
import mimetypes
import requests
import numpy as np

# استخدام opencv-python-headless لتجنب مشاكل libGL
cv2_available = False
try:
    import cv2
    cv2_available = True
except ImportError:
    st.warning("مكتبة 'opencv-python-headless' غير مثبتة بشكل صحيح. يرجى التأكد من تثبيتها عبر requirements.txt.")

# التحقق من مكتبة numpy
numpy_available = True
try:
    _ = np.array([1])
except ImportError as e:
    st.error("فشل في تحميل مكتبة numpy: يرجى التأكد من تثبيتها بشكل صحيح.")
    numpy_available = False

# دالة لتحليل الصور باستخدام النماذج الثلاثة
def analyze_image(image_path):
    device = 'cuda' if torch and torch.cuda.is_available() else 'cpu'
    
    image = cv2.imread(image_path)
    if image is None:
        raise Exception("فشل في فتح ملف الصورة.")

    try:
        annotated_image = image
        # تمرير الصورة عبر النماذج الثلاثة
        for model in yolo_models:
            results = model.predict(image, device=device, conf=0.5, verbose=False)
            for result in results:
                annotated_image = result.plot()

        return annotated_image
    except Exception as e:
        st.error(f"خطأ في معالجة الصورة: {e}")
        return None

# تحميل نماذج YOLOv8 المدربة من مستودع GitHub
model_urls = [
    'https://github.com/Hussam247824/project/raw/master/best(3).pt',
    'https://github.com/Hussam247824/project/raw/master/best_v11_100epoch.pt',
    'https://github.com/Hussam247824/project/raw/master/best100epochs.pt'
]

yolo_models = []
if numpy_available:
    for url in model_urls:
        model_name = url.split('/')[-1]
        model_path = os.path.join(tempfile.gettempdir(), model_name)
        
        # تنزيل النموذج إذا لم يكن موجودًا محليًا
        if not os.path.exists(model_path):
            try:
                response = requests.get(url)
                if response.status_code == 200:
                    with open(model_path, 'wb') as f:
                        f.write(response.content)
                else:
                    continue
            except Exception as e:
                continue

        # تحميل النموذج
        try:
            from ultralytics import YOLO
            model = YOLO(model_path).to('cuda' if torch.cuda.is_available() else 'cpu')
            yolo_models.append(model)
        except Exception as e:
            continue

# الصفحة الرئيسية لرفع الصور
st.title("رفع صورة لتحليلها")

# نموذج لرفع الصورة
uploaded_file = st.file_uploader("اختر صورة لرفعها وتحليلها", type=["jpg", "jpeg", "png"])

# معالجة رفع الصورة وتحليلها
if uploaded_file is not None:
    mime_type, _ = mimetypes.guess_type(uploaded_file.name)
    if not mime_type:
        st.error("يرجى رفع ملف صحيح.")
    elif mime_type.startswith('image'):
        # حفظ الصورة المرفوعة في مجلد مؤقت
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            tmp_file.write(uploaded_file.read())
            image_path = tmp_file.name

        # تحليل الصورة إذا كانت النماذج محملة وcv2 وnumpy متاحة
        if yolo_models and cv2_available and numpy_available:
            try:
                annotated_image = analyze_image(image_path)
                if annotated_image is not None:
                    st.success("تم تحليل الصورة بنجاح!")
                    st.image(annotated_image, channels="BGR")
            except Exception as e:
                st.error(f"حدث خطأ أثناء تحليل الصورة: {e}")
        elif not cv2_available:
            st.error("مكتبة OpenCV غير متاحة، لا يمكن تحليل الصورة.")
        elif not numpy_available:
            st.error("مكتبة numpy غير متاحة، لا يمكن تحليل الصورة.")
        else:
            st.error("فشل في تحميل النماذج، يرجى المحاولة لاحقاً.")
