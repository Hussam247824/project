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

# تحديد المسار لحفظ الملفات المرفوعة
UPLOAD_FOLDER = 'uploads/'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# دالة لتحليل الفيديو باستخدام النماذج الثلاثة
def analyze_video(video_path, output_video_path):
    device = 'cuda' if torch and torch.cuda.is_available() else 'cpu'
    
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception("فشل في فتح ملف الفيديو.")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        try:
            annotated_frame = frame
            # تمرير الفيديو عبر النماذج الثلاثة
            for model in yolo_models:
                results = model.predict(frame, device=device, conf=0.5, verbose=False)
                
                annotated_frame = np.array(results[0].plot())

            out.write(annotated_frame)
            frame_count += 1
        except Exception as e:
            st.error(f"خطأ في معالجة الإطار {frame_count}: {e}")
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

# دالة لتحليل الصور باستخدام النماذج الثلاثة
def analyze_image(image_path, output_image_path):
    device = 'cuda' if torch and torch.cuda.is_available() else 'cpu'
    
    
    image = cv2.imread(image_path)
    if image is None:
        raise Exception("فشل في فتح ملف الصورة.")

    try:
        annotated_image = image
        # تمرير الصورة عبر النماذج الثلاثة
        for model in yolo_models:
            results = model.predict(image, device=device, conf=0.5, verbose=False)
            
            annotated_image = np.array(results[0].plot())

        # حفظ الصورة المعالجة
        cv2.imwrite(output_image_path, annotated_image)
    except Exception as e:
        st.error(f"خطأ في معالجة الصورة: {e}")

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
        model_path = os.path.join(UPLOAD_FOLDER, model_name)
        
        # تنزيل النموذج إذا لم يكن موجودًا محليًا
        if not os.path.exists(model_path):
            try:
                
                response = requests.get(url)
                if response.status_code == 200:
                    with open(model_path, 'wb') as f:
                        f.write(response.content)
                    
                else:
                    st.error(f"فشل في تنزيل النموذج {model_name}: حالة الاستجابة {response.status_code}")
                    continue
            except Exception as e:
                st.error(f"فشل في تنزيل النموذج {model_name}: {e}")
                continue

        # تحميل النموذج
        try:
            from ultralytics import YOLO
            model = YOLO(model_path).to('cuda' if torch.cuda.is_available() else 'cpu')
            yolo_models.append(model)
            
        except Exception as e:
            st.error(f"فشل في تحميل النموذج {model_name}: {e}")

# الصفحة الرئيسية لرفع الفيديو أو الصور
st.title("رفع فيديو أو صورة لتحليلها")

# نموذج لرفع الفيديو أو الصورة
uploaded_file = st.file_uploader("اختر فيديو أو صورة لرفعها وتحليلها", type=["mp4", "avi", "mov", "mkv", "jpg", "jpeg", "png"])

# معالجة رفع الفيديو أو الصورة وتحليلها
if uploaded_file is not None:
    mime_type, _ = mimetypes.guess_type(uploaded_file.name)
    if not mime_type:
        st.error("يرجى رفع ملف صحيح.")
    elif mime_type.startswith('video'):
        # حفظ الفيديو المرفوع في مجلد مؤقت
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.read())
            video_path = tmp_file.name

        st.video(uploaded_file)
        

        # تحليل الفيديو إذا كانت النماذج محملة وcv2 وnumpy متاحة
        if yolo_models and cv2_available and numpy_available:
            try:
                output_video_path = os.path.join(UPLOAD_FOLDER, 'output_video.mp4')
                analyze_video(video_path, output_video_path)
                st.success("تم تحليل الفيديو بنجاح!")
                st.video(output_video_path)
            except Exception as e:
                st.error(f"حدث خطأ أثناء تحليل الفيديو: {e}")
        elif not cv2_available:
            st.error("مكتبة OpenCV غير متاحة، لا يمكن تحليل الفيديو.")
        elif not numpy_available:
            st.error("مكتبة numpy غير متاحة، لا يمكن تحليل الفيديو.")
        else:
            st.error("فشل في تحميل النماذج، يرجى المحاولة لاحقاً.")
    elif mime_type.startswith('image'):
        # حفظ الصورة المرفوعة في مجلد مؤقت
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            tmp_file.write(uploaded_file.read())
            image_path = tmp_file.name

        st.image(uploaded_file)
        

        # تحليل الصورة إذا كانت النماذج محملة وcv2 وnumpy متاحة
        if yolo_models and cv2_available and numpy_available:
            try:
                output_image_path = os.path.join(UPLOAD_FOLDER, 'output_image.jpg')
                analyze_image(image_path, output_image_path)
                st.success("تم تحليل الصورة بنجاح!")
                st.image(output_image_path)
            except Exception as e:
                st.error(f"حدث خطأ أثناء تحليل الصورة: {e}")
        elif not cv2_available:
            st.error("مكتبة OpenCV غير متاحة، لا يمكن تحليل الصورة.")
        elif not numpy_available:
            st.error("مكتبة numpy غير متاحة، لا يمكن تحليل الصورة.")
        else:
            st.error("فشل في تحميل النماذج، يرجى المحاولة لاحقاً.")

# تأكد من أن المكتبات النظامية المطلوبة مثبتة
st.info("إذا استمرت المشكلة، يرجى التأكد من أن مكتبات النظام مثل libgl1-mesa-glx مثبتة بشكل صحيح باستخدام ملف apt.txt.")
