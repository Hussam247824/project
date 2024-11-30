import streamlit as st
import os
import torch
import tempfile
import mimetypes
import requests
import cv2

# تحديد المسار لحفظ الملفات المرفوعة
UPLOAD_FOLDER = 'uploads/'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# تحميل نماذج YOLOv8 المدربة من مستودع GitHub
model_urls = [
    'https://github.com/Hussam247824/project/raw/master/best(3).pt',
    'https://github.com/Hussam247824/project/raw/master/best_v11_100epoch.pt',
    'https://github.com/Hussam247824/project/raw/master/best100epochs.pt'
]

yolo_models = []
for url in model_urls:
    model_name = url.split('/')[-1]
    model_path = os.path.join(UPLOAD_FOLDER, model_name)
    
    # تنزيل النموذج إذا لم يكن موجودًا محليًا
    if not os.path.exists(model_path):
        try:
            st.info(f"جاري تنزيل النموذج: {model_name}")
            response = requests.get(url)
            if response.status_code == 200:
                with open(model_path, 'wb') as f:
                    f.write(response.content)
                st.success(f"تم تنزيل النموذج بنجاح: {model_name}")
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
        st.success(f"تم تحميل النموذج بنجاح: {model_name}")
    except Exception as e:
        st.error(f"فشل في تحميل النموذج {model_name}: {e}")

# الصفحة الرئيسية لرفع الفيديو أو تشغيل الكاميرا
st.title("رفع فيديو لتحليله أو تشغيل الكاميرا")

# نموذج لرفع الفيديو
uploaded_video = st.file_uploader("اختر فيديو لرفعه وتحليله", type=["mp4", "avi", "mov", "mkv"])

# معالجة رفع الفيديو وتحليله
if uploaded_video is not None:
    mime_type, _ = mimetypes.guess_type(uploaded_video.name)
    if not mime_type or not mime_type.startswith('video'):
        st.error("يرجى رفع ملف فيديو صحيح.")
    else:
        # حفظ الفيديو المرفوع في مجلد مؤقت
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_video.read())
            video_path = tmp_file.name

        st.video(uploaded_video)
        st.write(f"تم حفظ الفيديو في المسار: {video_path}")

        # تحليل الفيديو إذا كانت النماذج محملة
        if yolo_models:
            try:
                output_video_path = os.path.join(UPLOAD_FOLDER, 'output_video.mp4')
                analyze_video(video_path, output_video_path)
                st.success("تم تحليل الفيديو بنجاح!")
                st.video(output_video_path)
            except Exception as e:
                st.error(f"حدث خطأ أثناء تحليل الفيديو: {e}")
        else:
            st.error("فشل في تحميل النماذج، يرجى المحاولة لاحقاً.")

# دالة لتحليل الفيديو باستخدام النماذج الثلاثة
def analyze_video(video_path, output_video_path):
    device = 'cuda' if torch and torch.cuda.is_available() else 'cpu'
    st.write(f"باستخدام الجهاز: {device}")
    
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
                st.text(f"نتائج التنبؤ للإطار {frame_count}: {results}")
                annotated_frame = results[0].plot()

            out.write(annotated_frame)
            frame_count += 1
        except Exception as e:
            st.error(f"خطأ في معالجة الإطار {frame_count}: {e}")
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
