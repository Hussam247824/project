from flask import Flask, request, render_template, redirect, url_for, jsonify, send_file, Response
import os
import cv2
import mimetypes
import time
import socket

try:
    import torch
    from ultralytics import YOLO
except ModuleNotFoundError:
    torch = None
    YOLO = None
    print("Warning: torch or ultralytics not found. Some functionalities will be disabled.")

# تهيئة التطبيق
app = Flask(__name__)

# تحديد المسار لحفظ الملفات المرفوعة
UPLOAD_FOLDER = 'uploads/'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# تحميل نماذج YOLOv8 المدربة (إذا كان torch متاحًا)
if torch and YOLO:
    model_paths = [
        r'hh44/project/best(3).pt',
        r'hh44/project/best_v11_100epoch.pt',
        r'hh44/project/best100epochs.pt'
    ]
    
    yolo_models = []
    for path in model_paths:
        if os.path.exists(path):
            try:
                model = YOLO(path).to('cuda' if torch.cuda.is_available() else 'cpu')
                yolo_models.append(model)
                print(f"Model loaded successfully: {path}")
            except Exception as e:
                print(f"Failed to load model {path}: {e}")
        else:
            print(f"Model file not found: {path}")
else:
    yolo_models = []
    print("YOLO or torch not found, models will not load.")

# الصفحة الرئيسية لرفع الفيديو أو تشغيل الكاميرا
@app.route('/')
def index():
    return '''
    <!doctype html>
    <html lang="ar">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>رفع فيديو أو تشغيل الكاميرا</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                background-color: #f4f4f9;
                margin: 0;
                padding: 0;
                display: flex;
                justify-content: center;
                align-items: center;
                height: 100vh;
            }
            .container {
                text-align: center;
                background-color: #fff;
                padding: 30px;
                border-radius: 8px;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
                width: 70%;
                max-width: 600px;
            }
            h1 {
                color: #333;
                font-size: 24px;
                margin-bottom: 20px;
            }
            form {
                margin-bottom: 30px;
            }
            input[type="file"], input[type="submit"] {
                padding: 10px 20px;
                font-size: 16px;
                margin: 10px;
                border: 1px solid #ddd;
                border-radius: 4px;
                cursor: pointer;
            }
            input[type="submit"] {
                background-color: #4CAF50;
                color: white;
                border: none;
            }
            input[type="submit"]:hover {
                background-color: #45a049;
            }
            .separator {
                margin: 20px 0;
                border-top: 1px solid #ddd;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>رفع فيديو لتحليله أو تشغيل الكاميرا</h1>
            
            <!-- Form to upload video -->
            <form action="/upload" method="post" enctype="multipart/form-data">
                <input type="file" name="video" accept="video/*">
                <input type="submit" value="رفع الفيديو">
            </form>

            <div class="separator"></div>

            <!-- Form to start webcam -->
            <form action="/webcam" method="get">
                <input type="submit" value="تشغيل الكاميرا">
            </form>
        </div>
    </body>
    </html>
    '''

# معالجة رفع الفيديو وتحليله
@app.route('/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        print("No video part in the request.")
        return redirect(url_for('error_page', message='لم يتم العثور على ملف الفيديو في الطلب.'))

    video = request.files['video']
    mime_type, _ = mimetypes.guess_type(video.filename)

    if not mime_type or not mime_type.startswith('video'):
        print("Invalid file type.")
        return redirect(url_for('error_page', message='يرجى رفع ملف فيديو صحيح.'))

    video_path = os.path.join(app.config['UPLOAD_FOLDER'], video.filename)
    video.save(video_path)
    print(f"Video saved to: {video_path}")

    if yolo_models:
        try:
            output_video_path = analyze_video(video_path)
            print(f"Video analysis completed: {output_video_path}")
            return render_template('result.html', video_url=url_for('uploaded_file', filename=os.path.basename(output_video_path)))
        except Exception as e:
            print(f"Error during video analysis: {e}")
            return redirect(url_for('error_page', message=f'حدث خطأ أثناء تحليل الفيديو: {e}'))
    else:
        print("Models not loaded.")
        return redirect(url_for('error_page', message='فشل في تحميل النماذج، يرجى المحاولة لاحقاً.'))

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename))

# صفحة الخطأ
@app.route('/error')
def error_page():
    message = request.args.get('message', 'حدث خطأ غير معروف.')
    return '''
    <!doctype html>
    <title>خطأ</title>
    <h1>خطأ</h1>
    <p>{}</p>
    <a href="/">العودة إلى الصفحة الرئيسية</a>
    '''.format(message)

# دالة لتحليل الفيديو باستخدام النماذج الثلاثة
def analyze_video(video_path):
    device = 'cuda' if torch and torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception("Failed to open video file.")

    # تغيير مسار حفظ الفيديو الناتج
    temp_dir = r'hh44/project/videos'
    os.makedirs(temp_dir, exist_ok=True)
    output_video_path = os.path.join(temp_dir, 'output_video.mp4')
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
                print(f"Prediction results for frame {frame_count}: {results}")
                annotated_frame = results[0].plot()

            out.write(annotated_frame)
            frame_count += 1
            print(f"Processed frame {frame_count}")
        except Exception as e:
            print(f"Error processing frame {frame_count}: {e}")
            raise Exception(f"Error processing frame {frame_count}: {e}")

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    return output_video_path

# تشغيل الكاميرا الأمامية وتحليل الفيديو مباشرة
@app.route('/webcam', methods=['GET'])
def webcam_feed():
    return Response(webcam_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

def webcam_stream():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Failed to open webcam.")
        return Response("فشل في فتح الكاميرا. تأكد من أن الكاميرا متصلة وليست مستخدمة بواسطة تطبيق آخر.", status=400)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        start_time = time.time()

        try:
            annotated_frame = frame
            for model in yolo_models:
                results = model.predict(frame, device='cuda' if torch and torch.cuda.is_available() else 'cpu', conf=0.4, verbose=False)
                print(f"Prediction results for webcam frame: {results}")
                annotated_frame = results[0].plot()

            ret, buffer = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

            elapsed_time = time.time() - start_time
            time.sleep(max(0, (1 / 30) - elapsed_time))
        except Exception as e:
            print(f"Error processing webcam frame: {e}")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    port = 8001
    while True:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(('localhost', port)) != 0:
                break
            port += 1
    print(f"Starting Flask server on port {port}...")
    app.run(host="0.0.0.0", port=port, debug=False)
