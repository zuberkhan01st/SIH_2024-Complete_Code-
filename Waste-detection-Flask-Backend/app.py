import os
import cv2
import tempfile
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from ultralytics import YOLO
import logging

# Suppress warnings and logging from the YOLO model
logging.getLogger('ultralytics').setLevel(logging.ERROR)

app = Flask(__name__)
CORS(app)

IMAGE_DIR = 'images' 
VIDEO_DIR = 'videos'  

os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(VIDEO_DIR, exist_ok=True)

model_path = "runs/detect/train/weights/last.pt"
model = YOLO(model_path)

def area_calc(x1, y1, x2, y2):
    length = abs(x1 - x2)
    width = abs(y1 - y2)
    return length * width

def process_frame(frame):
    height, width = frame.shape[:2]
    new_size = (int(width * 0.5), int(height * 0.5))
    resized_frame = cv2.resize(frame, new_size)
    r_img = cv2.resize(resized_frame, (640, 640))
    results = model(r_img)
    area = 0
    boxes = []

    if results and results[0].boxes is not None:
        boxes_list = results[0].boxes.data.tolist()
        for box in boxes_list:
            x1, y1, x2, y2, score, class_id = box
            cv2.rectangle(r_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            area += area_calc(x1, y1, x2, y2)
            boxes.append({
                'x1': int(x1),
                'y1': int(y1),
                'x2': int(x2),
                'y2': int(y2),
                'score': score,
                'class_id': class_id
            })

    return r_img, area, boxes

@app.route('/detect/image', methods=['POST'])
def detect_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file uploaded'}), 400

    uploaded_image = request.files['image']
    img_bytes = np.frombuffer(uploaded_image.read(), np.uint8)
    img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)

    processed_img, total_area, boxes = process_frame(img)

    # Save the processed image temporarily
    temp_file_path = os.path.join(IMAGE_DIR, 'processed_image.jpg')
    cv2.imwrite(temp_file_path, processed_img)

    # Calculate waste percentage
    image_area = 640 * 640
    percentage_waste = round((total_area / image_area) * 100, 2)

    response = {
        'total_waste_area': total_area,
        'image_area': image_area,
        'percentage_waste': percentage_waste,
        'boxes': boxes,
        'processed_image_path': 'processed_image.jpg'  
    }

    return jsonify(response)

@app.route('/detect/video', methods=['POST'])
def detect_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file uploaded'}), 400

    uploaded_video = request.files['video']
    temp_file_path = os.path.join(VIDEO_DIR, 'uploaded_video.mp4')
    uploaded_video.save(temp_file_path)

    cap = cv2.VideoCapture(temp_file_path)
    total_waste_area = 0
    total_frames = 0
    frame_count = 0
    frame_interval = 5
    image_area = 640 * 640

    processed_video_path = os.path.join(VIDEO_DIR, 'processed_video.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(processed_video_path, fourcc, 20.0, (640, 640))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            processed_frame, waste_area, _ = process_frame(frame)
            total_waste_area += waste_area
            total_frames += 1
            out.write(processed_frame)

        frame_count += 1

    cap.release()
    out.release()

    if total_frames > 0:
        average_waste_area = total_waste_area / total_frames
        percentage_waste = round((average_waste_area / image_area) * 100, 2)
    else:
        percentage_waste = 0

    response = {
        'total_waste_area': total_waste_area,
        'average_waste_area_per_frame': total_waste_area / total_frames if total_frames > 0 else 0,
        'percentage_waste': percentage_waste,
        'processed_video_path': 'processed_video.mp4'  # Relative path
    }

    return jsonify(response)

@app.route('/download/image', methods=['GET'])
def download_image():
    image_path = request.args.get('image_path')
    if image_path and os.path.exists(os.path.join(IMAGE_DIR, image_path)):
        return send_from_directory(IMAGE_DIR, image_path, as_attachment=True)
    return jsonify({'error': 'Image not found'}), 404

# Flask route to download processed video
@app.route('/download/video', methods=['GET'])
def download_video():
    video_path = request.args.get('video_path')
    if video_path and os.path.exists(os.path.join(VIDEO_DIR, video_path)):
        return send_from_directory(VIDEO_DIR, video_path, mimetype='video/mp4')
    return jsonify({'error': 'Video not found'}), 404

if __name__ == '__main__':
    app.run(debug=True)