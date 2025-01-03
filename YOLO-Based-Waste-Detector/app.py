import streamlit as st
import cv2
from ultralytics import YOLO
import tempfile
import os
import logging
import time
import numpy as np
import cloudinary
import cloudinary.api
import requests

# Suppress warnings and logging from the YOLO model
logging.getLogger('ultralytics').setLevel(logging.ERROR)

# Cloudinary configuration
cloudinary.config(
    cloud_name='djelnepic',
    api_key='478999451717814',
    api_secret='C_9m-GBrzfAspRuUacCRnkMgFwg'
)

# Load YOLO model
model_path = "runs/detect/train/weights/best.pt"
model = YOLO(model_path)

def area_calc(x1, y1, x2, y2):
    length = abs(x1 - x2)
    width = abs(y1 - y2)
    return length * width

def get_latest_resource(resource_type='image'):
    """Fetches the latest resource (image or video) from Cloudinary."""
    try:
        response = cloudinary.api.resources(
            resource_type=resource_type,
            type='upload',
            max_results=1,
            sort_by=[{'uploaded_at': 'desc'}]  # Sort by most recent
        )
        if response['resources']:
            return response['resources'][0]['public_id']
        else:
            st.error("No resources found in Cloudinary.")
            return None
    except Exception as e:
        st.error(f"Error fetching latest resource: {e}")
        return None

def download_from_cloudinary(public_id, resource_type):
    """Downloads a file from Cloudinary."""
    try:
        response = cloudinary.utils.cloudinary_url(
            public_id,
            resource_type=resource_type,
            format='jpg' if resource_type == 'image' else 'mp4',
            fetch_format='auto'
        )
        return response[0]
    except Exception as e:
        st.error(f"Failed to fetch the file from Cloudinary: {e}")
        return None

def process_frame(frame):
    height, width = frame.shape[:2]
    new_size = (int(width * 0.5), int(height * 0.5))
    resized_frame = cv2.resize(frame, new_size)
    r_img = cv2.resize(resized_frame, (640, 640))
    results = model(r_img)
    area = 0
    if results and results[0].boxes is not None:
        boxes_list = results[0].boxes.data.tolist()
        for box in boxes_list:
            x1, y1, x2, y2, score, class_id = box
            cv2.rectangle(r_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            area += area_calc(x1, y1, x2, y2)

    return r_img, area

st.title('Waste Detection in Image/Video')

option = st.selectbox('Select Input Type:', ('Image', 'Video'))

if option == 'Image':
    public_id = get_latest_resource(resource_type='image')
    if public_id:
        url = download_from_cloudinary(public_id, resource_type='image')
        if url:
            file_bytes = np.asarray(bytearray(requests.get(url).content), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, 1)
            processed_img, total_area = process_frame(img)
            st.image(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB), caption='Processed Image with Detection', use_column_width=True)
            image_area = 640 * 640
            percentage_waste = round((total_area / image_area) * 100)
            st.write(f"Total Area of waste detected: {total_area} unit sq")
            st.write(f"Area of the image frame: {image_area} unit sq")
            st.write(f"The Percentage of Waste detected in the image is: {percentage_waste}%")

elif option == 'Video':
    public_id = get_latest_resource(resource_type='video')
    if public_id:
        url = download_from_cloudinary(public_id, resource_type='video')
        if url:
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            temp_file.write(requests.get(url).content)
            temp_file.close()
            cap = cv2.VideoCapture(temp_file.name)
            stframe = st.empty()
            total_area = 0
            total_frames = 0
            frame_count = 0
            frame_interval = 5
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_duration = 1 / fps

            image_area = 640 * 640
            total_waste_area = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count % frame_interval == 0:
                    processed_frame, waste_area = process_frame(frame)
                    total_waste_area += waste_area
                    total_frames += 1

                    stframe.image(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB), caption='Processed Video Frame with Detection', use_column_width=True)

                frame_count += 1
                time.sleep(frame_duration)

            cap.release()
            os.remove(temp_file.name)

            if total_frames > 0:
                average_waste_area = total_waste_area / total_frames
                percentage_waste = round((average_waste_area / image_area) * 100, 2)

                st.write(f"Total Waste Area detected in video: {total_waste_area} unit sq")
                st.write(f"Area of a single video frame: {image_area} unit sq")
                st.write(f"Average Waste Area per frame: {average_waste_area} unit sq")
                st.write(f"The Percentage of Waste detected in the video is: {percentage_waste}%")
