from flask import Flask, request, render_template, send_from_directory
import detectree as dtr
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Save uploaded file
        file = request.files['file']
        if file and file.filename:
            file_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(file_path)

            # Process the image
            processed_file_path, tree_percentage = process_image(file_path)

            # Return results
            return render_template('results.html',
                                   original_image=file.filename,
                                   processed_image=os.path.basename(processed_file_path),
                                   tree_percentage=tree_percentage)

    return render_template('index.html')

def process_image(file_path):
    # Use detectree to process the image
    y_pred = dtr.Classifier().predict_img(file_path)

    # Calculate the percentage of detected trees
    tree_percentage = calculate_tree_percentage(y_pred)

    # Save the processed image
    processed_image_path = os.path.join(PROCESSED_FOLDER, 'processed_' + os.path.basename(file_path))
    plt.imshow(y_pred, cmap='gray')  # Use gray colormap for binary masks
    plt.axis('off')
    plt.savefig(processed_image_path, bbox_inches='tight', pad_inches=0)
    plt.close()

    return processed_image_path, tree_percentage

def calculate_tree_percentage(y_pred):
    # Convert y_pred to a NumPy array if it's not already
    if isinstance(y_pred, Image.Image):
        y_pred = np.array(y_pred)

    # Count white pixels (value of 255 for grayscale images)
    total_pixels = y_pred.size
    white_pixels = np.sum(y_pred == 255)  # White pixels are often represented as 255 in grayscale
    tree_percentage = (white_pixels / total_pixels) * 100

    return tree_percentage

@app.route('/uploads/<filename>')
def upload_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/processed/<filename>')
def processed_file(filename):
    return send_from_directory(PROCESSED_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=True)
