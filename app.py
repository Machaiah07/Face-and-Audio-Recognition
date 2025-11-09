from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import os
import cv2
import numpy as np
import base64
from io import BytesIO
from PIL import Image as PILImage
from werkzeug.utils import secure_filename
from model_utils import predict_image, predict_audio, capture_from_webcam

UPLOAD_FOLDER = 'uploads'
ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg'}
ALLOWED_AUDIO_EXTENSIONS = {'wav', 'mp3', 'm4a'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename, allowed_ext):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_ext

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict_image', methods=['POST'])
def handle_image():
    if 'image' not in request.files:
        return redirect(request.url)
    file = request.files['image']
    if file.filename == '' or not allowed_file(file.filename, ALLOWED_IMAGE_EXTENSIONS):
        return "❌ Invalid image file."

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    image = cv2.imread(filepath)
    result = predict_image(image)

    # ✅ Convert file path to image URL
    image_url = url_for('uploaded_file', filename=filename)

    return render_template('index.html', prediction=result, image_url=image_url)

@app.route('/predict_audio', methods=['POST'])
def handle_audio():
    if 'audio' not in request.files:
        return redirect(request.url)
    file = request.files['audio']
    if file.filename == '' or not allowed_file(file.filename, ALLOWED_AUDIO_EXTENSIONS):
        return "❌ Invalid audio file."

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    result = predict_audio(filepath)
    return render_template('index.html', prediction=result)

@app.route('/capture_webcam', methods=['GET'])
def handle_webcam():
    image = capture_from_webcam()
    if image is not None:
        filename = 'webcam.jpg'
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        cv2.imwrite(filepath, image)

        result = predict_image(image)
        image_url = url_for('uploaded_file', filename=filename)

        return render_template('index.html', prediction=result, image_url=image_url)
    else:
        return render_template('index.html', prediction="❌ Failed to capture image from webcam.")

@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
