import os
import torch
import numpy as np
import librosa
import cv2
import torchaudio
from PIL import Image
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from insightface.app import FaceAnalysis
from speechbrain.inference import SpeakerRecognition

# === Face Models ===
facenet = InceptionResnetV1(pretrained='vggface2').eval()
arcface = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
arcface.prepare(ctx_id=0)  # ✅ fixed the typo here

# === Audio Model ===
speaker_model = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="pretrained_ecapa")

# === Embedding Functions ===
def extract_arcface_embedding(image):
    faces = arcface.get(image)
    if not faces:
        return None
    return faces[0]['embedding']

def extract_facenet_embedding(image):
    img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    transform = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    img_tensor = transform(img).unsqueeze(0)
    return facenet(img_tensor).detach().numpy().flatten()

def extract_audio_embedding(file_path):
    waveform, sr = librosa.load(file_path, sr=16000)
    return speaker_model.encode_batch(torch.tensor(waveform).unsqueeze(0)).squeeze().detach().numpy()

# === Load Dataset Once and Train Classifiers ===
dataset_path = "/Users/divinmachaiah/Desktop/Dataset"
image_embeddings = []
image_labels = []
audio_embeddings = []
audio_labels = []

for person in os.listdir(dataset_path):
    person_path = os.path.join(dataset_path, person)
    if not os.path.isdir(person_path):
        continue

    picture_folder = os.path.join(person_path, 'Pictures')
    audio_folder = os.path.join(person_path, 'Audio')

    if not os.path.isdir(picture_folder) or not os.path.isdir(audio_folder):
        continue

    for img_file in os.listdir(picture_folder):
        if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        img_path = os.path.join(picture_folder, img_file)
        image = cv2.imread(img_path)
        if image is None:
            continue
        emb1 = extract_arcface_embedding(image)
        emb2 = extract_facenet_embedding(image)
        if emb1 is not None and emb2 is not None:
            combined = np.concatenate((emb1, emb2))
            image_embeddings.append(combined)
            image_labels.append(person)

    for audio_file in os.listdir(audio_folder):
        if not audio_file.lower().endswith(('.wav', '.mp3', '.m4a')):
            continue
        audio_path = os.path.join(audio_folder, audio_file)
        try:
            emb = extract_audio_embedding(audio_path)
            audio_embeddings.append(emb)
            audio_labels.append(person)
        except:
            continue

label_encoder = LabelEncoder()
y_face = label_encoder.fit_transform(image_labels)
y_audio = label_encoder.transform(audio_labels)

face_clf = SVC(kernel='linear', probability=True)
face_clf.fit(image_embeddings, y_face)

audio_clf = SVC(kernel='linear', probability=True)
audio_clf.fit(audio_embeddings, y_audio)

# === Predict Functions ===
def predict_image(image, threshold=0.7):
    emb1 = extract_arcface_embedding(image)
    emb2 = extract_facenet_embedding(image)
    if emb1 is None or emb2 is None:
        return "Face not detected"
    combined = np.concatenate((emb1, emb2)).reshape(1, -1)
    probs = face_clf.predict_proba(combined)
    max_prob = np.max(probs)
    pred = np.argmax(probs)
    return label_encoder.inverse_transform([pred])[0] if max_prob >= threshold else "Person not identified"

def predict_audio(file_path, threshold=0.7):
    emb = extract_audio_embedding(file_path).reshape(1, -1)
    probs = audio_clf.predict_proba(emb)
    max_prob = np.max(probs)
    pred = np.argmax(probs)
    return label_encoder.inverse_transform([pred])[0] if max_prob >= threshold else "Person not identified"

# === Webcam Capture Function ===
def capture_from_webcam():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not access webcam.")
        return None
    print("📸 Capturing image in 3 seconds...")
    import time
    time.sleep(3)
    ret, frame = cap.read()
    cap.release()
    cv2.destroyAllWindows()
    if not ret:
        print("❌ Failed to capture image.")
        return None
    return frame
