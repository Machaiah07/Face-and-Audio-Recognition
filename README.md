

---

# Multimodal Biometric Recognition System

This project is an **AI-powered biometric recognition system** that identifies individuals using both **face and voice** inputs. It combines state-of-the-art models like **ArcFace**, **FaceNet**, and **ECAPA-TDNN** for high-accuracy multimodal person identification.

---

## Features

* 🔊 **Voice Recognition:** Uses **ECAPA-TDNN** and **MFCC features** for accurate speaker identification.
* 🧠 **Face Recognition:** Leverages **ArcFace** and **FaceNet** for extracting deep face embeddings.
* 📸 **Real-time Webcam Capture:** Allows image capture directly via webcam for live recognition.
* 💻 **CLI Interface:** Simple command-line menu to upload images, audio, or use the webcam.
* 🧩 **Unified ML Pipeline:** Combines face and voice embeddings into a robust classifier using **SVM**.

---

## Project Structure

```
Multimodal-Biometric-Recognition-System/
│
├── app.py                 # Main Flask app for running the system
├── model_utils.py         # Feature extraction and model utilities
├── prompts.txt            # Run commands and setup prompts
├── requirements.txt       # Dependencies list
├── Dataset/               # (User-provided) Folder with subfolders for each person
│   ├── Person1/
│   │   ├── Pictures/*.jpg
│   │   └── Audio/*.m4a
│   ├── Person2/
│   └── ...
└── README.md
```

---

## ⚙️ Installation

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/Machaiah07/Multimodal-Biometric-Recognition-System.git
cd Multimodal-Biometric-Recognition-System
```

### 2️⃣ Create Virtual Environment (Recommended)

```bash
conda create -n faceaudio python=3.10
conda activate faceaudio
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🧪 Usage

### ▶️ Run the Application

```bash
python app.py
```

Or simply follow the prompts file:

```
Prompt 1: conda activate faceaudio
Prompt 2: python app.py
```

### 🧍 Dataset Structure

Make sure your dataset is organized as:

```
Dataset/
│
├── Divin Machaiah/
│   ├── Pictures/
│   │   ├── img1.jpg
│   │   └── img2.jpg
│   └── Audio/
│       ├── voice1.m4a
│       └── voice2.m4a
```

---

## 🧰 Technologies Used

* **Flask** — Web framework
* **OpenCV & Pillow** — Image handling
* **Facenet-PyTorch & InsightFace** — Face embedding extraction
* **SpeechBrain & Librosa** — Audio feature extraction
* **Scikit-Learn** — Classification (SVM)
* **Torch & ONNXRuntime** — Deep learning model execution

---

## 🧠 Models Used

| Modality | Model                     | Description                      |
| -------- | ------------------------- | -------------------------------- |
| Face     | **FaceNet**               | Generates 128D facial embeddings |
| Face     | **ArcFace (InsightFace)** | Highly discriminative embeddings |
| Audio    | **ECAPA-TDNN**            | Robust voice embeddings          |
| Audio    | **MFCC**                  | Traditional spectral features    |

---

## 🎯 Future Enhancements

* 🌐 Deploy as a full web application
* 📱 Add live face + voice authentication
* ☁️ Integrate with cloud storage for dataset management

---

## 🧑‍💻 Author

**Divin Machaiah KV**

> AI & ML | B.Tech CSE (AI & ML) | Passionate about Deep Learning & Biometrics
> 📍 [GitHub](https://github.com/Machaiah07)

---

