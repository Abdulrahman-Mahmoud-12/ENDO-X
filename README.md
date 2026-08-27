# 🩺 ENDO-X: Intelligent Gastrointestinal Endoscopy Vision System

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18-61DAFB.svg)](https://reactjs.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5-EE4C2C.svg)](https://pytorch.org/)
[![Ultralytics YOLO](https://img.shields.io/badge/YOLO-v8%2Fv11-00FFFF.svg)](https://docs.ultralytics.com/)

**ENDO-X** is an end-to-end computer vision platform designed for real-time and offline intelligent analysis of gastrointestinal (GI) endoscopy images, video recordings, and live camera streams. Developed as part of the **NTI Summer Training (Computer Vision Track)** graduation project.

The system combines state-of-the-art deep learning architectures for **polyp object detection**, **pixel-level segmentation**, and **real-time object tracking** with a high-performance **FastAPI backend** and an intuitive **React frontend**.

> ⚠️ **Disclaimer:** *ENDO-X is strictly an educational and research project. It is not intended, certified, or validated for clinical diagnosis or direct medical decision-making.*

---

## 🌟 Key Features

- 📸 **Image Analysis Mode:** Upload single endoscopy images (`.jpg`, `.jpeg`, `.png`) to receive bounding-box polyp detections, pixel-accurate segmentation overlays, confidence scores, and latency metrics.
- 🎬 **Video Processing Mode:** Upload frame sequences or endoscopy videos (`.mp4`, `.avi`, `.mov`) for frame-by-frame detection, segmentation, and continuous object tracking across frames.
- 🎥 **Live Camera Inference Mode:** Real-time stream processing over WebSockets directly from a camera device or webcam feed, monitoring latency and live FPS.
- 🎯 **Dual AI Vision Engine:**
  - **Object Detection:** YOLO-based detector locating polyps and returning bounding box coordinates and confidence levels.
  - **Semantic Segmentation:** U-Net / U-Net++ architectures for boundary-level mask extraction of detected polyp regions.
  - **Multi-Object Tracking:** ByteTrack / BoT-SORT object tracking to maintain persistent polyp IDs across frames.
- 📊 **Inference & Metrics Dashboard:** Displays real-time FPS, total latency, bounding box statistics, mask area (in pixels), and confidence metrics.

---

## 🏗️ System Architecture

```text
┌─────────────────────────────────────────────────────────────┐
│                      REACT FRONTEND                         │
│                                                             │
│  [ Image Mode ]       [ Video Mode ]    [ Live Camera Mode ]│
│  - Drag & Drop        - Frame Player     - WebCam Stream    │
│  - Masks & BBoxes     - Video Exporter   - WebSocket Stats  │
└──────────────────────────┬──────────────────────────────────┘
                           │
                    HTTP / WebSocket API
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                     FASTAPI BACKEND                         │
│                                                             │
│  ├── GET  /api/v1/health          (System Readiness)        │
│  ├── POST /api/v1/predict/image   (Image Inference)         │
│  ├── POST /api/v1/predict/video   (Video Processing)        │
│  └── WS   /api/v1/predict/live    (Real-time Streams)       │
└───────────────┬──────────────────────┬──────────────────────┘
                │                      │
                ▼                      ▼
┌─────────────────────────┐   ┌──────────────────────────────┐
│     INFERENCE ENGINE    │   │      UTILITIES & STORAGE     │
│  - YOLO Object Detector │   │  - Pre/Post Processing       │
│  - U-Net Segmenter      │   │  - Frame Extraction          │
│  - Object Tracker       │   │  - Uploads & Outputs Storage │
└─────────────────────────┘   └──────────────────────────────┘
```

---

## 📂 Project Structure

```text
ENDO-X/
├── backend/                  # FastAPI Application
│   ├── app/
│   │   ├── api/              # V1 API endpoints (health, image, video, live)
│   │   ├── core/             # Configuration & lifespan settings
│   │   ├── models/           # YOLO Detector, U-Net Segmenter & Tracker wrappers
│   │   ├── pipeline/         # Image & Video processing pipelines
│   │   ├── schemas/          # Pydantic data models & request/response types
│   │   ├── services/         # Preprocessing, inference & media services
│   │   └── utils/            # Image/video decoding, resizing & visualizers
│   ├── Dockerfile
│   └── requirements.txt      # Python dependencies
│
├── frontend/                 # React Web Application
│   ├── public/               # Static assets & index.html
│   ├── src/
│   │   ├── api/              # Axios & WebSocket client connections
│   │   ├── components/       # Header, VideoPlayer, ImageResults, UI elements
│   │   ├── pages/            # ImageAnalysis & VideoStream pages
│   │   └── styles/           # CSS & color theme definitions
│   └── package.json
│
├── models/                   # Model Weights Storage
│   ├── detector/             # YOLO detector weights (e.g., best.pt)
│   └── segmenter/            # U-Net segmenter weights (e.g., best.pth)
│
├── training/                 # Model Training & Evaluation Scripts
│   ├── configs/              # Hyperparameter YAML configs
│   ├── detection/            # Detector training & validation scripts
│   ├── segmentation/         # Segmenter training & validation scripts
│   └── evaluation/           # mAP, Dice, IoU & benchmark scripts
│
├── run_backend.bat           # Windows startup script for backend
├── run_frontend.bat          # Windows startup script for frontend
├── .env.example              # Environment variables template
├── projext_overview.md       # Detailed technical design document
└── README.md
```

---

## 🛠️ Technology Stack

| Domain | Tools & Frameworks |
| :--- | :--- |
| **Frontend** | React 18, JavaScript/JSX, CSS |
| **Backend API** | FastAPI, Uvicorn, Pydantic v2, WebSockets, Aiofiles |
| **AI & Computer Vision** | PyTorch, Ultralytics (YOLO), Segmentation Models PyTorch (U-Net++), OpenCV, Pillow, Albumentations |
| **Datasets** | Kvasir-SEG (Training & Validation), PolypDB (External Evaluation) |
| **Tooling & Environment** | Python 3.10+, Node.js 18+, Docker |

---

## 🚀 Getting Started

### Prerequisites

- **Python**: `^3.10`
- **Node.js**: `^18.0` (with `npm`)
- **CUDA** *(Optional, recommended for fast GPU inference)*: NVIDIA GPU with CUDA support

---

### 1. Clone the Repository & Configure Environment

```bash
git clone https://github.com/Abdulrahman-Mahmoud-12/ENDO-X.git
cd ENDO-X
```

Copy the sample configuration file to `.env`:

```bash
cp .env.example .env
```

---

### 2. Backend Setup & Run

#### Option A: Quick Run (Windows Script)
Double-click `run_backend.bat` or run in terminal:
```cmd
run_backend.bat
```

#### Option B: Manual Setup
```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# Linux/macOS:
# source .venv/bin/activate

# Install backend dependencies
cd backend
pip install -r requirements.txt

# Run FastAPI server
python -m uvicorn app.main:app --host 127.0.0.1 --port 8080 --reload
```

The backend server will start at `http://127.0.0.1:8080`.
Documentation endpoints:
- Swagger UI: `http://127.0.0.1:8080/docs`
- Redoc: `http://127.0.0.1:8080/redoc`

---

### 3. Frontend Setup & Run

#### Option A: Quick Run (Windows Script)
Double-click `run_frontend.bat` or run in terminal:
```cmd
run_frontend.bat
```

#### Option B: Manual Setup
```bash
cd frontend

# Install Node modules
npm install

# Start React development server
npm start
```

The frontend web application will open at `http://localhost:3000`.

---

## 📡 API Reference

### Health Check
- `GET /health` or `GET /api/v1/health`
- **Response:**
  ```json
  {
    "status": "healthy",
    "detector": "loaded",
    "segmenter": "loaded"
  }
  ```

### Image Analysis
- `POST /api/v1/predict/image`
- **Payload:** `multipart/form-data` with `file` key containing image (`.jpg`, `.png`).
- **Response:** JSON containing bounding boxes, segmentation masks, overlaid base64 preview, and timing stats.

### Video Processing
- `POST /api/v1/predict/video`
- **Payload:** `multipart/form-data` with `file` key containing video (`.mp4`, `.avi`).
- **Response:** JSON output detailing processed video path, total frames, tracked objects, and average FPS.

### Live Stream Processing
- `WebSocket /api/v1/predict/live`
- Transmits raw frame buffers and receives annotated frame results and live telemetry over WebSockets.

---

## 📊 Datasets & Training

The models in ENDO-X are trained and evaluated using standard GI endoscopy benchmarks:
- **[Kvasir-SEG](https://endovision.sintef.no/kvasir-seg.html):** Primary dataset providing endoscopy frames with pixel-level polyp segmentations and bounding box annotations.
- **[PolypDB](https://github.com/):** Secondary dataset utilized for cross-dataset external generalization testing.

### Training & Evaluation Metrics
- **Detection:** mAP@50, mAP@50:95, Precision, Recall.
- **Segmentation:** Dice Coefficient (F1-score), Intersection over Union (IoU).
- **Performance:** Inference Latency (ms), Frames Per Second (FPS), Memory footprint.

To launch training experiments, navigate to the `training/` folder:
```bash
# Detector training
python training/detection/train.py --config training/configs/detection.yaml

# Segmenter training
python training/segmentation/train.py --config training/configs/segmentation.yaml
```

---

## 🤝 Acknowledgments

Developed during the **NTI Summer Training - Computer Vision Track**. Special thanks to the mentors and team members contributing to the development of ENDO-X.
