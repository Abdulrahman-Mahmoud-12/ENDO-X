# ENDO-X

## 1. Project Overview

**ENDO-X** is a computer vision system designed to analyze gastrointestinal endoscopy images, uploaded videos, and live camera streams.

The system goes beyond simple polyp detection by combining:

- Polyp object detection
- Pixel-level polyp segmentation
- Video object tracking
- Image analysis
- Video analysis
- Real-time camera inference
- Interactive visualization
- FastAPI backend
- Web-based frontend

The main goal is to build a complete AI-powered endoscopy vision system rather than a standalone deep-learning notebook.

> **Important:** The system is intended as an educational/research graduation project and should not be considered a clinical diagnostic system.

---

# 2. Main Objective

The system receives gastrointestinal endoscopy visual data through three possible input modes:

```text
Image
Video
Live Camera
```

and processes the input through an AI computer vision pipeline:

```text
Input
  ↓
Preprocessing
  ↓
Polyp Detection
  ↓
Region of Interest
  ↓
Polyp Segmentation
  ↓
Tracking (Video / Live)
  ↓
Post-processing
  ↓
Visualization
  ↓
Frontend
```

---

# 3. Supported Input Modes

## 3.1 Image Mode

The user uploads an endoscopy image.

```text
Image
  ↓
Detection
  ↓
Segmentation
  ↓
Visualization
```

The system returns:

- Detected polyp bounding boxes
- Detection confidence
- Segmentation masks
- Segmentation overlay
- Number of detected polyps
- Inference time

Supported formats can include:

```text
.jpg
.jpeg
.png
```

---

## 3.2 Video Mode

The user uploads an endoscopy video.

```text
Video
  ↓
Frame Extraction
  ↓
Detection
  ↓
Segmentation
  ↓
Tracking
  ↓
Annotated Video
```

The system processes the video frame by frame.

The output should contain:

- Bounding boxes
- Segmentation masks
- Confidence scores
- Persistent object IDs
- FPS
- Processing information

Possible supported formats:

```text
.mp4
.avi
.mov
```

---

## 3.3 Live Camera Mode

The system can access a camera through the browser.

For the graduation-project demonstration, a normal webcam can be used to simulate an endoscopy camera.

```text
Camera
  ↓
Browser
  ↓
WebSocket
  ↓
FastAPI
  ↓
AI Inference
  ↓
Annotated Frame
  ↓
Browser
```

The live interface should display:

- Live camera stream
- Bounding boxes
- Segmentation masks
- Confidence
- FPS
- Inference latency

---

# 4. AI Pipeline

The main AI pipeline consists of several stages.

```text
                   INPUT
                     │
        ┌────────────┼────────────┐
        │            │            │
      Image        Video        Camera
        │            │            │
        └────────────┼────────────┘
                     ↓
              Preprocessing
                     ↓
             Object Detection
                     ↓
                Polyp ROI
                     ↓
               Segmentation
                     ↓
            Tracking if needed
                     ↓
             Post Processing
                     ↓
              Visualization
                     ↓
                 Result
```

---

# 5. Detection Model

The detection model identifies the location of polyps.

### Recommended model

```text
YOLO
```

The detector produces:

```text
Class
Confidence
Bounding Box
```

Example:

```json
{
  "class": "polyp",
  "confidence": 0.967,
  "bbox": [120, 85, 430, 350]
}
```

The detector answers:

> **Where is the polyp?**

---

# 6. Segmentation Model

The segmentation model identifies the exact pixels belonging to the detected polyp.

Possible architectures:

```text
U-Net
U-Net++
SegFormer
DeepLabV3+
```

The recommended starting point is:

```text
U-Net / U-Net++
```

The segmentation model produces a binary or probability mask.

The segmentation pipeline is:

```text
Detected Polyp
      ↓
ROI / Crop
      ↓
Segmentation Model
      ↓
Polyp Mask
      ↓
Overlay
```

The segmentation model answers:

> **Which pixels belong to the polyp?**

---

# 7. Video Tracking

For video and live-camera inference, object tracking can be added to maintain the identity of the same polyp across multiple frames.

Recommended trackers:

```text
ByteTrack
BoT-SORT
DeepSORT
```

Example:

```text
Frame 1 → Polyp #1
Frame 2 → Polyp #1
Frame 3 → Polyp #1
Frame 4 → Polyp #1
```

instead of treating every frame as a new detection.

Tracking output can contain:

```text
Object ID
Bounding Box
Confidence
Frame Count
```

---

# 8. Post-Processing

After detection and segmentation, the system performs lightweight post-processing.

Possible outputs include:

```text
Number of polyps
Bounding box coordinates
Segmentation mask
Confidence
Mask area in pixels
Bounding-box width
Bounding-box height
Aspect ratio
Inference latency
FPS
```

Physical measurements should not be reported unless the image/video has a valid scale calibration.

---

# 9. Visualization

The frontend should visualize the AI results directly on the input.

For images:

```text
Original Image
      +
Bounding Box
      +
Segmentation Mask
```

For videos:

```text
Video Frame
      +
Bounding Box
      +
Segmentation Mask
      +
Tracking ID
```

For live camera:

```text
Live Frame
      +
Bounding Box
      +
Segmentation Mask
      +
Confidence
      +
FPS
```

---

# 10. System Architecture

```text
┌─────────────────────────────────────────────────────────────┐
│                         FRONTEND                            │
│                                                             │
│  Image Mode │ Video Mode │ Live Camera Mode                 │
│                                                             │
│  Upload │ Camera │ Visualization │ Metrics                  │
└──────────────────────────┬──────────────────────────────────┘
                           │
                    HTTP / WebSocket
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                       FASTAPI BACKEND                       │
│                                                             │
│  Image API │ Video API │ Live WebSocket │ Health API        │
└───────────────┬──────────────────────┬──────────────────────┘
                │                      │
                ▼                      ▼
┌─────────────────────────┐   ┌──────────────────────────────┐
│     INFERENCE ENGINE    │   │       FILE PROCESSING        │
│                         │   │                              │
│ Detection               │   │ Image Processing             │
│ Segmentation            │   │ Video Processing             │
│ Tracking                │   │ Frame Extraction             │
│ Post-processing         │   │ Result Encoding              │
└────────────┬────────────┘   └──────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────────┐
│                       AI MODELS                             │
│                                                             │
│              YOLO Detector                                  │
│              U-Net / U-Net++ Segmenter                      │
│              ByteTrack / BoT-SORT Tracker                   │
└─────────────────────────────────────────────────────────────┘
```

---

# 11. Backend Architecture

FastAPI acts as the central communication layer.

```text
Frontend
   │
   ├── POST /predict/image
   ├── POST /predict/video
   ├── WebSocket /predict/live
   └── GET /health
            │
            ▼
         FastAPI
            │
            ├── Preprocessing
            ├── Inference
            ├── Tracking
            ├── Post-processing
            └── Response
```

The backend should load the AI models once during application startup instead of loading them for every request.

---

# 12. API Design

## Image Prediction

```http
POST /api/v1/predict/image
```

Input:

```text
Multipart image file
```

Output:

```json
{
  "detections": [],
  "segmentations": [],
  "inference_time": 0.042
}
```

---

## Video Prediction

```http
POST /api/v1/predict/video
```

Input:

```text
Video file
```

Output:

```text
Processed / annotated video
```

---

## Live Prediction

```text
WebSocket /api/v1/predict/live
```

The client sends frames to the server and receives annotated frames/results.

---

## Health Check

```http
GET /api/v1/health
```

Returns the service and model status.

Example:

```json
{
  "status": "healthy",
  "detector": "loaded",
  "segmenter": "loaded"
}
```

---

# 13. Frontend Structure

The frontend should contain three primary modes.

```text
┌────────────────────────────────────────────┐
│                    ENDO-X                  │
├────────────────────────────────────────────┤
│                                            │
│  [ IMAGE ] [ VIDEO ] [ LIVE CAMERA ]       │
│                                            │
├────────────────────────────────────────────┤
│                                            │
│                VIEWER                      │
│                                            │
├────────────────────────────────────────────┤
│  Confidence │ FPS │ Latency │ Objects      │
└────────────────────────────────────────────┘
```

---

# 14. Dataset

## Kvasir-SEG

Kvasir-SEG is used as the primary dataset for polyp segmentation and detection training.

It provides:

- Endoscopy images
- Polyp segmentation masks
- Bounding-box information

---

## PolypDB

PolypDB can be used as an additional dataset for testing model generalization.

It contains images from different sources and imaging modalities.

A good experimental strategy is:

```text
Training
   ↓
Kvasir-SEG
   ↓
Validation
   ↓
Kvasir-SEG
   ↓
External Evaluation
   ↓
PolypDB
```

This allows the project to investigate whether the trained model generalizes to data from different sources.

---

# 15. Training Pipeline

Training is separated from the production application.

```text
Dataset
   ↓
Preprocessing
   ↓
Augmentation
   ↓
Train / Validation Split
   ↓
Model Training
   ↓
Evaluation
   ↓
Best Checkpoint
   ↓
Production Weights
```

The final deployment should only contain the trained model weights required for inference.

---

# 16. Evaluation

## Detection Metrics

```text
Precision
Recall
mAP@50
mAP@50:95
Inference Time
FPS
```

## Segmentation Metrics

```text
Dice Score
IoU
Precision
Recall
Inference Time
```

## Real-Time Metrics

```text
FPS
Average Latency
Model Size
GPU/CPU Memory Usage
```

---

# 17. Technology Stack

| Component               | Technology               |
| ----------------------- | ------------------------ |
| Frontend                | React + TypeScript       |
| UI                      | Tailwind CSS             |
| Backend                 | FastAPI                  |
| API Validation          | Pydantic                 |
| Real-Time Communication | WebSocket                |
| Deep Learning           | PyTorch                  |
| Detection               | YOLO                     |
| Segmentation            | U-Net / U-Net++          |
| Tracking                | ByteTrack / BoT-SORT     |
| Image Processing        | OpenCV                   |
| Dataset                 | Kvasir-SEG + PolypDB     |
| Containerization        | Docker                   |
| Deployment              | Docker-compatible server |

---

# 18. Project Scope

## Core Features

- [x] Endoscopy image upload
- [x] Polyp detection
- [x] Polyp segmentation
- [x] Result visualization
- [x] Endoscopy video upload
- [x] Frame-by-frame inference
- [x] Object tracking
- [x] Live camera inference
- [x] Real-time visualization
- [x] FPS and latency monitoring
- [x] FastAPI backend
- [x] React frontend

## Optional Features

- [ ] Compare multiple detection architectures
- [ ] Compare multiple segmentation architectures
- [ ] Model confidence threshold control
- [ ] Adjustable inference resolution
- [ ] GPU/CPU inference selection
- [ ] Model performance dashboard

## Explicitly Outside the Scope

The project does **not** require:

- Patient management
- Medical records
- Patient history
- PDF reports
- Authentication systems
- RAG
- LLM
- Clinical diagnosis
- Hospital management
- Electronic health records

This keeps the project focused on computer vision, AI inference, and real-time software engineering.

---

# 19. Final Project Concept

The final system can be described as:

> **A web-based gastrointestinal endoscopy computer vision platform that supports image, video, and live-camera inputs and uses deep-learning-based polyp detection, pixel-level segmentation, and object tracking to provide real-time visual analysis and quantitative inference results.**
