# ENDO-X 🔬

**Real-Time Endoscopic Polyp Detection & Segmentation Platform**

A Computer Vision project for intelligent analysis of gastrointestinal endoscopy images and videos, developed as part of NTI Summer Training - Computer Vision track.

---

## Architecture

```
ENDO-X/
├── backend/                          # FastAPI Backend Server
│   ├── app/
│   │   ├── main.py                   # App entry + lifespan model loading
│   │   ├── config.py                 # Central configuration
│   │   ├── models/
│   │   │   ├── base.py               # Abstract model interface
│   │   │   ├── mock_detector.py      # Dummy YOLO detection (CPU-blocking)
│   │   │   └── mock_segmentor.py     # Dummy segmentation (CPU-blocking)
│   │   ├── services/
│   │   │   ├── model_manager.py      # Model lifecycle management
│   │   │   └── frame_processor.py    # Thread-safe concurrent inference
│   │   ├── routers/
│   │   │   ├── ws.py                 # Binary WebSocket endpoint
│   │   │   └── health.py             # REST health checks
│   │   └── utils/
│   │       └── drawing.py            # OpenCV overlay rendering
│   ├── requirements.txt
│   └── run.py                        # Uvicorn launcher
│
├── frontend/                         # Vanilla JS Dashboard
│   ├── index.html                    # Single-page medical dashboard
│   ├── css/styles.css                # Dark medical theme (glassmorphism)
│   └── js/
│       ├── app.js                    # Main orchestrator + backpressure
│       ├── websocket.js              # Binary WebSocket manager
│       ├── ui.js                     # DOM & canvas rendering
│       └── telemetry.js              # FPS & stats tracking
│
└── YOLO_Dataset_Strict/              # Training dataset
```

## Key Technical Decisions

| Challenge | Solution |
|---|---|
| **Python GIL Blocking** | `asyncio.to_thread()` offloads sync model inference to thread pool |
| **Network Bandwidth** | Raw binary WebSocket frames (no Base64 overhead) |
| **Frame Flooding** | Backpressure mechanism — client waits for response before sending next frame |
| **Binary Protocol** | `[4B header: JSON length] + [JSON metadata] + [JPEG bytes]` |

## Quick Start

### Backend
```bash
cd backend
pip install -r requirements.txt
python run.py
# → Server at http://localhost:8000
```

### Frontend
Open `frontend/index.html` in your browser, click **Load Video** to select an endoscopy clip, then click **Connect** to start streaming.

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `WS` | `/ws/video` | Binary WebSocket for real-time frame processing |
| `GET` | `/api/health` | Server health check |
| `GET` | `/api/models/status` | Loaded model status |
