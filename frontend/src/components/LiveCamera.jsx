import { useEffect, useRef, useState } from "react";

const API_BASE_URL = process.env.REACT_APP_API_URL || "http://localhost:8000";

export default function LiveCamera() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const socketRef = useRef(null);
  const timerRef = useRef(null);
  const [running, setRunning] = useState(false);
  const [annotatedFrame, setAnnotatedFrame] = useState(null);
  const [stats, setStats] = useState({ fps: 0, latency: 0, detections: 0 });
  const [error, setError] = useState(null);
  const sendingRef = useRef(false);

  const stop = () => {
    if (timerRef.current) window.clearInterval(timerRef.current);
    timerRef.current = null;
    if (socketRef.current) socketRef.current.close();
    socketRef.current = null;
    sendingRef.current = false;
    if (videoRef.current?.srcObject) {
      videoRef.current.srcObject.getTracks().forEach((track) => track.stop());
      videoRef.current.srcObject = null;
    }
    setRunning(false);
  };

  const start = async () => {
    setError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 640 },
          height: { ideal: 360 },
          frameRate: { ideal: 15, max: 15 },
        },
        audio: false,
      });
      videoRef.current.srcObject = stream;
      await videoRef.current.play();

      const socketUrl =
        API_BASE_URL.replace(/^http/, "ws") + "/api/v1/predict/live";
      const socket = new WebSocket(socketUrl);
      socket.binaryType = "arraybuffer";
      socket.onopen = () => {
        setRunning(true);
        timerRef.current = window.setInterval(() => {
          const video = videoRef.current;
          const canvas = canvasRef.current;
          if (
            !video ||
            !canvas ||
            socket.readyState !== WebSocket.OPEN ||
            sendingRef.current
          )
            return;
          sendingRef.current = true;
          canvas.width = 640;
          canvas.height = 360;
          canvas
            .getContext("2d")
            .drawImage(video, 0, 0, canvas.width, canvas.height);
          canvas.toBlob(
            (blob) => {
              if (blob && socket.readyState === WebSocket.OPEN)
                socket.send(blob);
              else sendingRef.current = false;
            },
            "image/jpeg",
            0.55,
          );
        }, 1000 / 15);
      };
      socket.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if (data.status === "error") {
          setError(data.message || "Live inference failed");
          return;
        }
        setAnnotatedFrame(`data:image/jpeg;base64,${data.frame}`);
        setStats({
          fps: data.fps || 0,
          latency: data.latency_ms || 0,
          detections: data.detections?.length || 0,
        });
        sendingRef.current = false;
      };
      socket.onerror = () =>
        setError("Could not connect to the live inference endpoint");
      socket.onclose = () => setRunning(false);
      socketRef.current = socket;
    } catch (err) {
      stop();
      setError(err.message || "Camera access was denied");
    }
  };

  useEffect(() => stop, []);

  return (
    <div
      className="live-card"
      style={{
        background: "rgba(16,33,36,0.92)",
        border: "1px solid #203a3b",
        borderRadius: 10,
        padding: 16,
        marginBottom: 14,
        boxShadow: "0 16px 36px rgba(0,0,0,0.14)",
      }}
    >
      <div className="live-grid">
        <div>
          <div className="live-panel-label">CAMERA INPUT</div>
          <video ref={videoRef} muted playsInline className="live-screen" />
        </div>
        <div>
          <div className="live-panel-label">AI ANNOTATED OUTPUT</div>
          {annotatedFrame ? (
            <img
              src={annotatedFrame}
              alt="Live annotated camera frame"
              className="live-screen"
            />
          ) : (
            <div className="live-screen" />
          )}
        </div>
      </div>
      <canvas ref={canvasRef} style={{ display: "none" }} />
      <div className="live-stats">
        <button className="cb primary" onClick={running ? stop : start}>
          {running ? "Stop Camera" : "Start Camera"}
        </button>
        <span>FPS: {stats.fps.toFixed(1)}</span>
        <span>Latency: {stats.latency.toFixed(1)} ms</span>
        <span>Detections: {stats.detections}</span>
        {error && <span style={{ color: "#ef4444" }}>{error}</span>}
      </div>
    </div>
  );
}
