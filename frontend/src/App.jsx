import { useState, useRef, useEffect } from "react";
import "./styles/global.css";
import { checkHealth } from "./api/healthApi";
import { analyzeImage as apiAnalyzeImage } from "./api/imageApi";
import { processVideo as apiProcessVideo } from "./api/videoApi";

const C = {
  bg:        "#0d1b3e",
  bgDeep:    "#091429",
  panel:     "#0f2147",
  border:    "#1a3a6e",
  borderGlow:"#2563eb",
  accent:    "#3b82f6",
  accentBr:  "#60a5fa",
  cyan:      "#22d3ee",
  red:       "#ef4444",
  green:     "#10b981",
  yellow:    "#f59e0b",
  text:      "#e2e8f0",
  textSub:   "#94a3b8",
  textMuted: "#475569",
};

const css = `
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');
  *{box-sizing:border-box;margin:0;padding:0}
  body{background:${C.bgDeep};color:${C.text};font-family:'Inter',sans-serif;min-height:100vh}
  .app{display:flex;flex-direction:column;min-height:100vh;background:radial-gradient(ellipse at 20% 0%,rgba(37,99,235,0.15) 0%,transparent 60%),radial-gradient(ellipse at 80% 100%,rgba(34,211,238,0.08) 0%,transparent 50%),${C.bgDeep}}

  .hdr{display:flex;align-items:center;padding:0 28px;height:52px;background:${C.bg};border-bottom:1px solid ${C.border};gap:16px}
  .hdr-dot{width:12px;height:12px;border-radius:50%;background:${C.accent};box-shadow:0 0 10px ${C.accent},0 0 20px rgba(59,130,246,0.4);animation:dotP 2s ease-in-out infinite}
  @keyframes dotP{0%,100%{box-shadow:0 0 10px ${C.accent},0 0 20px rgba(59,130,246,0.4)}50%{box-shadow:0 0 16px ${C.cyan},0 0 30px rgba(34,211,238,0.5)}}
  .hdr-name{font-size:16px;font-weight:700;letter-spacing:0.06em}
  .hdr-tag{font-size:11px;color:${C.textMuted};letter-spacing:0.12em;font-weight:500}
  .hdr-right{margin-left:auto;display:flex;align-items:center;gap:20px}
  .status-pill{display:flex;align-items:center;gap:7px;font-size:12px;font-weight:500}
  .sdot{width:8px;height:8px;border-radius:50%}
  .sdot.on{background:${C.green};box-shadow:0 0 8px ${C.green}}
  .sdot.off{background:${C.red};box-shadow:0 0 8px ${C.red}}
  .hdr-time{font-family:'JetBrains Mono',monospace;font-size:13px;color:${C.textSub}}

  .body{flex:1;padding:20px 24px;display:flex;flex-direction:column;gap:14px}

  .feeds{display:grid;grid-template-columns:1fr 1fr;gap:16px}
  .feed-card{background:${C.panel};border:1px solid ${C.border};border-radius:14px;overflow:hidden;position:relative;transition:border-color 0.2s}
  .feed-card:hover{border-color:${C.borderGlow}}
  .feed-hdr{display:flex;align-items:center;justify-content:space-between;padding:12px 16px;border-bottom:1px solid ${C.border}}
  .feed-lbl{font-size:11px;font-weight:700;letter-spacing:0.12em}
  .badge{font-size:10px;font-weight:700;letter-spacing:0.1em;padding:3px 10px;border-radius:5px}
  .b-live{background:${C.red};color:#fff}
  .b-ai{background:rgba(34,211,238,0.15);color:${C.cyan};border:1px solid rgba(34,211,238,0.3)}
  .b-warn{background:rgba(239,68,68,0.15);color:${C.red};border:1px solid rgba(239,68,68,0.3)}
  .b-std{color:${C.textMuted};font-size:10px;letter-spacing:0.1em}
  .feed-body{height:320px;display:flex;align-items:center;justify-content:center;position:relative;background:${C.bgDeep};overflow:hidden}
  .feed-media{width:100%;height:100%;object-fit:contain;display:block}
  .feed-ph{text-align:center;color:${C.textMuted}}
  .feed-ph svg{margin-bottom:10px;opacity:0.3}
  .feed-ph p{font-size:12px}
  .corner-tl{position:absolute;top:8px;left:8px;width:14px;height:14px;border-top:2px solid ${C.borderGlow};border-left:2px solid ${C.borderGlow};border-radius:2px 0 0 0;opacity:0.5}
  .corner-br{position:absolute;bottom:8px;right:8px;width:14px;height:14px;border-bottom:2px solid ${C.borderGlow};border-right:2px solid ${C.borderGlow};border-radius:0 0 2px 0;opacity:0.5}
  .scan{position:absolute;left:0;right:0;height:2px;background:linear-gradient(90deg,transparent,${C.cyan},transparent);animation:scan 3s linear infinite;opacity:0.35;pointer-events:none}
  @keyframes scan{0%{top:0}100%{top:100%}}

  .metrics{display:grid;grid-template-columns:repeat(5,1fr);gap:12px}
  .mc{background:${C.panel};border:1px solid ${C.border};border-radius:12px;padding:16px 20px;transition:border-color 0.2s}
  .mc:hover{border-color:${C.borderGlow}}
  .ml{font-size:10px;font-weight:700;letter-spacing:0.14em;color:${C.textMuted};margin-bottom:8px}
  .mv{font-family:'JetBrains Mono',monospace;font-size:24px;font-weight:600;line-height:1}
  .mv.acc{color:${C.accentBr}}
  .mv.grn{color:${C.green}}
  .mv.red{color:${C.red}}
  .mu{font-size:13px;font-weight:400;color:${C.textMuted};margin-left:3px}

  .controls{background:${C.panel};border:1px solid ${C.border};border-radius:12px;padding:14px 20px;display:flex;align-items:center;gap:12px;flex-wrap:wrap}
  .cb{display:flex;align-items:center;gap:7px;padding:9px 18px;border-radius:8px;font-size:13px;font-weight:600;cursor:pointer;border:none;font-family:inherit;transition:all 0.2s;letter-spacing:0.02em;white-space:nowrap}
  .cb.primary{background:${C.accent};color:#fff}
  .cb.primary:hover{background:${C.accentBr};color:#000}
  .cb.danger{background:transparent;color:${C.red};border:1px solid rgba(239,68,68,0.4)}
  .cb.danger:hover{background:rgba(239,68,68,0.1)}
  .cb.warn{background:transparent;color:${C.yellow};border:1px solid rgba(245,158,11,0.4)}
  .cb.warn:hover{background:rgba(245,158,11,0.1)}
  .cb:disabled{opacity:0.35;cursor:not-allowed}
  .cdiv{width:1px;height:28px;background:${C.border}}
  .fname{font-size:12px;color:${C.textMuted}}

  .mode-switch{display:flex;gap:2px;margin-left:auto;background:${C.bgDeep};border:1px solid ${C.border};border-radius:8px;padding:3px}
  .mb{padding:6px 16px;font-size:12px;font-weight:600;letter-spacing:0.06em;cursor:pointer;border:none;border-radius:6px;font-family:inherit;transition:all 0.2s;background:transparent;color:${C.textMuted}}
  .mb.active{background:rgba(59,130,246,0.2);color:${C.accentBr};border:1px solid rgba(59,130,246,0.3)}
  .mb:hover:not(.active){color:${C.text}}

  .error-banner{background:rgba(239,68,68,0.15);border:1px solid rgba(239,68,68,0.3);color:${C.red};padding:10px 16px;border-radius:8px;font-size:13px;display:flex;align-items:center;justify-content:space-between}
`;

function Clock() {
  const [t, setT] = useState(new Date());
  useEffect(() => {
    const id = setInterval(() => setT(new Date()), 1000);
    return () => clearInterval(id);
  }, []);
  return <span className="hdr-time">{t.toLocaleTimeString()}</span>;
}

export default function App() {
  const [mode, setMode]           = useState("video"); // "video" | "image"
  const [connected, setConnected] = useState(false);
  const [errorMsg, setErrorMsg]   = useState(null);

  // video state
  const [videoFile, setVideoFile]       = useState(null);
  const [videoPreview, setVideoPreview] = useState(null);
  const [processingVideo, setProcessingVideo] = useState(false);
  const [processedVideoUrl, setProcessedVideoUrl] = useState(null);

  // image state
  const [imageFile, setImageFile]     = useState(null);
  const [imagePreview, setPreview]    = useState(null);
  const [analyzingImage, setAnalyzingImage] = useState(false);
  const [annotatedImageUrl, setAnnotatedImageUrl] = useState(null);

  // shared metrics
  const [metrics, setMetrics] = useState({ fps: "0.0", polyps: 0, confidence: "—", frames: "—", latency: "—" });

  const videoRef = useRef();
  const imageRef = useRef();

  // Automatic connection check on mount
  useEffect(() => {
    handleCheckHealth();
  }, []);

  const handleCheckHealth = async () => {
    setErrorMsg(null);
    const res = await checkHealth();
    setConnected(res.connected);
    if (!res.connected) {
      setErrorMsg(`Backend offline (${res.error || "Cannot reach http://localhost:8000"})`);
    }
  };

  // ── Video handlers ───────────────────────────────────────────────────
  const handleVideoFile = (e) => {
    const f = e.target.files[0];
    if (!f) return;
    setVideoFile(f);
    setVideoPreview(URL.createObjectURL(f));
    setProcessedVideoUrl(null);
    setErrorMsg(null);
    setMetrics({ fps: "0.0", polyps: 0, confidence: "—", frames: "—", latency: "—" });
  };

  const processVideo = async () => {
    if (!videoFile) return;
    setProcessingVideo(true);
    setErrorMsg(null);
    try {
      const res = await apiProcessVideo(videoFile, 1);
      setProcessedVideoUrl(res.output_video_url);
      setMetrics({
        fps:        res.avg_fps ? res.avg_fps.toFixed(1) : "0.0",
        polyps:     res.frames_with_polyp,
        confidence: res.frames_with_polyp > 0 ? "Detected" : "Clean",
        frames:     `${res.frames_with_polyp} / ${res.total_frames}`,
        latency:    res.avg_latency_ms ? res.avg_latency_ms.toFixed(1) : "0.0",
      });
    } catch (err) {
      setErrorMsg(`Video processing failed: ${err.message}`);
    } finally {
      setProcessingVideo(false);
    }
  };

  const resetVideo = () => {
    setVideoFile(null);
    setVideoPreview(null);
    setProcessedVideoUrl(null);
    setErrorMsg(null);
    setMetrics({ fps: "0.0", polyps: 0, confidence: "—", frames: "—", latency: "—" });
  };

  // ── Image handlers ───────────────────────────────────────────────────
  const handleImageFile = (e) => {
    const f = e.target.files[0];
    if (!f) return;
    setImageFile(f);
    setPreview(URL.createObjectURL(f));
    setAnnotatedImageUrl(null);
    setErrorMsg(null);
    setMetrics({ fps: "0.0", polyps: 0, confidence: "—", frames: "—", latency: "—" });
  };

  const analyzeImage = async () => {
    if (!imageFile) return;
    setAnalyzingImage(true);
    setErrorMsg(null);
    try {
      const res = await apiAnalyzeImage(imageFile);
      setAnnotatedImageUrl(res.annotated_image);
      setMetrics({
        fps:        "—",
        polyps:     res.polyps_found,
        confidence: res.polyps_found > 0 ? `${(res.confidence * 100).toFixed(0)}%` : "Clean",
        frames:     "1 frame",
        latency:    res.latency_ms ? res.latency_ms.toFixed(1) : "0.0",
      });
    } catch (err) {
      setErrorMsg(`Image analysis failed: ${err.message}`);
    } finally {
      setAnalyzingImage(false);
    }
  };

  const resetImage = () => {
    setImageFile(null);
    setPreview(null);
    setAnnotatedImageUrl(null);
    setErrorMsg(null);
    setMetrics({ fps: "0.0", polyps: 0, confidence: "—", frames: "—", latency: "—" });
  };

  const switchMode = (m) => {
    if (m === mode) return;
    setMode(m);
    resetVideo();
    resetImage();
  };

  const origSrc = mode === "video" ? videoPreview : imagePreview;

  return (
    <>
      <style>{css}</style>
      <div className="app">

        {/* Header */}
        <header className="hdr">
          <div className="hdr-dot" />
          <span className="hdr-name">ENDO-X</span>
          <span className="hdr-tag">GI ENDOSCOPY AI BACKEND TESTER</span>
          <div className="hdr-right">
            <div className="status-pill">
              <div className={`sdot ${connected ? "on" : "off"}`} />
              <span style={{ color: connected ? C.green : C.red }}>
                {connected ? "Backend Online" : "Backend Offline"}
              </span>
            </div>
            <Clock />
          </div>
        </header>

        <div className="body">

          {errorMsg && (
            <div className="error-banner">
              <span>{errorMsg}</span>
              <button onClick={() => setErrorMsg(null)} style={{ background: "none", border: "none", color: C.red, cursor: "pointer", fontWeight: "bold" }}>✕</button>
            </div>
          )}

          {/* Feeds */}
          <div className="feeds">

            {/* Original Input Feed */}
            <div className="feed-card">
              <div className="corner-tl" /><div className="corner-br" />
              <div className="feed-hdr">
                <span className="feed-lbl">ORIGINAL INPUT ({mode.toUpperCase()})</span>
                {origSrc ? <span className="badge b-live">READY</span> : <span className="b-std">STANDBY</span>}
              </div>
              <div className="feed-body">
                {mode === "video" && videoPreview ? (
                  <video src={videoPreview} controls className="feed-media" />
                ) : mode === "image" && imagePreview ? (
                  <img src={imagePreview} alt="original" className="feed-media" />
                ) : (
                  <div className="feed-ph">
                    <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke={C.textMuted} strokeWidth="1.5">
                      <rect x="2" y="7" width="20" height="15" rx="2"/>
                      <path d="M16 3l4 4-4 4"/>
                    </svg>
                    <p>Select a {mode} file to preview</p>
                  </div>
                )}
              </div>
            </div>

            {/* AI Processed Output */}
            <div className="feed-card">
              <div className="corner-tl" /><div className="corner-br" />
              <div className="feed-hdr">
                <span className="feed-lbl">AI ANNOTATED OUTPUT</span>
                {(processingVideo || analyzingImage) ? (
                  <span className="badge b-warn">PROCESSING</span>
                ) : (processedVideoUrl || annotatedImageUrl) ? (
                  <span className={`badge ${metrics.polyps > 0 ? "b-warn" : "b-ai"}`}>
                    {metrics.polyps > 0 ? "POLYP DETECTED" : "AI PROCESSED"}
                  </span>
                ) : (
                  <span className="b-std">STANDBY</span>
                )}
              </div>
              <div className="feed-body">
                {mode === "video" && processedVideoUrl ? (
                  <video src={processedVideoUrl} controls autoPlay loop className="feed-media" />
                ) : mode === "image" && annotatedImageUrl ? (
                  <img src={annotatedImageUrl} alt="annotated output" className="feed-media" />
                ) : (processingVideo || analyzingImage) ? (
                  <div className="feed-ph">
                    <svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke={C.accent} strokeWidth="1.5" style={{ animation: "dotP 1s linear infinite" }}>
                      <circle cx="12" cy="12" r="10"/>
                      <path d="M12 6v6l4 2"/>
                    </svg>
                    <p style={{ color: C.accent, marginTop: 8 }}>
                      Running PyTorch Inference ({mode === "video" ? "Frame-by-Frame Video Pipeline" : "YOLO + U-Net"})...
                    </p>
                  </div>
                ) : (
                  <div className="feed-ph">
                    <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke={C.textMuted} strokeWidth="1.5">
                      <circle cx="12" cy="12" r="10"/>
                      <path d="M12 8v4l3 3"/>
                    </svg>
                    <p>Awaiting inference execution</p>
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Metrics */}
          <div className="metrics">
            <div className="mc"><div className="ml">AVG FPS</div><div className="mv acc">{metrics.fps}</div></div>
            <div className="mc"><div className="ml">POLYPS FOUND</div><div className={`mv ${metrics.polyps > 0 ? "red" : ""}`}>{metrics.polyps}</div></div>
            <div className="mc"><div className="ml">CONFIDENCE</div><div className={`mv ${metrics.polyps > 0 ? "grn" : ""}`}>{metrics.confidence}</div></div>
            <div className="mc"><div className="ml">POSITIVE FRAMES</div><div className="mv acc">{metrics.frames}</div></div>
            <div className="mc"><div className="ml">LATENCY</div><div className="mv">{metrics.latency}<span className="mu">ms</span></div></div>
          </div>

          {/* Controls */}
          <div className="controls">

            {/* Check Connection */}
            <button className="cb primary" onClick={handleCheckHealth}>
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5"><path d="M13 10V3L4 14h7v7l9-11h-7z"/></svg>
              Check Backend
            </button>

            <div className="cdiv" />

            {/* VIDEO mode controls */}
            {mode === "video" && (
              <>
                <button className="cb warn" onClick={() => videoRef.current.click()}>
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5"><path d="M15 10l4.553-2.276A1 1 0 0121 8.723v6.554a1 1 0 01-1.447.894L15 14M3 8a2 2 0 012-2h8a2 2 0 012 2v8a2 2 0 01-2 2H5a2 2 0 01-2-2V8z"/></svg>
                  Select Video
                </button>
                <input ref={videoRef} type="file" accept="video/*" style={{ display: "none" }} onChange={handleVideoFile} />
                <span className="fname">{videoFile ? videoFile.name : "No video selected"}</span>
                {videoFile && !processingVideo && (
                  <button className="cb primary" onClick={processVideo} disabled={!connected}>Process Video</button>
                )}
                {videoFile && (
                  <button className="cb warn" onClick={resetVideo} style={{ opacity: 0.7 }}>Reset</button>
                )}
              </>
            )}

            {/* IMAGE mode controls */}
            {mode === "image" && (
              <>
                <button className="cb warn" onClick={() => imageRef.current.click()}>
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5"><rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="8.5" cy="8.5" r="1.5"/><path d="M21 15l-5-5L5 21"/></svg>
                  Select Image
                </button>
                <input ref={imageRef} type="file" accept="image/*" style={{ display: "none" }} onChange={handleImageFile} />
                <span className="fname">{imageFile ? imageFile.name : "No image selected"}</span>
                {imageFile && !analyzingImage && (
                  <button className="cb primary" onClick={analyzeImage} disabled={!connected}>Analyze Image</button>
                )}
                {imageFile && (
                  <button className="cb warn" onClick={resetImage} style={{ opacity: 0.7 }}>Reset</button>
                )}
              </>
            )}

            {/* Mode Switch */}
            <div className="mode-switch">
              <button className={`mb ${mode === "video" ? "active" : ""}`} onClick={() => switchMode("video")}>VIDEO</button>
              <button className={`mb ${mode === "image" ? "active" : ""}`} onClick={() => switchMode("image")}>IMAGE</button>
            </div>

          </div>
        </div>
      </div>
    </>
  );
}
