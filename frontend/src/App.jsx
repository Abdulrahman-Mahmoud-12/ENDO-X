import { useState, useRef, useEffect } from "react";
import "./styles/global.css";

// ── Mock APIs — replace with real endpoints ──────────────────────────────
const mockAnalyzeImage = async (file) => {
  await new Promise(r => setTimeout(r, 1800));
  return {
    annotated_image:   URL.createObjectURL(file),
    segmentation_mask: URL.createObjectURL(file),
    confidence: 0.93,
    size_mm2:   118.4,
    diameter_mm: 12.3,
  };
};

const mockStreamVideo = (file, onFrame, onDone) => {
  let frame = 0;
  const total = 60;
  const url   = URL.createObjectURL(file);
  const id    = setInterval(() => {
    frame++;
    onFrame({
      frame_url:    url,
      frame_index:  frame,
      total_frames: total,
      polyps_found: frame % 6 === 0,
      confidence:   frame % 6 === 0 ? (0.88 + Math.random() * 0.1).toFixed(2) : 0,
    });
    if (frame >= total) { clearInterval(id); onDone(); }
  }, 200);
  return () => clearInterval(id);
};
// ─────────────────────────────────────────────────────────────────────────

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
  .feed-body{height:300px;display:flex;align-items:center;justify-content:center;position:relative;background:${C.bgDeep};overflow:hidden}
  .feed-img{width:100%;height:100%;object-fit:cover;display:block}
  .feed-ph{text-align:center;color:${C.textMuted}}
  .feed-ph svg{margin-bottom:10px;opacity:0.3}
  .feed-ph p{font-size:12px}
  .corner-tl{position:absolute;top:8px;left:8px;width:14px;height:14px;border-top:2px solid ${C.borderGlow};border-left:2px solid ${C.borderGlow};border-radius:2px 0 0 0;opacity:0.5}
  .corner-br{position:absolute;bottom:8px;right:8px;width:14px;height:14px;border-bottom:2px solid ${C.borderGlow};border-right:2px solid ${C.borderGlow};border-radius:0 0 2px 0;opacity:0.5}
  .scan{position:absolute;left:0;right:0;height:2px;background:linear-gradient(90deg,transparent,${C.cyan},transparent);animation:scan 3s linear infinite;opacity:0.35;pointer-events:none}
  @keyframes scan{0%{top:0}100%{top:100%}}
  .polyp-box{position:absolute;top:30%;left:35%;width:28%;height:35%;border:2px solid ${C.red};border-radius:4px;box-shadow:0 0 12px rgba(239,68,68,0.5)}
  .polyp-lbl{position:absolute;top:-20px;left:0;font-family:'JetBrains Mono',monospace;font-size:10px;background:${C.red};color:#fff;padding:2px 8px;border-radius:3px}
  .seg-fill{position:absolute;top:30%;left:35%;width:28%;height:35%;background:rgba(239,68,68,0.22);border-radius:4px}

  .metrics{display:grid;grid-template-columns:repeat(5,1fr);gap:12px}
  .mc{background:${C.panel};border:1px solid ${C.border};border-radius:12px;padding:16px 20px;transition:border-color 0.2s}
  .mc:hover{border-color:${C.borderGlow}}
  .ml{font-size:10px;font-weight:700;letter-spacing:0.14em;color:${C.textMuted};margin-bottom:8px}
  .mv{font-family:'JetBrains Mono',monospace;font-size:26px;font-weight:600;line-height:1}
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

  .progress-wrap{position:absolute;bottom:0;left:0;right:0;height:3px;background:rgba(255,255,255,0.05)}
  .progress-fill{height:100%;background:linear-gradient(90deg,${C.accent},${C.cyan});transition:width 0.2s}

  .img-result-row{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-top:2px}
  .result-info{display:flex;flex-direction:column;gap:8px;padding:12px 16px;border-top:1px solid ${C.border};background:rgba(0,0,0,0.2)}
  .ri-row{display:flex;justify-content:space-between;font-size:12px}
  .ri-lbl{color:${C.textMuted};letter-spacing:0.06em;font-size:10px;font-weight:600}
  .ri-val{font-family:'JetBrains Mono',monospace;color:${C.accentBr}}
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

  // video state
  const [videoFile, setVideoFile]     = useState(null);
  const [streaming, setStreaming]     = useState(false);
  const [streamDone, setStreamDone]   = useState(false);
  const [currentFrame, setFrame]      = useState(null);
  const [processedFrame, setProcessed]= useState(null);
  const [progress, setProgress]       = useState(0);
  const [frameInfo, setFrameInfo]     = useState(null);
  const stopRef                       = useRef(null);

  // image state
  const [imageFile, setImageFile]   = useState(null);
  const [imagePreview, setPreview]  = useState(null);
  const [analyzing, setAnalyzing]   = useState(false);
  const [imageResult, setResult]    = useState(null);

  // shared metrics
  const [metrics, setMetrics] = useState({ fps: "0.0", polyps: 0, confidence: "—", size: "—", latency: "—" });

  const videoRef = useRef();
  const imageRef = useRef();
  const fpsRef   = useRef({ count: 0, start: Date.now() });

  // ── Video handlers ───────────────────────────────────────────────────
  const handleVideoFile = (e) => {
    const f = e.target.files[0];
    if (!f) return;
    setVideoFile(f);
    setFrame(URL.createObjectURL(f));
    setProcessed(null);
    setStreamDone(false);
    setProgress(0);
    setFrameInfo(null);
  };

  const startVideoStream = () => {
    if (!videoFile) return;
    setStreaming(true);
    setStreamDone(false);
    setProgress(0);
    fpsRef.current = { count: 0, start: Date.now() };

    // Replace mockStreamVideo with real WebSocket:
    // const ws = new WebSocket("ws://localhost:8000/ws/video");
    // ws.onopen = async () => ws.send(await videoFile.arrayBuffer());
    // ws.onmessage = (e) => handleVideoFrame(JSON.parse(e.data));
    // ws.onclose = () => { setStreaming(false); setStreamDone(true); };
    // stopRef.current = () => ws.close();

    stopRef.current = mockStreamVideo(
      videoFile,
      (frame) => {
        fpsRef.current.count++;
        const elapsed = (Date.now() - fpsRef.current.start) / 1000;
        setProcessed(frame.frame_url);
        setFrameInfo(frame);
        setProgress((frame.frame_index / frame.total_frames) * 100);
        setMetrics({
          fps:        (fpsRef.current.count / elapsed).toFixed(1),
          polyps:     frame.polyps_found ? 1 : 0,
          confidence: frame.polyps_found ? `${(frame.confidence * 100).toFixed(0)}%` : "—",
          size:       frame.polyps_found ? "108.2" : "—",
          latency:    (12 + Math.random() * 6).toFixed(1),
        });
      },
      () => {
        setStreaming(false);
        setStreamDone(true);
        setProgress(100);
      }
    );
  };

  const stopStream = () => {
    if (stopRef.current) stopRef.current();
    setStreaming(false);
  };

  const resetVideo = () => {
    stopStream();
    setVideoFile(null);
    setFrame(null);
    setProcessed(null);
    setProgress(0);
    setStreamDone(false);
    setFrameInfo(null);
    setMetrics({ fps: "0.0", polyps: 0, confidence: "—", size: "—", latency: "—" });
  };

  // ── Image handlers ───────────────────────────────────────────────────
  const handleImageFile = (e) => {
    const f = e.target.files[0];
    if (!f) return;
    setImageFile(f);
    setPreview(URL.createObjectURL(f));
    setResult(null);
    setMetrics({ fps: "0.0", polyps: 0, confidence: "—", size: "—", latency: "—" });
  };

  const analyzeImage = async () => {
    if (!imageFile) return;
    setAnalyzing(true);
    const t0 = Date.now();

    // Replace mockAnalyzeImage with real API:
    // const fd = new FormData(); fd.append("file", imageFile);
    // const res = await fetch("http://localhost:8000/api/analyze", { method: "POST", body: fd });
    // const data = await res.json();

    const data = await mockAnalyzeImage(imageFile);
    const latency = Date.now() - t0;
    setResult(data);
    setAnalyzing(false);
    setMetrics({
      fps:        "—",
      polyps:     data.confidence > 0 ? 1 : 0,
      confidence: `${(data.confidence * 100).toFixed(0)}%`,
      size:       data.size_mm2,
      latency:    latency,
    });
  };

  const resetImage = () => {
    setImageFile(null);
    setPreview(null);
    setResult(null);
    setMetrics({ fps: "0.0", polyps: 0, confidence: "—", size: "—", latency: "—" });
  };

  // ── Connect ──────────────────────────────────────────────────────────
  const connect = () => {
    setConnected(true);
    if (mode === "video" && videoFile) startVideoStream();
  };

  const disconnect = () => {
    setConnected(false);
    stopStream();
    setMetrics({ fps: "0.0", polyps: 0, confidence: "—", size: "—", latency: "—" });
  };

  const switchMode = (m) => {
    if (m === mode) return;
    stopStream();
    setMode(m);
    resetVideo();
    resetImage();
    setConnected(false);
  };

  // ── Derived UI state ─────────────────────────────────────────────────
  const isVideoActive  = mode === "video" && streaming;
  const isImageResult  = mode === "image" && imageResult;
  const showPolyp      = frameInfo?.polyps_found;
  const origSrc        = mode === "video" ? currentFrame : imagePreview;
  const procSrc        = mode === "video" ? processedFrame : (imageResult?.annotated_image || null);

  return (
    <>
      <style>{css}</style>
      <div className="app">

        {/* Header */}
        <header className="hdr">
          <div className="hdr-dot" />
          <span className="hdr-name">ENDO-X</span>
          <span className="hdr-tag">AI POLYP DETECTION SYSTEM</span>
          <div className="hdr-right">
            <div className="status-pill">
              <div className={`sdot ${connected ? "on" : "off"}`} />
              <span style={{ color: connected ? C.green : C.red }}>
                {connected ? "Connected" : "Disconnected"}
              </span>
            </div>
            <Clock />
          </div>
        </header>

        <div className="body">

          {/* Feeds */}
          <div className="feeds">

            {/* Original Feed */}
            <div className="feed-card">
              <div className="corner-tl" /><div className="corner-br" />
              <div className="feed-hdr">
                <span className="feed-lbl">ORIGINAL FEED</span>
                {isVideoActive
                  ? <span className="badge b-live">LIVE</span>
                  : <span className="b-std">STANDBY</span>}
              </div>
              <div className="feed-body">
                {origSrc
                  ? <img src={origSrc} alt="original" className="feed-img" />
                  : (
                    <div className="feed-ph">
                      <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke={C.textMuted} strokeWidth="1.5">
                        <rect x="2" y="7" width="20" height="15" rx="2"/>
                        <path d="M16 3l4 4-4 4"/>
                      </svg>
                      <p>No signal</p>
                    </div>
                  )}
                {isVideoActive && <div className="scan" />}
                {isVideoActive && (
                  <div className="progress-wrap">
                    <div className="progress-fill" style={{ width: `${progress}%` }} />
                  </div>
                )}
              </div>
            </div>

            {/* AI Processed */}
            <div className="feed-card">
              <div className="corner-tl" /><div className="corner-br" />
              <div className="feed-hdr">
                <span className="feed-lbl">AI PROCESSED</span>
                {isVideoActive
                  ? <span className={`badge ${showPolyp ? "b-warn" : "b-ai"}`}>
                      {showPolyp ? "POLYP DETECTED" : "AI ACTIVE"}
                    </span>
                  : isImageResult
                  ? <span className="badge b-ai">RESULT</span>
                  : <span className="b-std">STANDBY</span>}
              </div>
              <div className="feed-body">
                {procSrc
                  ? (
                    <>
                      <img src={procSrc} alt="processed" className="feed-img" />
                      {showPolyp && (
                        <>
                          <div className="seg-fill" />
                          <div className="polyp-box">
                            <span className="polyp-lbl">POLYP {metrics.confidence}</span>
                          </div>
                        </>
                      )}
                      {isVideoActive && <div className="scan" style={{ animationDelay: "1.5s" }} />}
                    </>
                  )
                  : analyzing
                  ? (
                    <div className="feed-ph">
                      <svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke={C.accent} strokeWidth="1.5" style={{ animation: "dotP 1s linear infinite" }}>
                        <circle cx="12" cy="12" r="10"/>
                        <path d="M12 6v6l4 2"/>
                      </svg>
                      <p style={{ color: C.accent, marginTop: 8 }}>Analyzing...</p>
                    </div>
                  )
                  : (
                    <div className="feed-ph">
                      <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke={C.textMuted} strokeWidth="1.5">
                        <circle cx="12" cy="12" r="10"/>
                        <path d="M12 8v4l3 3"/>
                      </svg>
                      <p>Awaiting input</p>
                    </div>
                  )}
                {isVideoActive && (
                  <div className="progress-wrap">
                    <div className="progress-fill" style={{ width: `${progress}%` }} />
                  </div>
                )}
              </div>
              {isImageResult && (
                <div className="result-info">
                  <div className="ri-row">
                    <span className="ri-lbl">SIZE</span>
                    <span className="ri-val">{imageResult.size_mm2} mm²</span>
                    <span className="ri-lbl">DIAMETER</span>
                    <span className="ri-val">{imageResult.diameter_mm} mm</span>
                    <span className="ri-lbl">CONFIDENCE</span>
                    <span className="ri-val" style={{ color: C.green }}>{(imageResult.confidence * 100).toFixed(0)}%</span>
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Metrics */}
          <div className="metrics">
            <div className="mc"><div className="ml">FPS</div><div className="mv acc">{metrics.fps}</div></div>
            <div className="mc"><div className="ml">POLYPS</div><div className={`mv ${metrics.polyps > 0 ? "red" : ""}`}>{metrics.polyps}</div></div>
            <div className="mc"><div className="ml">CONFIDENCE</div><div className={`mv ${metrics.confidence !== "—" ? "grn" : ""}`}>{metrics.confidence}</div></div>
            <div className="mc">
              <div className="ml">SIZE (mm²)</div>
              <div className="mv acc">
                {metrics.size}{metrics.size !== "—" && <span className="mu">mm²</span>}
              </div>
            </div>
            <div className="mc">
              <div className="ml">LATENCY</div>
              <div className="mv">
                {metrics.latency}{metrics.latency !== "—" && <span className="mu">ms</span>}
              </div>
            </div>
          </div>

          {/* Controls */}
          <div className="controls">

            {/* Connect / Disconnect */}
            <button className="cb primary" onClick={connect} disabled={connected}>
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5"><path d="M13 10V3L4 14h7v7l9-11h-7z"/></svg>
              Connect
            </button>
            <button className="cb danger" onClick={disconnect} disabled={!connected}>
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5"><rect x="3" y="3" width="18" height="18" rx="2"/></svg>
              Disconnect
            </button>

            <div className="cdiv" />

            {/* VIDEO mode controls */}
            {mode === "video" && (
              <>
                <button className="cb warn" onClick={() => videoRef.current.click()}>
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5"><path d="M15 10l4.553-2.276A1 1 0 0121 8.723v6.554a1 1 0 01-1.447.894L15 14M3 8a2 2 0 012-2h8a2 2 0 012 2v8a2 2 0 01-2 2H5a2 2 0 01-2-2V8z"/></svg>
                  Upload Video
                </button>
                <input ref={videoRef} type="file" accept="video/*" style={{ display: "none" }} onChange={handleVideoFile} />
                <span className="fname">{videoFile ? videoFile.name : "No video loaded"}</span>
                {connected && videoFile && !streaming && !streamDone && (
                  <button className="cb primary" onClick={startVideoStream}>Start</button>
                )}
                {streaming && (
                  <button className="cb danger" onClick={stopStream}>Stop</button>
                )}
                {(videoFile || streaming || streamDone) && (
                  <button className="cb warn" onClick={resetVideo} style={{ opacity: 0.7 }}>Reset</button>
                )}
              </>
            )}

            {/* IMAGE mode controls */}
            {mode === "image" && (
              <>
                <button className="cb warn" onClick={() => imageRef.current.click()}>
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5"><rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="8.5" cy="8.5" r="1.5"/><path d="M21 15l-5-5L5 21"/></svg>
                  Upload Image
                </button>
                <input ref={imageRef} type="file" accept="image/*" style={{ display: "none" }} onChange={handleImageFile} />
                <span className="fname">{imageFile ? imageFile.name : "No image loaded"}</span>
                {imageFile && !imageResult && !analyzing && (
                  <button className="cb primary" onClick={analyzeImage}>Analyze</button>
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
