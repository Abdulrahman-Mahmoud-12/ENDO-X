import { useState, useRef } from "react";
import { streamVideo } from "../api/videoApi";
import { UploadZone, Btn } from "../components/UI";
import VideoPlayer from "../components/VideoPlayer";
import COLORS from "../styles/colors";

export default function VideoStream() {
  const [file, setFile]               = useState(null);
  const [streaming, setStreaming]     = useState(false);
  const [done, setDone]               = useState(false);
  const [currentFrame, setFrame]      = useState(null);
  const [progress, setProgress]       = useState(0);
  const [frameInfo, setFrameInfo]     = useState(null);
  const stopRef                       = useRef(null);
  const [drag, setDrag]               = useState(false);

  const handleFile = (f) => {
    if (!f || !f.type.startsWith("video/")) return;
    setFile(f);
    setFrame(null);
    setProgress(0);
    setDone(false);
    setFrameInfo(null);
  };

  const startStream = () => {
    if (!file) return;
    setStreaming(true);
    setDone(false);
    setProgress(0);

    stopRef.current = streamVideo(
      file,
      (frame) => {
        setFrame(frame.frame_url);
        setFrameInfo(frame);
        setProgress((frame.frame_index / frame.total_frames) * 100);
      },
      () => {
        setStreaming(false);
        setDone(true);
        setProgress(100);
      }
    );
  };

  const stop = () => {
    if (stopRef.current) stopRef.current();
    setStreaming(false);
  };

  const reset = () => {
    stop();
    setFile(null);
    setFrame(null);
    setProgress(0);
    setDone(false);
    setFrameInfo(null);
  };

  if (!file) {
    return (
      <UploadZone
        icon="🎥"
        title="Drop endoscopy video here"
        sub="MP4, MOV supported — frames processed in real-time"
        accept="video/*"
        onFile={handleFile}
        drag={drag}
        onDrag={setDrag}
      />
    );
  }

  return (
    <div>
      <div style={{ display: "flex", gap: 16, alignItems: "center", marginBottom: 24 }}>
        <div style={{ fontSize: 32 }}>🎬</div>
        <div>
          <div style={{ fontSize: 14, fontWeight: 500 }}>{file.name}</div>
          <div style={{ fontSize: 12, color: COLORS.textMuted }}>
            {(file.size / 1024 / 1024).toFixed(1)} MB
          </div>
        </div>
        <div style={{ marginLeft: "auto", display: "flex", gap: 10 }}>
          <Btn variant="ghost" onClick={reset}>Reset</Btn>
          {!streaming && !done && (
            <Btn onClick={startStream}>Start Stream</Btn>
          )}
          {streaming && (
            <Btn
              variant="ghost"
              onClick={stop}
              style={{ borderColor: "#ef4444", color: "#ef4444" }}
            >
              Stop
            </Btn>
          )}
          {done && <Btn onClick={reset}>New Video</Btn>}
        </div>
      </div>

      {(streaming || currentFrame || done) && (
        <VideoPlayer
          currentFrame={currentFrame}
          frameInfo={frameInfo}
          progress={progress}
          streaming={streaming}
          done={done}
        />
      )}
    </div>
  );
}
