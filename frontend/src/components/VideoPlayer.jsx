import COLORS from "../styles/colors";
import { Spinner } from "./UI";

export default function VideoPlayer({ currentFrame, frameInfo, progress, streaming, done }) {
  return (
    <div className="video-panel">
      <div className="progress-bar">
        <div className="progress-fill" style={{ width: `${progress}%` }} />
      </div>

      <div className="video-stream">
        {currentFrame ? (
          <>
            <img src={currentFrame} alt="stream frame" className="stream-img" />
            {frameInfo && (
              <div className="frame-counter" style={{ color: COLORS.text }}>
                {frameInfo.frame_index} / {frameInfo.total_frames}
              </div>
            )}
          </>
        ) : (
          <div className="stream-placeholder">
            <Spinner label="Initializing stream..." />
          </div>
        )}
      </div>

      <div className="status-row">
        {streaming && <div className="pulse" />}
        {streaming && (
          <span style={{ color: COLORS.textMuted }}>
            Processing frame {frameInfo?.frame_index || 0} of {frameInfo?.total_frames || "..."}
          </span>
        )}
        {done && (
          <span style={{ color: COLORS.success }}>✓ Processing complete</span>
        )}
        {frameInfo && (
          <span style={{ marginLeft: "auto" }}>
            {frameInfo.polyps_found
              ? <span className="badge badge-polyp">⚠ Polyp detected</span>
              : <span className="badge badge-clear">✓ Clear</span>
            }
          </span>
        )}
      </div>
    </div>
  );
}
