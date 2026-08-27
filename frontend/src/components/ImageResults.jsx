import COLORS from "../styles/colors";

export default function ImageResults({ result }) {
  return (
    <div>
      <div className="results-grid">
        <div className="result-card">
          <div className="result-card-header">
            <div className="dot" style={{ background: COLORS.accent }} />
            Annotated Image
          </div>
          <div className="img-wrap">
            <img src={result.annotated_image} alt="annotated" />
          </div>
        </div>

        <div className="result-card">
          <div className="result-card-header">
            <div className="dot" style={{ background: "#ef4444" }} />
            Segmentation Mask
          </div>
          <div className="img-wrap">
            <img src={result.segmentation_mask} alt="segmentation" />
            <div className="seg-overlay" />
          </div>
        </div>
      </div>

      <div style={{ marginTop: 24 }}>
        <div className="result-card">
          <div className="result-card-header">
            <div className="dot" style={{ background: COLORS.success }} />
            Measurements
          </div>
          <div className="metrics">
            <div className="metric">
              <div className="metric-label">Area</div>
              <div className="metric-value">
                {result.size_mm2.toFixed(1)}
                <span className="metric-unit">mm²</span>
              </div>
            </div>
            <div className="metric">
              <div className="metric-label">Diameter</div>
              <div className="metric-value">
                {result.diameter_mm.toFixed(1)}
                <span className="metric-unit">mm</span>
              </div>
            </div>
            <div className="metric" style={{ gridColumn: "1 / -1" }}>
              <div className="metric-label">Model confidence</div>
              <div style={{ display: "flex", alignItems: "center", gap: 12, marginTop: 6 }}>
                <div className="confidence-bar" style={{ flex: 1 }}>
                  <div
                    className="confidence-fill"
                    style={{ width: `${result.confidence * 100}%` }}
                  />
                </div>
                <span style={{ fontFamily: "JetBrains Mono", fontSize: 14, color: COLORS.success }}>
                  {(result.confidence * 100).toFixed(0)}%
                </span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
