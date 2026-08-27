import { useState } from "react";
import { analyzeImage } from "../api/imageApi";
import { UploadZone, Spinner, Btn } from "../components/UI";
import ImageResults from "../components/ImageResults";
import COLORS from "../styles/colors";

export default function ImageAnalysis() {
  const [file, setFile]       = useState(null);
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult]   = useState(null);
  const [drag, setDrag]       = useState(false);

  const handleFile = (f) => {
    if (!f || !f.type.startsWith("image/")) return;
    setFile(f);
    setPreview(URL.createObjectURL(f));
    setResult(null);
  };

  const analyze = async () => {
    if (!file) return;
    setLoading(true);
    try {
      const data = await analyzeImage(file);
      setResult(data);
    } finally {
      setLoading(false);
    }
  };

  const reset = () => {
    setFile(null);
    setPreview(null);
    setResult(null);
  };

  if (!file) {
    return (
      <UploadZone
        icon="🔬"
        title="Drop endoscopy image here"
        sub="or click to browse — JPG, PNG supported"
        accept="image/*"
        onFile={handleFile}
        drag={drag}
        onDrag={setDrag}
      />
    );
  }

  return (
    <div>
      <div style={{ display: "flex", gap: 16, alignItems: "center", marginBottom: 24 }}>
        <img
          src={preview}
          alt="preview"
          style={{ width: 80, height: 60, objectFit: "cover", borderRadius: 8, border: `1px solid ${COLORS.border}` }}
        />
        <div>
          <div style={{ fontSize: 14, fontWeight: 500 }}>{file.name}</div>
          <div style={{ fontSize: 12, color: COLORS.textMuted }}>
            {(file.size / 1024).toFixed(0)} KB
          </div>
        </div>
        <div style={{ marginLeft: "auto", display: "flex", gap: 10 }}>
          <Btn variant="ghost" onClick={reset}>Reset</Btn>
          <Btn onClick={analyze} disabled={loading}>
            {loading ? "Analyzing..." : "Analyze"}
          </Btn>
        </div>
      </div>

      {loading && <Spinner label="Running detection model..." />}
      {result && <ImageResults result={result} />}
    </div>
  );
}
