import COLORS from "../styles/colors";

export function UploadZone({ icon, title, sub, accept, onFile, drag, onDrag }) {
  const inputRef = { current: null };

  const handleDrop = (e) => {
    e.preventDefault();
    onDrag(false);
    const f = e.dataTransfer.files[0];
    if (f) onFile(f);
  };

  return (
    <div
      className={`upload-zone ${drag ? "drag" : ""}`}
      onClick={() => document.getElementById(`upload-${accept}`).click()}
      onDragOver={(e) => { e.preventDefault(); onDrag(true); }}
      onDragLeave={() => onDrag(false)}
      onDrop={handleDrop}
    >
      <div className="upload-icon">{icon}</div>
      <div className="upload-title">{title}</div>
      <div className="upload-sub">{sub}</div>
      <input
        id={`upload-${accept}`}
        type="file"
        accept={accept}
        style={{ display: "none" }}
        onChange={(e) => onFile(e.target.files[0])}
      />
    </div>
  );
}

export function Spinner({ label = "Processing..." }) {
  return (
    <div style={{ textAlign: "center", padding: 60 }}>
      <div className="spinner" />
      <div style={{ color: COLORS.textMuted, fontSize: 14 }}>{label}</div>
    </div>
  );
}

export function Btn({ children, variant = "primary", onClick, disabled, style }) {
  return (
    <button
      className={`btn btn-${variant}`}
      onClick={onClick}
      disabled={disabled}
      style={style}
    >
      {children}
    </button>
  );
}
