const API_BASE_URL = process.env.REACT_APP_API_URL || "http://localhost:8000";

export const analyzeImage = async (file, confidenceThreshold = null) => {
  const formData = new FormData();
  formData.append("file", file);

  let url = `${API_BASE_URL}/api/v1/predict/image`;
  if (confidenceThreshold !== null) {
    url += `?confidence_threshold=${confidenceThreshold}`;
  }

  const res = await fetch(url, {
    method: "POST",
    body: formData,
  });

  const data = await res.json();

  if (!res.ok || data.status === "error") {
    throw new Error(data.message || data.detail || `Upload failed with status ${res.status}`);
  }

  const detections = data.detections || [];
  const maxConf = detections.length > 0
    ? Math.max(...detections.map((d) => d.confidence))
    : 0;

  return {
    raw: data,
    status: data.status,
    annotated_image: data.overlay_image_url
      ? `${API_BASE_URL}${data.overlay_image_url}`
      : null,
    polyps_found: detections.length,
    confidence: maxConf,
    latency_ms: data.inference_time_ms || 0,
    detections: data.detections,
    segmentations: data.segmentations,
  };
};
