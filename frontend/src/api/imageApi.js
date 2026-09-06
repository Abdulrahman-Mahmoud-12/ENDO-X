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
    headers: { "ngrok-skip-browser-warning": "true" },
    body: formData,
  });

  const data = await res.json();

  if (!res.ok || data.status === "error") {
    throw new Error(
      data.message || data.detail || `Upload failed with status ${res.status}`,
    );
  }

  const detections = data.detections || [];
  const maxConf =
    detections.length > 0
      ? Math.max(...detections.map((d) => d.confidence))
      : 0;

  let annotatedImageUrl = null;
  if (data.overlay_image_url) {
    const overlayResponse = await fetch(
      `${API_BASE_URL}${data.overlay_image_url}`,
      { headers: { "ngrok-skip-browser-warning": "true" } },
    );
    if (!overlayResponse.ok) {
      throw new Error(
        `Annotated image failed with status ${overlayResponse.status}`,
      );
    }
    const overlayBlob = await overlayResponse.blob();
    if (!overlayBlob.type.startsWith("image/")) {
      throw new Error("Backend returned an invalid annotated image");
    }
    annotatedImageUrl = URL.createObjectURL(overlayBlob);
  }

  return {
    raw: data,
    status: data.status,
    annotated_image: annotatedImageUrl,
    polyps_found: detections.length,
    confidence: maxConf,
    latency_ms: data.inference_time_ms || 0,
    detections: data.detections,
    segmentations: data.segmentations,
  };
};
