const API_BASE_URL = process.env.REACT_APP_API_URL || "http://localhost:8000";

export const processVideo = async (file, sampleRate = 1) => {
  const formData = new FormData();
  formData.append("file", file);

  const url = `${API_BASE_URL}/api/v1/predict/video?sample_rate=${sampleRate}`;

  const res = await fetch(url, {
    method: "POST",
    body: formData,
  });

  const data = await res.json();

  if (!res.ok || data.status === "error") {
    throw new Error(data.message || data.detail || `Video processing failed with status ${res.status}`);
  }

  const summary = data.summary || {};

  return {
    raw: data,
    status: data.status,
    output_video_url: data.output_video_url
      ? `${API_BASE_URL}${data.output_video_url}`
      : null,
    total_frames: summary.total_frames || 0,
    frames_with_polyp: summary.frames_with_polyp || 0,
    avg_fps: summary.avg_fps || 0,
    avg_latency_ms: summary.avg_latency_ms || 0,
  };
};
