// ── Replace this entire file with real API calls when backend is ready ──

export const analyzeImage = async (file) => {
  // MOCK — delete and replace with:
  // const formData = new FormData();
  // formData.append("image", file);
  // const res = await fetch("/api/analyze", { method: "POST", body: formData });
  // return await res.json();

  await new Promise((r) => setTimeout(r, 2000));
  return {
    annotated_image:    URL.createObjectURL(file),
    segmentation_mask:  URL.createObjectURL(file),
    size_mm2:           124.5,
    diameter_mm:        12.6,
    confidence:         0.94,
  };
};
