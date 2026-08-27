// ── Replace this entire file with real WebSocket when backend is ready ──

export const streamVideo = (file, onFrame, onDone) => {
  // MOCK — delete and replace with:
  // const ws = new WebSocket("ws://localhost:8000/ws/video");
  // ws.onopen = async () => { ws.send(await file.arrayBuffer()); };
  // ws.onmessage = (e) => { onFrame(JSON.parse(e.data)); };
  // ws.onclose = () => onDone();
  // return () => ws.close();

  let frame = 0;
  const total = 30;
  const url = URL.createObjectURL(file);

  const interval = setInterval(() => {
    frame++;
    onFrame({
      frame_url:    url,
      frame_index:  frame,
      total_frames: total,
      polyps_found: frame % 5 === 0,
    });
    if (frame >= total) {
      clearInterval(interval);
      onDone();
    }
  }, 200);

  return () => clearInterval(interval);
};
