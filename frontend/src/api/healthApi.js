const API_BASE_URL = process.env.REACT_APP_API_URL || "http://localhost:8000";

export const checkHealth = async () => {
  try {
    const res = await fetch(`${API_BASE_URL}/api/v1/health`, {
      method: "GET",
      headers: { Accept: "application/json" },
    });
    if (!res.ok) {
      throw new Error(`HTTP error ${res.status}`);
    }
    const data = await res.json();
    return { connected: true, data };
  } catch (err) {
    return { connected: false, error: err.message };
  }
};
