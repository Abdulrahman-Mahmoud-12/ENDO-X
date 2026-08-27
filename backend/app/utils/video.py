"""OpenCV-based video I/O helpers: frame extraction, frame writing, FPS
accounting, and format validation.

Kept dependency-light (just OpenCV) and framework-free, same rationale as
models/*.py — testable without FastAPI running.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from app.core.config import Settings, get_settings
from app.core.exceptions import UnsupportedVideoFormatError

# Codec choice per container. mp4v is the widely-available OpenCV/ffmpeg
# fourcc for .mp4 (H.264 via 'avc1' is nicer but not reliably present in
# every OpenCV build); XVID is the standard safe choice for .avi. .mov
# gets mp4v too since QuickTime containers commonly hold MPEG-4 video and
# OpenCV's mov muxing support is otherwise inconsistent across platforms.
_FOURCC_BY_EXTENSION = {
    ".mp4": "mp4v",
    ".mov": "mp4v",
    ".avi": "XVID",
}


def validate_video_extension(filename: str, settings: Settings | None = None) -> str:
    """Raise UnsupportedVideoFormatError if `filename`'s extension isn't
    in Settings.allowed_video_extensions. Returns the lowercased extension
    (with leading dot) on success."""
    settings = settings or get_settings()
    ext = Path(filename).suffix.lower()
    if ext not in settings.allowed_video_extensions_list:
        raise UnsupportedVideoFormatError(
            f"Unsupported video extension {ext!r}; allowed: {settings.allowed_video_extensions_list}"
        )
    return ext


class FrameReader:
    """Context-manager wrapper around cv2.VideoCapture yielding RGB frames.

    Usage:
        with FrameReader(path) as reader:
            for frame in reader.frames():
                ...
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._cap: cv2.VideoCapture | None = None

    def __enter__(self) -> "FrameReader":
        self._cap = cv2.VideoCapture(str(self.path))
        if not self._cap.isOpened():
            raise UnsupportedVideoFormatError(
                f"OpenCV could not open {self.path.name} — the file may be "
                f"corrupt or use a codec that isn't supported even though "
                f"its extension is allowed."
            )
        return self

    def __exit__(self, *exc_info) -> None:
        if self._cap is not None:
            self._cap.release()

    @property
    def fps(self) -> float:
        fps = self._cap.get(cv2.CAP_PROP_FPS)
        # Some containers report 0/NaN FPS in their metadata; fall back to
        # a sane default rather than propagating a broken value into the
        # output VideoWriter (which would otherwise produce a 0-fps file).
        return fps if fps and fps > 0 else 25.0

    @property
    def frame_count(self) -> int:
        return int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

    @property
    def width(self) -> int:
        return int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    @property
    def height(self) -> int:
        return int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def frames(self) -> Iterator[np.ndarray]:
        """Yield frames as RGB uint8 (H, W, 3) arrays — matches
        Detector.predict()'s and Segmenter.predict()'s expected input
        color order, so callers never have to think about BGR vs RGB."""
        assert self._cap is not None, "use FrameReader as a context manager"
        while True:
            ok, frame_bgr = self._cap.read()
            if not ok:
                break
            yield cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


class FrameWriter:
    """Context-manager wrapper around cv2.VideoWriter, accepting RGB frames."""

    def __init__(self, path: str | Path, fps: float, frame_size: tuple[int, int]) -> None:
        self.path = Path(path)
        self.fps = fps
        self.frame_size = frame_size  # (width, height)
        self._writer: cv2.VideoWriter | None = None

    def __enter__(self) -> "FrameWriter":
        ext = self.path.suffix.lower()
        fourcc_str = _FOURCC_BY_EXTENSION.get(ext, "mp4v")
        fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._writer = cv2.VideoWriter(str(self.path), fourcc, self.fps, self.frame_size)
        if not self._writer.isOpened():
            raise UnsupportedVideoFormatError(
                f"OpenCV could not open a VideoWriter for {self.path.name} "
                f"(fourcc={fourcc_str!r}) — check that ffmpeg/OpenCV was "
                f"built with this codec available."
            )
        return self

    def __exit__(self, *exc_info) -> None:
        if self._writer is not None:
            self._writer.release()

    def write(self, frame_rgb: np.ndarray) -> None:
        assert self._writer is not None, "use FrameWriter as a context manager"
        self._writer.write(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))


class FPSAccountant:
    """Rolling average latency/FPS across processed frames, for the
    summary stats in the API response. Only frames actually run through
    the model (not sample_rate-skipped ones) should be `record()`ed."""

    def __init__(self) -> None:
        self._latencies_s: list[float] = []

    def record(self, latency_s: float) -> None:
        self._latencies_s.append(latency_s)

    @property
    def avg_latency_ms(self) -> float:
        if not self._latencies_s:
            return 0.0
        return 1000.0 * sum(self._latencies_s) / len(self._latencies_s)

    @property
    def avg_fps(self) -> float:
        avg_s = self.avg_latency_ms / 1000.0
        return (1.0 / avg_s) if avg_s > 0 else 0.0


class timed:
    """Tiny elapsed-time context manager: `with timed() as t: ...`; read
    `t.elapsed` (seconds) afterward."""

    def __enter__(self) -> "timed":
        self._start = time.perf_counter()
        self.elapsed = 0.0
        return self

    def __exit__(self, *exc_info) -> None:
        self.elapsed = time.perf_counter() - self._start
