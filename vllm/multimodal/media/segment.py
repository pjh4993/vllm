# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Window specs carried on a video URL, and decoding for them.

A multi-turn agent reading a long video asks about a different window each
turn.  Today the caller clips the window itself and uploads the result, which
means the uploaded file is a fresh temporary every turn.  vLLM identifies
multi-modal inputs by hashing what it decoded, so two turns asking about the
same window are two unrelated cache entries and the clip has to be decoded
before its identity is even known.

Declaring the window on the URL fixes both halves::

    file:///tmp/clip.mp4#src=lecture-42.mp4&t=125.4,135.4

``src`` names what the window was taken from and ``t`` names the window.  The
pair is the identity, so the uploaded file's path no longer matters, and a
repeat visit is answered without opening the upload at all.

Windows are snapped to a grid first.  Two turns asking for 125.4-135.4 s and
125.9-135.7 s then decode to the identical frame array, which is what lets
them share one cache entry.  The default grid is one second, the span of a
single temporal patch at the usual 2 fps sampling, so snapping never moves a
frame across a patch boundary.

Omitting ``src`` means the URL points at the full video rather than a clip; the
window is then decoded from it by seeking, which costs the same wherever the
window sits instead of growing with its offset.
"""

import math
import os
import re
from dataclasses import dataclass
from typing import Any
from urllib.parse import unquote, urlparse

import numpy as np
import numpy.typing as npt

from vllm.logger import init_logger

logger = init_logger(__name__)

# `#t=start,end` follows the W3C Media Fragments syntax; `#segment=` is
# accepted as a more explicit spelling.
_FRAG = re.compile(r"(?:^|[#&])(?:t|segment)=([0-9.]+),([0-9.]+)")
_SRC = re.compile(r"(?:^|[#&])src=([^#&]+)")

DEFAULT_GRID_S = 1.0
DEFAULT_SAMPLE_FPS = 2.0


@dataclass(frozen=True)
class Segment:
    """A half-open window of a video, in seconds."""

    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start

    def quantized(self, grid: float) -> "Segment":
        """Snap outward to *grid* so the window never loses content."""
        if grid <= 0:
            return self
        return Segment(
            start=math.floor(self.start / grid) * grid,
            end=math.ceil(self.end / grid) * grid,
        )

    def key(self) -> str:
        return f"{self.start:.3f}-{self.end:.3f}"


def split_url(url: str) -> tuple[str, Segment | None]:
    """Return the URL without its window spec, plus the window if present."""
    match = _FRAG.search(url)
    if match is None:
        return url, None
    start, end = float(match.group(1)), float(match.group(2))
    if end <= start:
        logger.warning("Ignoring non-positive video window in %s", url[-60:])
        return url.split("#", 1)[0], None
    return url.split("#", 1)[0], Segment(start, end)


def declared_source(url: str) -> str | None:
    """The `#src=` identity, if the caller declared one."""
    match = _SRC.search(url)
    return unquote(match.group(1)) if match is not None else None


def local_path(url: str) -> str | None:
    """Filesystem path for a `file://` URL, else None."""
    parsed = urlparse(url)
    if parsed.scheme not in ("file", ""):
        return None
    return unquote(parsed.path) if parsed.scheme == "file" else url


def _metadata(
    frames: npt.NDArray, duration: float, sample_fps: float, backend: str
) -> dict[str, Any]:
    return {
        "total_num_frames": int(frames.shape[0]),
        "fps": sample_fps,
        "duration": duration,
        "video_backend": backend,
        "frames_indices": list(range(int(frames.shape[0]))),
        # Frames are already sampled here; re-sampling would drop content.
        "do_sample_frames": False,
        "height": int(frames.shape[1]),
        "width": int(frames.shape[2]),
    }


def _sample_times(start: float, duration: float, sample_fps: float,
                  min_frames: int, max_frames: int) -> npt.NDArray:
    count = int(round(duration * sample_fps))
    count = max(min_frames, min(count, max_frames))
    return start + np.linspace(0.0, duration, count, endpoint=False)


def _collect(container, stream, want: npt.NDArray) -> list[npt.NDArray]:
    """Convert only the frames that survive sampling.

    Decoding a ten second clip yields a few hundred frames while a couple of
    dozen are kept; converting all of them to RGB costs several hundred
    milliseconds and is thrown away immediately.
    """
    picked: list[npt.NDArray] = []
    index = 0
    last: npt.NDArray | None = None
    for frame in container.decode(stream):
        if index >= len(want):
            break
        if frame.pts is None:
            continue
        timestamp = float(frame.pts * stream.time_base)
        converted = None
        while index < len(want) and timestamp >= want[index]:
            if converted is None:
                converted = frame.to_ndarray(format="rgb24")
            picked.append(converted)
            index += 1
        if converted is not None:
            last = converted
    # A file that ends early pads with its last frame, so the array shape stays
    # a function of the declared window alone.
    while index < len(want):
        if last is None:
            raise RuntimeError("no frames decoded")
        picked.append(last)
        index += 1
    return picked


def decode_window(
    path: str,
    segment: Segment,
    *,
    sample_fps: float = DEFAULT_SAMPLE_FPS,
    min_frames: int = 4,
    max_frames: int = 768,
) -> tuple[npt.NDArray, dict[str, Any]]:
    """Decode *segment* out of the full video at *path* by seeking to it."""
    import av

    container = av.open(path)
    try:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        if stream.time_base is None:
            raise RuntimeError(f"{path} has no time base")
        want = _sample_times(segment.start, segment.duration, sample_fps,
                             min_frames, max_frames)
        # seek() lands on the keyframe at or before the target; decoding
        # forward from there is what keeps the cost independent of where the
        # window sits in the file.
        container.seek(int(segment.start / float(stream.time_base)),
                       stream=stream, any_frame=False, backward=True)
        frames = np.stack(_collect(container, stream, want))
    finally:
        container.close()
    return frames, _metadata(frames, segment.duration, sample_fps, "vllm_window")


def decode_clip(
    path: str,
    *,
    sample_fps: float = DEFAULT_SAMPLE_FPS,
    min_frames: int = 4,
    max_frames: int = 768,
) -> tuple[npt.NDArray, dict[str, Any]]:
    """Decode a file that is already the window, end to end."""
    import av

    container = av.open(path)
    try:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        duration = 0.0
        if stream.duration and stream.time_base:
            duration = float(stream.duration * stream.time_base)
        if duration <= 0 and container.duration:
            duration = container.duration / 1_000_000.0
        if duration <= 0 and stream.frames:
            duration = stream.frames / float(stream.average_rate or 25)
        if duration <= 0:
            raise RuntimeError(f"cannot determine duration of {path}")
        want = _sample_times(0.0, duration, sample_fps, min_frames, max_frames)
        frames = np.stack(_collect(container, stream, want))
    finally:
        container.close()
    return frames, _metadata(frames, duration, sample_fps, "vllm_clip")


def file_identity(path: str) -> str:
    """Path plus mtime and size, so an edited file invalidates its entries."""
    try:
        stat = os.stat(path)
    except OSError:
        return path
    return f"{path}:{stat.st_mtime_ns}:{stat.st_size}"
