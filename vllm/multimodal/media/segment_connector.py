# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A media connector that understands video windows and caches them on disk.

Select it with ``VLLM_MEDIA_CONNECTOR=segment_cache`` and configure it through
``--media-io-kwargs``::

    --media-io-kwargs '{"video": {"segment_cache_dir": "/local/nvme/videowin",
                                  "segment_cache_gb": 200,
                                  "segment_grid_s": 1.0}}'

Requests without a window spec are served by the default connector, so the
option is inert for workloads that do not use one.
"""

import hashlib
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from vllm.logger import init_logger

from .base import MediaWithBytes
from .connector import MEDIA_CONNECTOR_REGISTRY, MediaConnector
from .segment import (DEFAULT_GRID_S, DEFAULT_SAMPLE_FPS, Segment, decode_clip,
                      decode_window, declared_source, file_identity,
                      local_path, split_url)

logger = init_logger(__name__)


class _WindowCache:
    """Decoded windows on local disk, one file per entry.

    Writes go to a temporary name and are renamed into place, so a reader sees
    either a complete entry or none, and a crashed writer leaves no half entry
    that would be read back as garbage frames.
    """

    def __init__(self, root: str, capacity_bytes: int, io_threads: int = 4):
        self._root = root
        self._capacity = capacity_bytes
        self._pool = ThreadPoolExecutor(max_workers=io_threads,
                                        thread_name_prefix="video-window")
        self._lock = threading.Lock()
        os.makedirs(root, exist_ok=True)
        self.hits = 0
        self.misses = 0

    @staticmethod
    def key_for(identity: str) -> str:
        return hashlib.sha256(identity.encode()).hexdigest()[:32]

    def _paths(self, key: str) -> tuple[str, str]:
        directory = os.path.join(self._root, key[:2])
        return (os.path.join(directory, key + ".npy"),
                os.path.join(directory, key + ".json"))

    def get(self, key: str) -> tuple[npt.NDArray, dict[str, Any]] | None:
        frames_path, meta_path = self._paths(key)
        try:
            # mmap keeps the read lazy: the processor slices the array, so only
            # the pages it touches are faulted in.
            frames = np.load(frames_path, mmap_mode="r")
            with open(meta_path) as handle:
                metadata = json.load(handle)
        except (OSError, ValueError):
            self.misses += 1
            return None
        self.hits += 1
        return np.asarray(frames), metadata

    def put(self, key: str, frames: npt.NDArray,
            metadata: dict[str, Any]) -> None:
        frames_path, meta_path = self._paths(key)
        os.makedirs(os.path.dirname(frames_path), exist_ok=True)
        suffix = f".{os.getpid()}.{threading.get_ident()}.tmp"
        try:
            with open(frames_path + suffix, "wb") as handle:
                np.save(handle, frames)
            os.replace(frames_path + suffix, frames_path)
            with open(meta_path + suffix, "w") as handle:
                json.dump(metadata, handle)
            os.replace(meta_path + suffix, meta_path)
        except OSError as exc:
            logger.warning("Video window cache write failed for %s: %s",
                           key[:12], exc)
            for stale in (frames_path + suffix, meta_path + suffix):
                try:
                    os.unlink(stale)
                except OSError:
                    pass
            return
        self._trim()

    def _trim(self) -> None:
        """Drop the oldest entries once the directory exceeds its budget."""
        if self._capacity <= 0:
            return
        with self._lock:
            entries: list[tuple[float, int, str]] = []
            total = 0
            for dirpath, _, names in os.walk(self._root):
                for name in names:
                    if not name.endswith(".npy"):
                        continue
                    full = os.path.join(dirpath, name)
                    try:
                        stat = os.stat(full)
                    except OSError:
                        continue
                    entries.append((stat.st_mtime, stat.st_size, full))
                    total += stat.st_size
            if total <= self._capacity:
                return
            entries.sort()
            for _mtime, size, full in entries:
                if total <= self._capacity:
                    break
                for path in (full, full[: -len(".npy")] + ".json"):
                    try:
                        os.unlink(path)
                    except OSError:
                        pass
                total -= size

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False)


_cache: _WindowCache | None = None
_cache_lock = threading.Lock()


def _get_cache(config: dict[str, Any]) -> _WindowCache | None:
    """The process-wide cache, built on first use.

    One cache per process rather than per connector: a connector instance is
    created per request, so per-instance state would never be reused.
    """
    global _cache
    directory = config.get("segment_cache_dir") or os.environ.get(
        "VLLM_VIDEO_SEGMENT_CACHE_DIR")
    if not directory:
        return None
    if _cache is None:
        with _cache_lock:
            if _cache is None:
                capacity = float(config.get("segment_cache_gb", 50))
                _cache = _WindowCache(directory, int(capacity * (1 << 30)))
                logger.info("Video window cache at %s (budget %.1f GiB)",
                            directory, capacity)
    return _cache


@MEDIA_CONNECTOR_REGISTRY.register("segment_cache")
class SegmentCacheMediaConnector(MediaConnector):
    """Default connector plus window-aware fetching and a disk cache."""

    def _video_config(self) -> dict[str, Any]:
        return dict(self.media_io_kwargs.get("video", {}))

    def _checked_path(self, url: str) -> str:
        """Resolve a local URL under the configured sandbox.

        This path does not go through ``load_from_url``, so the check that
        normally guards local reads has to be repeated here.
        """
        path = local_path(url)
        if path is None:
            raise ValueError("video windows require a local file URL")
        allowed = self.allowed_local_media_path
        if allowed is None:
            raise RuntimeError(
                "Cannot load local files without `--allowed-local-media-path`.")
        resolved = Path(path).resolve()
        if Path(allowed) not in resolved.parents:
            raise ValueError(
                f"The file path {resolved} must be a subpath of "
                f"`--allowed-local-media-path {allowed}`.")
        return str(resolved)

    def _fetch_window(self, video_url: str):
        """Serve a windowed request, or return None if this is not one."""
        base, segment = split_url(video_url)
        if segment is None:
            return None

        config = self._video_config()
        grid = float(config.get("segment_grid_s", DEFAULT_GRID_S))
        sample_fps = float(config.get("segment_fps", DEFAULT_SAMPLE_FPS))
        window = segment.quantized(grid)
        source = declared_source(video_url)

        # The identity is the declared source plus the quantized window when the
        # caller clipped it, and the file plus the window when it did not. Either
        # way it is stable across turns and a few dozen bytes long, so vLLM
        # hashes it instead of a multi-megabyte frame array.
        if source is not None:
            identity = f"{source}#{window.key()}@{sample_fps:g}"
        else:
            identity = f"{file_identity(self._checked_path(base))}" \
                       f"#{window.key()}@{sample_fps:g}"
        identity_bytes = identity.encode()

        cache = _get_cache(config)
        if cache is not None:
            hit = cache.get(cache.key_for(identity))
            if hit is not None:
                frames, metadata = hit
                return MediaWithBytes(media=(frames, metadata),
                                      original_bytes=identity_bytes)

        path = self._checked_path(base)
        if source is not None:
            frames, metadata = decode_clip(path, sample_fps=sample_fps)
        else:
            frames, metadata = decode_window(path, window,
                                             sample_fps=sample_fps)
        if cache is not None:
            cache.put(cache.key_for(identity), frames, metadata)
        return MediaWithBytes(media=(frames, metadata),
                              original_bytes=identity_bytes)

    def fetch_video(self, video_url: str, *, image_mode: str | None = "RGB",
                    video_processor: str | None = None):
        if video_url:
            windowed = self._fetch_window(video_url)
            if windowed is not None:
                return windowed
        return super().fetch_video(video_url, image_mode=image_mode,
                                   video_processor=video_processor)

    async def fetch_video_async(self, video_url: str, *,
                                image_mode: str | None = "RGB",
                                video_processor: str | None = None):
        # Both entry points are overridden: the OpenAI server parses chat
        # content asynchronously and calls the async one, so overriding only the
        # synchronous variant leaves the cache dead.
        if video_url:
            windowed = self._fetch_window(video_url)
            if windowed is not None:
                return windowed
        return await super().fetch_video_async(
            video_url, image_mode=image_mode, video_processor=video_processor)
