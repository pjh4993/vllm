# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The window cache: identity, isolation and the sandbox check."""

import numpy as np
import pytest

from vllm.multimodal.media import MEDIA_CONNECTOR_REGISTRY
from vllm.multimodal.media.segment_connector import (
    SegmentCacheMediaConnector, _WindowCache)


def test_connector_is_registered():
    """Importing the module must make the name selectable."""
    loaded = MEDIA_CONNECTOR_REGISTRY.load(
        "segment_cache", media_io_kwargs={},
        allowed_local_media_path=None, allowed_media_domains=[])
    assert isinstance(loaded, SegmentCacheMediaConnector)


def _cache(tmp_path, capacity_gb=1):
    return _WindowCache(str(tmp_path), int(capacity_gb * (1 << 30)))


def test_roundtrip_preserves_frames(tmp_path):
    cache = _cache(tmp_path)
    frames = np.arange(2 * 3 * 4 * 3, dtype=np.uint8).reshape(2, 3, 4, 3)
    meta = {"fps": 2.0, "total_num_frames": 2}
    key = cache.key_for("clip.mp4#1.000-2.000@2")
    cache.put(key, frames, meta)
    got = cache.get(key)
    assert got is not None
    assert np.array_equal(got[0], frames)
    assert got[1] == meta


def test_distinct_windows_do_not_collide(tmp_path):
    cache = _cache(tmp_path)
    a = cache.key_for("clip.mp4#1.000-2.000@2")
    b = cache.key_for("clip.mp4#2.000-3.000@2")
    assert a != b
    cache.put(a, np.zeros((1, 2, 2, 3), np.uint8), {})
    assert cache.get(b) is None


def test_missing_entry_reports_miss(tmp_path):
    cache = _cache(tmp_path)
    assert cache.get(cache.key_for("absent")) is None
    assert cache.misses == 1


def test_capacity_evicts_oldest(tmp_path):
    """A budget smaller than the corpus must not grow without bound."""
    frames = np.zeros((4, 16, 16, 3), np.uint8)
    cache = _WindowCache(str(tmp_path), capacity_bytes=frames.nbytes * 2)
    for i in range(6):
        cache.put(cache.key_for(f"w{i}"), frames, {})
    stored = list(tmp_path.rglob("*.npy"))
    assert len(stored) <= 3


def test_partial_write_is_not_visible(tmp_path):
    """A crashed writer leaves a temp file, never a readable half entry."""
    cache = _cache(tmp_path)
    key = cache.key_for("w")
    stray = tmp_path / key[:2]
    stray.mkdir(parents=True, exist_ok=True)
    (stray / f"{key}.npy.1234.tmp").write_bytes(b"garbage")
    assert cache.get(key) is None


def test_sandbox_rejects_paths_outside_root(tmp_path):
    connector = SegmentCacheMediaConnector(
        media_io_kwargs={}, allowed_local_media_path=str(tmp_path),
        allowed_media_domains=[])
    inside = tmp_path / "ok.mp4"
    inside.write_bytes(b"x")
    assert connector._checked_path(f"file://{inside}") == str(inside.resolve())
    with pytest.raises(ValueError, match="must be a subpath"):
        connector._checked_path("file:///etc/passwd")


def test_sandbox_requires_configuration(tmp_path):
    connector = SegmentCacheMediaConnector(
        media_io_kwargs={}, allowed_local_media_path=None,
        allowed_media_domains=[])
    with pytest.raises(RuntimeError, match="allowed-local-media-path"):
        connector._checked_path("file:///tmp/x.mp4")
