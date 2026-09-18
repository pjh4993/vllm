# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Window specs on video URLs, and the identity they produce."""

import numpy as np
import pytest

from vllm.multimodal.media.segment import (Segment, declared_source,
                                           file_identity, local_path,
                                           split_url)


def test_split_url_extracts_window():
    base, seg = split_url("file:///v.mp4#t=125.4,135.4")
    assert base == "file:///v.mp4"
    assert seg == Segment(125.4, 135.4)


def test_split_url_accepts_explicit_spelling():
    _, seg = split_url("file:///v.mp4#segment=10,20")
    assert seg == Segment(10.0, 20.0)


def test_split_url_without_window():
    base, seg = split_url("file:///v.mp4")
    assert base == "file:///v.mp4"
    assert seg is None


@pytest.mark.parametrize("spec", ["#t=20,10", "#t=5,5"])
def test_split_url_rejects_empty_window(spec):
    """A window that ends at or before it starts is dropped, not decoded."""
    _, seg = split_url("file:///v.mp4" + spec)
    assert seg is None


def test_quantize_snaps_outward():
    """Snapping outward keeps every frame the caller asked for."""
    assert Segment(125.4, 135.4).quantized(1.0) == Segment(125.0, 136.0)
    assert Segment(125.0, 135.0).quantized(1.0) == Segment(125.0, 135.0)


def test_nearby_windows_share_one_key():
    """This is what makes repeated turns reuse a single cache entry."""
    a = Segment(125.4, 135.4).quantized(1.0)
    b = Segment(125.9, 135.7).quantized(1.0)
    assert a == b
    assert a.key() == b.key()


def test_quantize_disabled_by_zero_grid():
    seg = Segment(1.25, 2.75)
    assert seg.quantized(0.0) == seg


def test_declared_source():
    url = "file:///tmp/clip.mp4#src=lecture-42.mp4&t=1,2"
    assert declared_source(url) == "lecture-42.mp4"
    assert declared_source("file:///tmp/clip.mp4#t=1,2") is None


def test_declared_source_is_url_decoded():
    url = "file:///tmp/clip.mp4#src=a%20b.mp4&t=1,2"
    assert declared_source(url) == "a b.mp4"


def test_local_path():
    assert local_path("file:///tmp/a.mp4") == "/tmp/a.mp4"
    assert local_path("https://example.com/a.mp4") is None


def test_file_identity_tracks_content(tmp_path):
    """An edited file must not answer with the entries of its old content."""
    path = tmp_path / "v.mp4"
    path.write_bytes(b"one")
    first = file_identity(str(path))
    path.write_bytes(b"two different bytes")
    assert file_identity(str(path)) != first


def test_file_identity_of_missing_file_is_stable():
    assert file_identity("/nonexistent/v.mp4") == "/nonexistent/v.mp4"
