import asyncio
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from edge_server.vision import VisionProcessor, Detection

FAKE_JPEG = b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00' + b'\x00' * 500 + b'\xff\xd9'


@pytest.fixture
def proc():
    return VisionProcessor(backend="simulated")


@pytest.mark.asyncio
async def test_returns_detections(proc):
    dets, ms = await proc.process(FAKE_JPEG)
    assert isinstance(dets, list)
    assert len(dets) >= 1


@pytest.mark.asyncio
async def test_inference_time_realistic(proc):
    _, ms = await proc.process(FAKE_JPEG)
    assert 0.5 < ms < 30.0


@pytest.mark.asyncio
async def test_detection_fields(proc):
    dets, _ = await proc.process(FAKE_JPEG)
    for d in dets:
        assert isinstance(d, Detection)
        assert 0.0 <= d.confidence <= 1.0
        assert len(d.bbox) == 4
        assert d.class_name


@pytest.mark.asyncio
async def test_to_dict(proc):
    dets, _ = await proc.process(FAKE_JPEG)
    for d in dets:
        dd = d.to_dict()
        assert "class" in dd and "confidence" in dd and "bbox" in dd


@pytest.mark.asyncio
async def test_avg_processing_tracked(proc):
    for _ in range(3):
        await proc.process(FAKE_JPEG)
    assert proc.avg_processing_ms > 0
