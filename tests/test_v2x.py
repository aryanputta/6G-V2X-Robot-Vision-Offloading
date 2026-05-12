import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from network.v2x import (BSM, SPATMessage, VisionOffloadRequest,
                          VisionOffloadResponse, V2XHub, MsgType)


def test_bsm_roundtrip():
    bsm = BSM(vehicle_id="robot-001", msg_count=42, speed_mps=1.5)
    b2  = BSM.from_json(bsm.to_json())
    assert b2.vehicle_id == "robot-001"
    assert b2.msg_count  == 42
    assert abs(b2.speed_mps - 1.5) < 1e-9


def test_vision_request_roundtrip():
    req  = VisionOffloadRequest(frame_id=7, robot_id="r1", frame_jpeg_b64="AAAA")
    req2 = VisionOffloadRequest.from_json(req.to_json())
    assert req2.frame_id == 7 and req2.frame_jpeg_b64 == "AAAA"


def test_vision_response_roundtrip():
    resp  = VisionOffloadResponse(frame_id=7, server_processing_ms=4.3)
    resp2 = VisionOffloadResponse.from_json(resp.to_json())
    assert resp2.frame_id == 7
    assert abs(resp2.server_processing_ms - 4.3) < 1e-6


def test_v2x_hub_spat_valid():
    hub  = V2XHub()
    spat = hub.get_spat()
    assert spat.phase in ("GREEN", "YELLOW", "RED")
    assert spat.time_remaining_s >= 0
    assert spat.msg_id == MsgType.SPAT


def test_bsm_msg_id():
    assert BSM().msg_id == MsgType.BSM
