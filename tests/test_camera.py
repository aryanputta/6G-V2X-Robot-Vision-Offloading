import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from robot.camera import CameraSimulator


def test_capture_returns_bytes():
    cam = CameraSimulator()
    jpeg, objs = cam.capture()
    assert isinstance(jpeg, bytes) and len(jpeg) > 100


def test_frame_count_increments():
    cam = CameraSimulator()
    assert cam.frame_count == 0
    cam.capture(); assert cam.frame_count == 1
    cam.capture(); assert cam.frame_count == 2


def test_ground_truth_has_objects():
    cam = CameraSimulator()
    cam.capture()
    assert len(cam.ground_truth) >= 3


def test_jpeg_soi_marker():
    cam = CameraSimulator()
    jpeg, _ = cam.capture()
    assert jpeg[:2] == b'\xff\xd8'


def test_objects_move():
    cam = CameraSimulator()
    cam.capture()
    pos0 = [(o.x, o.y) for o in cam.ground_truth]
    for _ in range(15):
        cam.capture()
    pos1 = [(o.x, o.y) for o in cam.ground_truth]
    assert pos0 != pos1
