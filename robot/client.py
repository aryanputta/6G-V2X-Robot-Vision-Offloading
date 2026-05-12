import asyncio
import base64
import json
import time
from collections import deque
from typing import Callable, Deque, Dict, Optional

import websockets

from network.sim import NetworkSimulator
from network.v2x import BSM, VisionOffloadRequest, MsgType
from robot.camera import CameraSimulator
from shared_state import FrameMetrics, NetworkGen


class RobotClient:
    """
    Simulates the Rutgers Makerspace autonomous robot.

    Pipeline per frame:
      1. Capture synthetic 640x480 JPEG from CameraSimulator
      2. Simulate uplink latency (6G/5G/4G)
      3. Send VisionOffloadRequest to edge server
      4. Receive DetectionResponse, compute RTT
      5. Invoke metrics callback for dashboard

    Also broadcasts BSM at 10 Hz and polls SPAT at 1 Hz.
    """

    ROBOT_ID    = "robot-001"
    TARGET_FPS  = 30
    BSM_HZ      = 10

    def __init__(
        self,
        server_uri:        str              = "ws://localhost:8765",
        network_gen:       str              = NetworkGen.SIX_G,
        metrics_callback:  Optional[Callable] = None,
    ):
        self.server_uri       = server_uri
        self.network_sim      = NetworkSimulator(network_gen)
        self.camera           = CameraSimulator()
        self.metrics_callback = metrics_callback
        self._frame_id        = 0
        self._bsm_count       = 0
        self._running         = True
        self._pending: Dict[int, dict] = {}

    # ------------------------------------------------------------------ #
    def set_network(self, generation: str):
        self.network_sim.set_network(generation)

    # ------------------------------------------------------------------ #
    async def run(self):
        while self._running:
            try:
                await self._session()
            except (OSError, websockets.exceptions.WebSocketException):
                await asyncio.sleep(1.0)

    async def _session(self):
        async with websockets.connect(self.server_uri) as ws:
            self._send_lock = asyncio.Lock()
            tasks = [
                asyncio.create_task(self._frame_loop(ws)),
                asyncio.create_task(self._bsm_loop(ws)),
                asyncio.create_task(self._spat_loop(ws)),
                asyncio.create_task(self._recv_loop(ws)),
            ]
            done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)
            for t in pending:
                t.cancel()
            for t in done:
                if not t.cancelled() and t.exception():
                    raise t.exception()

    # ------------------------------------------------------------------ #
    async def _safe_send(self, ws, payload: str):
        async with self._send_lock:
            await ws.send(payload)

    async def _frame_loop(self, ws):
        interval = 1.0 / self.TARGET_FPS
        while self._running:
            t0 = time.perf_counter()
            jpeg, _ = self.camera.capture()
            self._frame_id += 1

            uplink_ms = await self.network_sim.simulate_uplink(len(jpeg))

            req = VisionOffloadRequest(
                frame_id       = self._frame_id,
                robot_id       = self.ROBOT_ID,
                frame_jpeg_b64 = base64.b64encode(jpeg).decode(),
                frame_width    = self.camera.WIDTH,
                frame_height   = self.camera.HEIGHT,
            )
            t_send = time.perf_counter()
            self._pending[self._frame_id] = {
                "t_send":     t_send,
                "uplink_ms":  uplink_ms,
                "frame_size": len(jpeg),
            }
            await self._safe_send(ws, req.to_json())

            elapsed = time.perf_counter() - t0
            await asyncio.sleep(max(0.0, interval - elapsed))

    async def _recv_loop(self, ws):
        async for raw in ws:
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue

            msg_id = msg.get("msg_id", "")

            if msg_id == MsgType.DETECTION:
                frame_id = msg.get("frame_id", 0)
                info     = self._pending.pop(frame_id, None)
                if info:
                    rtt_ms      = (time.perf_counter() - info["t_send"]) * 1000
                    downlink_ms = rtt_ms - info["uplink_ms"] - msg.get("server_processing_ms", 0)
                    m = FrameMetrics(
                        frame_id            = frame_id,
                        rtt_ms              = round(rtt_ms, 2),
                        server_processing_ms= msg.get("server_processing_ms", 0.0),
                        uplink_ms           = info["uplink_ms"],
                        downlink_ms         = max(0.0, downlink_ms),
                        n_detections        = len(msg.get("detections", [])),
                        frame_size_bytes    = info["frame_size"],
                    )
                    if self.metrics_callback:
                        self.metrics_callback(m, msg.get("detections", []))

            elif msg_id == MsgType.SPAT:
                if self.metrics_callback:
                    self.metrics_callback("spat", msg)

    async def _bsm_loop(self, ws):
        interval = 1.0 / self.BSM_HZ
        while self._running:
            self._bsm_count += 1
            bsm = BSM(
                msg_count  = self._bsm_count,
                vehicle_id = self.ROBOT_ID,
                speed_mps  = round(0.3 + 0.5 * abs((self._bsm_count % 20) / 20 - 0.5), 2),
            )
            await self._safe_send(ws, bsm.to_json())
            await asyncio.sleep(interval)

    async def _spat_loop(self, ws):
        while self._running:
            await self._safe_send(ws, json.dumps({"msg_id": MsgType.SPAT_REQUEST}))
            await asyncio.sleep(1.0)

    def stop(self):
        self._running = False
