import asyncio
import base64
import json
import time
from typing import Callable, Optional, Set

import websockets
from websockets.server import WebSocketServerProtocol

from network.sim import NetworkSimulator
from network.v2x import VisionOffloadRequest, VisionOffloadResponse, V2XHub, MsgType
from edge_server.vision import VisionProcessor
from shared_state import NetworkGen


class EdgeServer:
    """
    6G MEC (Multi-access Edge Computing) server.

    Receives JPEG vision frames from robot clients via WebSocket,
    runs GPU inference (simulated), and returns detections.
    Also serves as a V2X infrastructure node (SPAT broadcasts).
    """

    SERVER_ID = "edge-mec-01"

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8765,
        network_gen: str = NetworkGen.SIX_G,
        state_callback: Optional[Callable] = None,
        vision_backend: str = "simulated",
    ):
        self.host = host
        self.port = port
        self.network_sim    = NetworkSimulator(network_gen)
        self.vision         = VisionProcessor(backend=vision_backend)
        self.v2x_hub        = V2XHub()
        self.state_callback = state_callback
        self._clients: Set[WebSocketServerProtocol] = set()
        self._frame_count   = 0
        self._queue_depth   = 0
        self._start_time    = time.time()

    # ------------------------------------------------------------------ #
    def set_network(self, generation: str):
        self.network_sim.set_network(generation)

    # ------------------------------------------------------------------ #
    async def handle_client(self, websocket: WebSocketServerProtocol):
        self._clients.add(websocket)
        try:
            async for raw in websocket:
                await self._dispatch(websocket, raw)
        except (websockets.exceptions.ConnectionClosed, OSError):
            pass
        finally:
            self._clients.discard(websocket)

    async def _dispatch(self, ws: WebSocketServerProtocol, raw: str):
        try:
            msg = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return

        msg_id = msg.get("msg_id", "")
        if msg_id == MsgType.VISION:
            await self._handle_vision(ws, msg)
        elif msg_id == MsgType.BSM:
            if self.state_callback:
                self.state_callback("bsm", msg)
        elif msg_id == MsgType.SPAT_REQUEST:
            spat = self.v2x_hub.get_spat()
            await ws.send(spat.to_json())

    async def _handle_vision(self, ws: WebSocketServerProtocol, msg: dict):
        frame_id  = msg.get("frame_id",       0)
        robot_id  = msg.get("robot_id",       "unknown")
        jpeg_b64  = msg.get("frame_jpeg_b64", "")

        if not jpeg_b64:
            return

        self._queue_depth += 1
        try:
            jpeg_bytes = base64.b64decode(jpeg_b64)

            # GPU inference on edge server
            detections, proc_ms = await self.vision.process(jpeg_bytes)
            self._frame_count += 1

            response = VisionOffloadResponse(
                frame_id=frame_id,
                robot_id=robot_id,
                timestamp=time.time(),
                server_processing_ms=round(proc_ms, 2),
                detections=[d.to_dict() for d in detections],
                network_gen=self.network_sim.generation,
                edge_server_id=self.SERVER_ID,
            )
            resp_json = response.to_json()

            # Simulate downlink latency
            await self.network_sim.simulate_downlink(len(resp_json.encode()))
            await ws.send(resp_json)

            if self.state_callback:
                self.state_callback("detection", {
                    "frame_id":      frame_id,
                    "detections":    response.detections,
                    "processing_ms": proc_ms,
                })
        finally:
            self._queue_depth -= 1

    # ------------------------------------------------------------------ #
    @staticmethod
    def _make_dummy_jpeg() -> bytes:
        """Create a minimal valid JPEG for model warm-up."""
        try:
            import io
            import PIL.Image
            img = PIL.Image.new("RGB", (64, 64), color=(100, 120, 140))
            buf = io.BytesIO()
            img.save(buf, format="JPEG")
            return buf.getvalue()
        except Exception:
            return b""

    async def _warmup(self):
        """Pre-warm the vision model — non-fatal if it fails."""
        try:
            dummy = self._make_dummy_jpeg()
            if dummy:
                await self.vision.process(dummy)
        except Exception:
            pass

    async def run(self):
        async with websockets.serve(self.handle_client, self.host, self.port):
            await self._warmup()
            await asyncio.Future()  # run until cancelled

    @property
    def connected_clients(self) -> int:
        return len(self._clients)

    @property
    def queue_depth(self) -> int:
        return self._queue_depth

    @property
    def total_frames_processed(self) -> int:
        return self._frame_count
