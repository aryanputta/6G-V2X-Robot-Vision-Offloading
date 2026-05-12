FROM python:3.11-slim AS base

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Optional GPU deps (comment out for CPU-only)
# RUN pip install --no-cache-dir ultralytics opencv-python-headless

COPY . .
RUN mkdir -p assets

# ── Edge Server ────────────────────────────────────────────────────────────
FROM base AS edge-server
EXPOSE 8765
CMD ["python", "-c", "\
import asyncio, sys; sys.path.insert(0, '.');\
from edge_server.server import EdgeServer;\
from shared_state import NetworkGen;\
server = EdgeServer(host='0.0.0.0', port=8765, network_gen=NetworkGen.SIX_G);\
asyncio.run(server.run())"]

# ── Robot Client ───────────────────────────────────────────────────────────
FROM base AS robot
ENV SERVER_URI=ws://edge-server:8765
CMD ["python", "-c", "\
import asyncio, os, sys; sys.path.insert(0, '.');\
from robot.client import RobotClient;\
from shared_state import NetworkGen;\
client = RobotClient(\
    server_uri=os.environ.get('SERVER_URI', 'ws://localhost:8765'),\
    network_gen=NetworkGen.SIX_G);\
asyncio.run(client.run())"]

# ── Full Demo ─────────────────────────────────────────────────────────────
FROM base AS demo
CMD ["python", "demo.py", "--headless", "--benchmark"]
