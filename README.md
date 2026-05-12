# 6G V2X Robot Vision Offloading

> **Built at the Rutgers Makerspace** — demonstrates how 6G's sub-millisecond latency enables a low-cost autonomous robot to offload heavy computer vision to a cloud-edge server, solving the *compute problem* in self-driving cars.

---

## The Problem

Autonomous vehicles need high-frequency, high-accuracy environment sensing — ideally **100+ frames per second** for safe highway operation at speed. At 100 mph, a 50ms processing delay means the car travels **2.2 metres blind**. Current approaches:

| Approach | Problem |
|---|---|
| Onboard GPU (NVIDIA Orin) | ~$500–$2000/unit, power-hungry, thermal limits |
| 4G LTE offload | 50ms RTT → only 20fps effective — unsafe |
| 5G offload | 8ms RTT → 120fps, but unreliable in handoff zones |
| **6G offload (this project)** | **~1ms RTT → 900+ fps, enables $35 Pi-class hardware** |

---

## What This Demonstrates

```
[Rutgers Robot]  ──6G V2X──►  [MEC Edge Server]
  Raspberry Pi                   A100 GPU
  Camera ($15)                   YOLOv8n inference
  IoU Tracker (CPU)              ◄── detections in 1ms
```

1. **Real YOLOv8 detection** — actual neural-network inference, not mock data
2. **IoU multi-object tracking** — persistent object IDs across frames, zero GPU on robot
3. **6G/5G/4G network simulator** — physics-accurate latency/bandwidth/jitter model
4. **V2X protocol stack** — BSM (Basic Safety Messages), SPAT (traffic light timing)
5. **Annotated video output** — bounding boxes, tracking trails, RTT HUD, demo MP4

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt
pip install ultralytics opencv-python-headless   # for real YOLO + video

# 1. Generate a test road-scene video
python demos/generate_test_video.py

# 2. Run the full terminal dashboard demo (auto-cycles 6G→4G→5G→6G)
python demo.py

# 3. Run the video offload demo (saves annotated MP4)
python demos/video_offload_demo.py --headless

# 4. Run the 3-panel comparison (Raw | Local YOLO | 6G Edge)
python demos/tracking_comparison.py --headless

# 5. Run the latency benchmark (proof-of-concept table)
python scripts/benchmark.py
```

---

## Demo Outputs

| File | Description |
|---|---|
| `assets/test_scene.mp4` | Synthetic 640×480 road scene with moving objects |
| `assets/demo_output.mp4` | Edge-offload demo: detections + tracking trails + RTT HUD |
| `assets/comparison_output.mp4` | 3-panel: Raw \| Local YOLOv8 \| 6G Edge side-by-side |
| `assets/benchmark_results.json` | Latency benchmark raw data |

---

## Benchmark Results (Typical)

```
╔══════════════╦═════════╦══════════╦══════════╦═══════════════╗
║ Network      ║ Avg RTT ║ P95 RTT  ║ Eff. FPS ║ L4 Viable?    ║
╠══════════════╬═════════╬══════════╬══════════╬═══════════════╣
║ 4G LTE       ║ 51.2ms  ║ 71.8ms   ║ 19 fps   ║ ✗ Too slow    ║
║ 5G NR mmWave ║ 8.8ms   ║ 11.1ms   ║ 113 fps  ║ ~ Marginal    ║
║ 6G Sub-THz   ║ 1.1ms   ║ 1.4ms    ║ 909 fps  ║ ✓ Excellent   ║
╚══════════════╩═════════╩══════════╩══════════╩═══════════════╝
```

---

## Architecture

```
demo.py
  ├── EdgeServer (asyncio WebSocket server, port 8765)
  │     ├── VisionProcessor  — YOLOv8n inference (real or simulated GPU)
  │     ├── NetworkSimulator — 4G/5G/6G channel model
  │     └── V2XHub          — SPAT traffic-light broadcasts
  │
  └── RobotClient (asyncio WebSocket client)
        ├── CameraSimulator — synthetic 640×480 road-scene JPEG frames
        ├── BSM broadcasts  — 10Hz Basic Safety Messages (SAE J2735)
        └── IouTracker      — lightweight CPU multi-object tracker

demos/
  ├── generate_test_video.py     — creates assets/test_scene.mp4
  ├── video_offload_demo.py      — 6G offload pipeline on real video
  └── tracking_comparison.py    — 3-panel: Raw | Local | Edge

scripts/
  └── benchmark.py              — latency benchmark with rich table output
```

---

## Network Model

The `NetworkSimulator` applies per-packet:

- **Propagation latency** — Gaussian-jittered one-way delay
- **Transmission delay** — `(bytes × 8) / bandwidth_bps`
- **Packet loss** — Bernoulli drop

| Parameter | 4G LTE | 5G NR | **6G Sub-THz** |
|---|---|---|---|
| One-way latency | 25ms | 4ms | **0.5ms** |
| Peak bandwidth | 100 Mbps | 1 Gbps | **100 Gbps** |
| Jitter (σ) | 5ms | 0.5ms | **0.05ms** |
| Packet loss | 0.1% | 0.01% | **0.001%** |
| Expected RTT | ~50ms | ~8ms | **~1ms** |

---

## V2X Protocol Messages

| Message | Direction | Rate | Purpose |
|---|---|---|---|
| `BSM` | Robot → Edge | 10 Hz | Position, speed, heading (SAE J2735) |
| `SPAT_REQUEST` | Robot → RSU | 1 Hz | Request traffic-light timing |
| `SPAT` | RSU → Robot | on-demand | Green/Yellow/Red + time remaining |
| `VISION` | Robot → Edge | 30 Hz | JPEG frame for inference |
| `DETECTION` | Edge → Robot | 30 Hz | YOLO bounding boxes + confidence |

---

## Running Tests

```bash
pip install pytest pytest-asyncio
pytest tests/ -v
```

---

## Docker

```bash
# Run edge server + robot client as separate containers
docker compose up --build

# Or build just the edge server
docker build --target edge-server -t 6g-edge .
docker run -p 8765:8765 6g-edge
```

---

## Connection to Tesla Autopilot

Tesla's Autopilot hardware (HW4 FSD chip) runs ~72 TOPS of on-vehicle inference. As sensor resolution increases (4D radar, lidar, 8MP cameras), on-vehicle compute will hit limits. 6G MEC edge offloading:

- **Removes the compute bottleneck** from the vehicle itself
- **Enables lighter, cheaper, cooler** robot/vehicle compute platforms
- **Scales inference capacity** elastically via cloud GPU pools
- **Shared road intelligence** — edge server sees all V2X participants

This prototype shows the fundamental latency math proving it's viable.

---

## File Structure

```
6g/
├── demo.py                        # Main terminal dashboard demo
├── shared_state.py                # Thread-safe metrics state
├── config.yaml                    # Network profiles + settings
├── requirements.txt
├── requirements-gpu.txt           # Optional: ultralytics, streamlit
├── Dockerfile
├── docker-compose.yml
│
├── network/
│   ├── sim.py                     # 4G/5G/6G channel simulator
│   └── v2x.py                     # BSM, SPAT, Vision V2X messages
│
├── edge_server/
│   ├── server.py                  # WebSocket MEC server
│   └── vision.py                  # YOLOv8 / simulated inference
│
├── robot/
│   ├── client.py                  # WebSocket robot client
│   └── camera.py                  # Synthetic road-scene camera
│
├── dashboard/
│   └── terminal.py                # Rich live terminal dashboard
│
├── demos/
│   ├── generate_test_video.py     # Synthetic video generator
│   ├── video_offload_demo.py      # Full video pipeline + MP4 output
│   └── tracking_comparison.py    # 3-panel side-by-side comparison
│
├── scripts/
│   └── benchmark.py              # Latency benchmark table
│
├── tests/
│   ├── test_network_sim.py
│   ├── test_vision.py
│   ├── test_camera.py
│   └── test_v2x.py
│
└── assets/                        # Generated outputs (gitignored)
    ├── test_scene.mp4
    ├── demo_output.mp4
    ├── comparison_output.mp4
    └── benchmark_results.json
```

---

*Built by Aryan Putta — Rutgers University Makerspace*
