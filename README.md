# Hybrid Traffic Management System
### Real-Time Emergency Vehicle Detection + Arduino Traffic Light Control

A complete Python + Arduino computer vision project that detects emergency vehicles (ambulances) in real-world traffic footage, animated/cartoon videos, or toy vehicle demo setups, and dynamically controls physical or virtual traffic lights to give them a green corridor.

---

## Project Structure

```
project/
├── main.py                 ← Single entry point — run this
├── vehicle_detection.py    ← 5-method hybrid detector
├── zone_detection.py       ← 4-zone top-view intersection division
├── decision.py             ← Traffic light state machine
├── traffic_light_ui.py     ← Virtual OpenCV signal panel
├── multi_input.py          ← 4-stream threaded video reader
├── communication.py        ← Arduino serial controller
└── traffic_lights.ino      ← Arduino firmware (upload separately)
```

---

## Installation

```bash
pip install ultralytics opencv-python numpy pyserial
```

> **pyserial is optional** — the system works without it (simulation mode).

---

## Quick Start

```bash
# Webcam, everything auto-detected
python main.py

# Your video file
python main.py --source intersection.mp4

# With Arduino connected (auto-detects USB port)
python main.py --source intersection.mp4

# Explicit port
python main.py --source intersection.mp4 --port COM3        # Windows
python main.py --source intersection.mp4 --port /dev/ttyUSB0  # Linux/Mac

# No Arduino (software only)
python main.py --source intersection.mp4 --no-arduino
```

---

## Input Modes

### Mode 2 — Single Top-View Camera (default)
One camera or video file placed above the intersection. The frame is divided into 4 zones automatically.

```bash
python main.py --source video.mp4
python main.py --source 0              # webcam index 0
```

### Mode 1 — Four Road Cameras
One video/camera per road direction. Displays as a 2×2 grid.

```bash
python main.py --north north.mp4 --south south.mp4 \
               --east  east.mp4  --west  west.mp4

# Partial (1–3 roads) — missing roads show a placeholder
python main.py --north north.mp4 --east east.mp4
```

---

## Scene Modes

The detection system adapts its behaviour based on what type of footage you're using.

| Flag | Behaviour |
|------|-----------|
| *(none)* / `--auto` | Rolling heuristic auto-detects scene each frame |
| `--toy` | Force toy/close-up mode — small blobs, bright colours |
| `--real` | Force real-traffic mode — YOLO primary, larger blobs, flash detection |

```bash
python main.py --source toy_demo.mp4  --toy
python main.py --source cctv.mp4      --real
```

---

## Detection Methods

Five methods run simultaneously and their results are merged via NMS:

| ID | Method | Detects | When active |
|----|--------|---------|-------------|
| M1 | **YOLO** | car, bus, truck, motorcycle, ambulance* | Always |
| M2 | **White + Red Cross** | Ambulances (animated, real, toy) | Always |
| M3 | **Red/Blue Flash** | Real emergency lights | Real mode |
| M4 | **Toy Vehicle Detector** | Bright small blobs, red-body toy ambulance | Toy/Auto mode |
| M5 | **Simulation fallback** | Flags first car as ambulance | `--simulate` only |

*Standard COCO YOLOv8 has no ambulance class. M2 handles this gap for all footage types.

Press **M** while running to show `[method]` tags on each detection box.

---

## All Run Options

```bash
python main.py [options]

Input:
  --source PATH       Video file or webcam index (default: 0)
  --north/south/east/west PATH   Four-road Mode 1 sources

Scene:
  --auto              Auto-detect (default)
  --toy               Force toy/demo mode
  --real              Force real-traffic mode

Detection:
  --model FILE        YOLOv8 weights (default: yolov8n.pt)
  --no-yolo           Color detection only — no GPU/model needed
  --simulate          Treat first detected car as ambulance (demo fallback)

Arduino:
  --port PORT         Serial port (auto-detect if omitted)
  --no-arduino        Disable serial entirely (pure software)

Output:
  --save              Record output to output.mp4
```

---

## Keyboard Controls

| Key | Action |
|-----|--------|
| `ESC` / `Q` | Quit |
| `SPACE` | Pause / resume |
| `S` | Save screenshot |
| `V` | Toggle vehicle bounding boxes |
| `M` | Toggle method debug tags (`[yolo]` `[cross]` `[toy]` `[flash]`) |
| `P` | Toggle embedded signal panel overlay |
| `Z` | Toggle zone-debug window (Mode 2 only) |
| `A` | Toggle Arduino serial output on/off at runtime |
| `H` | Print help to terminal |

---

## Arduino Setup

### Upload firmware
1. Open `traffic_lights.ino` in Arduino IDE
2. Select **Board → Arduino Uno**
3. Select your COM port
4. Click **Upload**

### Wiring

```
Arduino Uno Pin → 220Ω Resistor → LED Anode → LED Cathode → GND

NORTH:  Pin 2 (Red)   Pin 3 (Yellow)   Pin 4 (Green)
SOUTH:  Pin 5 (Red)   Pin 6 (Yellow)   Pin 7 (Green)
EAST:   Pin 8 (Red)   Pin 9 (Yellow)   Pin 10 (Green)
WEST:   Pin 11 (Red)  Pin 12 (Yellow)  Pin 13 (Green)
```

All 12 LED cathodes connect to the GND rail on the breadboard.

### Serial Protocol

The Arduino accepts two command formats:

**Single character** (fastest):
```
'0' → NORTH green    '2' → EAST green
'1' → SOUTH green    '3' → WEST green
'X' → all RED        'P' → ping
```

**Full string** (verbose):
```
SIGNAL:NORTH:GREEN\n
SIGNAL:SOUTH:RED\n
```

The Arduino replies with `ACK:NORTH:GREEN\n` after every command. These appear in the terminal as `[Arduino] ACK:NORTH:GREEN`.

---

## System Behaviour

### Normal mode (CYCLING)
Signals cycle GREEN through North → South → East → West every 4 seconds.

### Priority mode (EMERGENCY)
When an ambulance is detected and confirmed for 1.5 seconds:
- Its zone gets **GREEN**
- All other zones get **RED**
- Physical Arduino LEDs update immediately
- Emergency banner flashes on screen

### Clearing (YELLOW)
After the ambulance leaves, all signals go **YELLOW** for 3 seconds, then resume cycling.

### Timing configuration
Edit these values at the top of `decision.py`:
```python
CFG = {
    "trigger_hold_s":     1.5,   # how long ambulance must persist before activation
    "yellow_duration_s":  3.0,   # yellow phase duration
    "cycle_interval_s":   4.0,   # seconds per zone in cycling mode
}
```

---

## Common Issues

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError: ultralytics` | `pip install ultralytics` |
| `ModuleNotFoundError: cv2` | `pip install opencv-python` |
| Arduino not detected | Try `--port COM3` (Windows) or `--port /dev/ttyUSB0` (Linux) |
| No detections on video | Press `M` to see which methods are firing; try `--toy` or `--real` |
| YOLO not detecting ambulance | Normal — standard COCO weights have no ambulance class. M2 (cross detector) handles it |
| Low FPS | Use `--no-yolo` for pure color detection, or switch to `yolov8n.pt` (nano) |
| Video plays too fast | This is normal for video files — the system processes at native FPS |

---

## File Descriptions

| File | Purpose |
|------|---------|
| `main.py` | Entry point — CLI args, mode selection, video loops, all HUD rendering |
| `vehicle_detection.py` | `HybridDetector` class — runs M1–M5, merges with NMS, scene classification |
| `zone_detection.py` | `build_zones()`, `assign_zone()`, `draw_zones()` for top-view division |
| `decision.py` | `DecisionEngine` — CYCLING/PRIORITY/YELLOW state machine, Mode 1 + Mode 2 |
| `traffic_light_ui.py` | `render_panel()` — OpenCV compass-layout signal panel with glow effects |
| `multi_input.py` | `MultiStreamReader` — threaded 4-stream reader, `build_grid()` compositor |
| `communication.py` | `SerialController` — auto-detects Arduino, dual protocol, simulation fallback |
| `traffic_lights.ino` | Arduino firmware — dual-protocol parser, yellow transition, ACK replies |

---
## The Process
- First, I identified a common problem where ambulances get stuck in traffic and lose valuable time.
- I planned a system that could detect emergency vehicles and help them move through intersections faster.
- I used Python, OpenCV, and YOLOv8 to detect vehicles from camera footage.
- I added special detection methods to identify ambulances and emergency vehicles.
- The road was divided into different zones to know which direction the vehicle was coming from.
- I created a traffic control system that changes signals based on vehicle detection.
- When an ambulance is detected, the system gives a green light to its lane.
- I built a visual traffic signal simulator to show the signal status in real time.
- I connected the software to an Arduino to control physical traffic lights using LEDs.
- I tested the project with videos, webcam input, and model vehicles.
- After several improvements and testing, the system was able to automatically prioritize emergency vehicles and reduce waiting time at intersections.

---
## Possible Improvements
- Improve the ambulance detection system to work better in bad weather and at night.
- Add support for more types of emergency vehicles such as fire trucks and police cars.
- Use GPS data from ambulances for faster and more accurate signal control.
- Connect multiple intersections so a complete green corridor can be created.
- Develop a mobile app for traffic monitoring and system control.
- Store traffic data in a cloud database for analysis and reporting.
- Use machine learning to predict traffic congestion before it happens.
- Add live notifications for traffic authorities when an emergency vehicle is detected.
- Improve the user interface to make monitoring easier and more interactive.
- Test the system on real roads and larger traffic networks.
- Integrate the project with smart city infrastructure for city-wide traffic management.

---
## Media
**IOT SETUP**

<img width="393" height="699" alt="image" src="https://github.com/user-attachments/assets/15d6e908-1583-48de-9f06-05bf22eb4e65" />
<img width="393" height="699" alt="image" src="https://github.com/user-attachments/assets/44e96c6a-b5c2-4bde-b8ca-5b82e178f47b" />


**Normal Traffic Cycle(ie no emergency vehicles)**

https://github.com/user-attachments/assets/aed46932-d6a4-42e6-8587-f9cbecd87063

**When Ambualnce detected**

<img width="608" height="585" alt="image" src="https://github.com/user-attachments/assets/15bd48a9-c314-4dd7-9a70-b999df254e88" />

**using video files as input**









