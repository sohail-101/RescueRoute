"""
multi_input.py
Hybrid Traffic Management System
===================================
Mode 1 — Multi-Stream Input Handler

Opens up to 4 simultaneous video streams, one per road direction.
Each stream runs in a background thread; read_all() returns the latest
frame from each without blocking.  Missing / dead streams return None.

Public API
──────────
    reader = MultiStreamReader({"NORTH": "n.mp4", "SOUTH": 1, ...})
    with reader:
        frames = reader.read_all()   # dict[road → frame | None]

    grid = build_grid(frames, signals)   # 2×2 display image
"""

import cv2
import threading
import time
import numpy as np

ROAD_NAMES = ["NORTH", "SOUTH", "EAST", "WEST"]
_GRID_ORDER = [("NORTH","EAST"), ("WEST","SOUTH")]
_GRID_LABEL = {
    "NORTH": ("↑ NORTH", (200, 80,   0)),
    "SOUTH": ("↓ SOUTH", (  0,160, 200)),
    "EAST":  ("→ EAST",  (  0,200,  80)),
    "WEST":  ("← WEST",  ( 80,  0, 200)),
}
_SIG_BORDER = {
    "GREEN":  (0, 220,   0),
    "RED":    (0,   0, 220),
    "YELLOW": (0, 200, 255),
}


class _StreamThread(threading.Thread):
    def __init__(self, road: str, source, daemon=True):
        super().__init__(daemon=daemon, name=f"stream-{road}")
        self.road   = road
        self.source = source
        self.frame: np.ndarray | None = None
        self.alive  = False
        self.fps    = 30.0
        self._cap   = None
        self._stop  = threading.Event()
        self._lock  = threading.Lock()

    def open(self) -> bool:
        self._cap = cv2.VideoCapture(self.source)
        if not self._cap.isOpened():
            print(f"[Stream:{self.road}] Cannot open: {self.source}")
            return False
        self.fps   = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.alive = True
        print(f"[Stream:{self.road}] Opened: {self.source}  "
              f"({self._cap.get(cv2.CAP_PROP_FRAME_WIDTH):.0f}×"
              f"{self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT):.0f}  {self.fps:.0f}fps)")
        return True

    def run(self):
        if not self.open(): return
        while not self._stop.is_set():
            ret, frame = self._cap.read()
            if not ret:
                if isinstance(self.source, str):
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = self._cap.read()
                if not ret:
                    print(f"[Stream:{self.road}] Stream ended.")
                    self.alive = False; break
            with self._lock:
                self.frame = frame
        if self._cap: self._cap.release()
        self.alive = False

    def get_frame(self) -> np.ndarray | None:
        with self._lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self):
        self._stop.set()


class MultiStreamReader:
    def __init__(self, sources: dict[str, str | int]) -> None:
        if not sources:
            raise ValueError("No sources provided.")
        self._threads: dict[str, _StreamThread] = {}
        for road, src in sources.items():
            u = road.upper()
            if u in ROAD_NAMES:
                self._threads[u] = _StreamThread(u, src)
            else:
                print(f"[MultiStream] Unknown road '{road}' — skipped.")
        self._started = False

    def start(self) -> "MultiStreamReader":
        for t in self._threads.values(): t.start()
        time.sleep(0.3)
        self._started = True
        print(f"[MultiStream] Active: {[r for r,t in self._threads.items() if t.alive]}")
        return self

    def stop(self) -> None:
        for t in self._threads.values(): t.stop()
        for t in self._threads.values(): t.join(timeout=2.0)
        print("[MultiStream] All streams stopped.")

    def read_all(self) -> dict[str, np.ndarray | None]:
        frames = {r: None for r in ROAD_NAMES}
        for road, t in self._threads.items():
            frames[road] = t.get_frame() if t.alive else None
        return frames

    @property
    def alive_roads(self) -> list[str]:
        return [r for r,t in self._threads.items() if t.alive]

    @property
    def fps(self) -> float:
        vals = [t.fps for t in self._threads.values() if t.alive]
        return sum(vals)/len(vals) if vals else 30.0

    def __enter__(self):  return self.start()
    def __exit__(self,*_): self.stop()


def build_grid(
    frames:   dict[str, np.ndarray | None],
    signals:  dict[str, str],
    cell_w:   int = 480,
    cell_h:   int = 360,
) -> np.ndarray:
    rows = []
    for row_roads in _GRID_ORDER:
        cells = []
        for road in row_roads:
            frame = frames.get(road)
            if frame is None:
                cell = np.zeros((cell_h,cell_w,3),dtype=np.uint8)
                _shadow(cell,f"{road} — NO SIGNAL",
                        (cell_w//2-80,cell_h//2),0.6,(80,80,80))
            else:
                cell = cv2.resize(frame,(cell_w,cell_h),
                                  interpolation=cv2.INTER_AREA)
            sig    = signals.get(road,"RED")
            border = _SIG_BORDER.get(sig,(80,80,80))
            bw     = 4
            cv2.rectangle(cell,(bw,bw),(cell_w-bw,cell_h-bw),border,bw*2)
            lbl,lc = _GRID_LABEL.get(road,(road,(200,200,200)))
            _shadow(cell,lbl,(10,26),0.62,lc)
            sig_col= _SIG_BORDER.get(sig,(180,180,180))
            _shadow(cell,sig,(cell_w-80,26),0.58,sig_col)
            cells.append(cell)
        rows.append(np.hstack(cells))
    return np.vstack(rows)


def _shadow(img,text,pos,scale,color,thick=1):
    cv2.putText(img,text,(pos[0]+1,pos[1]+1),
                cv2.FONT_HERSHEY_SIMPLEX,scale,(0,0,0),thick+1,cv2.LINE_AA)
    cv2.putText(img,text,pos,cv2.FONT_HERSHEY_SIMPLEX,scale,color,thick,cv2.LINE_AA)
