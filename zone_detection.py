"""
zone_detection.py
Hybrid Traffic Management System
===================================
Zone Detection — Top-View Single Camera (Mode 2)

Divides the frame into four directional zones for the intersection:

         ┌──────────────────────────────┐
         │   y < h*0.35  →  NORTH       │
         ├──────┬───────────────┬───────┤
         │      │               │       │
   WEST  │x<35% │ INTERSECTION  │ x>65% │ EAST
         │      │               │       │
         ├──────┴───────────────┴───────┤
         │   y > h*0.65  →  SOUTH       │
         └──────────────────────────────┘

Public API
──────────
    zones = build_zones(frame_w, frame_h)
    zone  = assign_zone(centroid, zones)
    frame = draw_zones(frame, zones, signals, detections)
"""

import cv2
import numpy as np

ZONE_NAMES    = ["NORTH", "SOUTH", "EAST", "WEST"]
ROAD_FRACTION = 0.35

ZONE_TINT: dict[str, tuple] = {
    "NORTH": (200,  80,   0),
    "SOUTH": (  0, 160, 200),
    "EAST":  (  0, 200,  80),
    "WEST":  ( 80,   0, 200),
}

_ARROWS    = {"NORTH": "↑N", "SOUTH": "↓S", "EAST": "→E", "WEST": "←W"}
_SIG_BGR   = {"GREEN": (0,220,0), "RED": (0,0,220), "YELLOW": (0,200,255)}


class Zone:
    def __init__(self, name: str,
                 rect: tuple[int,int,int,int],
                 label_pos: tuple[int,int]) -> None:
        self.name      = name
        self.rect      = rect
        self.label_pos = label_pos
        self.tint      = ZONE_TINT[name]

    def contains(self, cx: int, cy: int) -> bool:
        x1, y1, x2, y2 = self.rect
        return x1 <= cx <= x2 and y1 <= cy <= y2


def build_zones(frame_w: int, frame_h: int) -> dict[str, Zone]:
    rf   = ROAD_FRACTION
    n_y2 = int(frame_h * rf);       s_y1 = int(frame_h * (1.0 - rf))
    w_x2 = int(frame_w * rf);       e_x1 = int(frame_w * (1.0 - rf))
    cx   = frame_w // 2;            cy   = frame_h // 2
    return {
        "NORTH": Zone("NORTH", (0,       0,       frame_w, n_y2),   (cx-30, n_y2-10)),
        "SOUTH": Zone("SOUTH", (0,       s_y1,    frame_w, frame_h),(cx-30, s_y1+22)),
        "WEST":  Zone("WEST",  (0,       0,       w_x2,    frame_h),(6,     cy)),
        "EAST":  Zone("EAST",  (e_x1,   0,       frame_w, frame_h),(e_x1+6, cy)),
    }


def assign_zone(centroid: tuple[int,int], zones: dict[str,Zone]) -> str:
    for name in ("NORTH","SOUTH","EAST","WEST"):
        if zones[name].contains(*centroid):
            return name
    return "INTERSECTION"


def count_per_zone(detections: list[dict]) -> dict[str,int]:
    counts = {z: 0 for z in ZONE_NAMES}
    counts["INTERSECTION"] = 0
    for det in detections:
        z = det.get("zone","UNKNOWN")
        if z in counts:
            counts[z] += 1
    return counts


def draw_zones(
    frame:      np.ndarray,
    zones:      dict[str, Zone],
    signals:    dict[str, str],
    detections: list[dict],
) -> np.ndarray:
    fh, fw  = frame.shape[:2]
    counts  = count_per_zone(detections)

    for name, zone in zones.items():
        x1,y1,x2,y2 = zone.rect
        sig   = signals.get(name, "RED")
        s_col = _SIG_BGR.get(sig, (100,100,100))
        count = counts.get(name, 0)

        ov = frame.copy()
        cv2.rectangle(ov, (x1,y1), (x2,y2), zone.tint, -1)
        cv2.addWeighted(ov, 0.06, frame, 0.94, 0, frame)

        bw = 3 if sig == "GREEN" else 1
        cv2.rectangle(frame, (x1,y1), (x2,y2), s_col, bw, cv2.LINE_AA)

        lx, ly = zone.label_pos
        _shadow(frame, f"{_ARROWS[name]}  {count}v", (lx, ly), 0.58, s_col)

    n_y2 = int(fh*ROAD_FRACTION); s_y1 = int(fh*(1-ROAD_FRACTION))
    w_x2 = int(fw*ROAD_FRACTION); e_x1 = int(fw*(1-ROAD_FRACTION))
    for y in (n_y2, s_y1):
        cv2.line(frame,(0,y),(fw,y),(65,65,65),1,cv2.LINE_AA)
    for x in (w_x2, e_x1):
        cv2.line(frame,(x,0),(x,fh),(65,65,65),1,cv2.LINE_AA)

    cv2.rectangle(frame,(w_x2,n_y2),(e_x1,s_y1),(55,55,55),1)
    _shadow(frame,"INTERSECTION",(w_x2+6,(n_y2+s_y1)//2+5),0.40,(110,110,110))
    return frame


def _shadow(img, text, pos, scale, color, thick=1):
    cv2.putText(img,text,(pos[0]+1,pos[1]+1),
                cv2.FONT_HERSHEY_SIMPLEX,scale,(0,0,0),thick+1,cv2.LINE_AA)
    cv2.putText(img,text,pos,cv2.FONT_HERSHEY_SIMPLEX,scale,color,thick,cv2.LINE_AA)
