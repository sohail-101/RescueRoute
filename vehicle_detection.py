"""
vehicle_detection.py
Hybrid Traffic Management System
===================================
5-Method Hybrid Detector — works across real traffic AND toy demo setups

DETECTION METHODS (run in this order, results merged via NMS)
─────────────────────────────────────────────────────────────
 M1 · YOLO          Standard YOLOv8 inference (car/bus/truck/motorcycle).
                    Handles real-world footage well. Flags emergency class
                    names ("ambulance", "fire", "police") automatically.

 M2 · White+Red     White blob isolation → red cross check inside each blob.
      Cross          Catches animated, cartoon, and rendered ambulances where
                    YOLO gets low confidence. Works on both toy and real video.

 M3 · Red/Blue      Detects emergency flashing lights (real footage).
      Flash          High red+blue pixel concentration on a moving foreground
                    blob → priority flag. Typical of real ambulances/police cars
                    seen from the side or at intersections.

 M4 · Toy Vehicle   Small, brightly-coloured blobs (area 60–1 800 px).
      Detector       Handles toy/Lego/plastic vehicles close-up. Two sub-modes:
                     (a) toy ambulance: white body + red cross (same as M2 but
                         smaller area range)
                     (b) general toy vehicle: bright-saturated blob of any colour.

 M5 · Simulation    Optional demo fallback — flags the first detected "car" as
      Fallback       priority when CFG["simulate_priority"] = True. Useful when
                    no real ambulance footage or toy ambulance is available.

SCENE AUTO-CLASSIFICATION
─────────────────────────
classify_scene(frame, yolo_dets) returns "real" | "toy" | "unknown".

Decision heuristics (all frames, rolling vote):
  • If YOLO detects any vehicle with conf ≥ 0.50          → vote REAL
  • If median foreground blob area < 400 px               → vote TOY
  • If median HSV saturation of foreground blobs > 100    → vote TOY
  • If frame is smaller than 400×300                      → vote TOY
  • Manual override: --toy / --real CLI flags

Scene classification affects:
  • Blob area thresholds in M2 and M4
  • Minimum YOLO confidence threshold
  • Background-subtractor apply/skip decision

PUBLIC API
──────────
    detector = HybridDetector("yolov8n.pt")
    bgsub    = create_bg_subtractor()          # call once before loop
    dets     = detector.detect(frame, bgsub)  # call each frame
    frame    = draw_detections(frame, dets)

    # Force scene mode
    detector.set_scene("toy")   # or "real" / "auto"

DETECTION DICT SCHEMA
─────────────────────
    box           (x1, y1, x2, y2)
    class_id      int    — COCO id; -1 for custom emergency/toy detections
    class_name    str
    confidence    float  — 0.0–1.0
    is_priority   bool
    centroid      (cx, cy)
    zone          str    — "UNKNOWN" until decision.py assigns a zone
    method        str    — "yolo" | "cross" | "flash" | "toy" | "sim"
                          (for debug display)
"""

import cv2
import math
import numpy as np
from collections import deque

try:
    from ultralytics import YOLO
    _YOLO_AVAILABLE = True
except ImportError:
    _YOLO_AVAILABLE = False
    YOLO = None  # type: ignore

# ─────────────────────────────────────────────────────────
#  Shared constants
# ─────────────────────────────────────────────────────────
VEHICLE_CLASSES: dict[int, str] = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
PRIORITY_KEYWORDS = frozenset({"ambulance","fire","police","emergency","rescue","firetruck"})

CLASS_COLORS: dict[int, tuple] = {
    2:  (0, 210, 255),   # car        — amber
    3:  (0, 255, 130),   # motorcycle — lime
    5:  (255, 100,   0), # bus        — blue
    7:  (100,  80, 255), # truck      — purple
    -1: (0,     0, 255), # priority   — red
}
PRIORITY_COLOR  = (0,   0, 255)
PRIORITY_RING   = (0,   0, 200)
METHOD_COLORS: dict[str, tuple] = {
    "yolo":  (0, 200, 255),
    "cross": (0,   0, 255),
    "flash": (180,  0, 255),
    "toy":   (0, 255, 128),
    "sim":   (0, 165, 255),
}

# ─────────────────────────────────────────────────────────
#  CFG — all tunable thresholds in one place
# ─────────────────────────────────────────────────────────
CFG: dict = {
    # ── M1 YOLO ─────────────────────────────────────────
    "yolo_conf_real":         0.35,
    "yolo_conf_toy":          0.25,   # lower for toy since YOLO is weaker here
    "yolo_iou":               0.45,

    # ── M2 White+Red Cross (animated/real ambulance) ────
    "cross_white_v_min":      195,    # HSV V threshold for white body
    "cross_white_s_max":      55,
    "cross_red_lo1": (  0, 110, 100), "cross_red_hi1": ( 13, 255, 255),
    "cross_red_lo2": (158, 110, 100), "cross_red_hi2": (179, 255, 255),
    "cross_red_thresh":       0.04,   # min red fraction in blob
    "cross_min_area_real":    500,    # pixel area bounds for REAL footage
    "cross_max_area_real":    6000,
    "cross_min_area_toy":     60,     # much smaller for toy close-up
    "cross_max_area_toy":     1800,
    "cross_ar_min":           0.30,
    "cross_ar_max":           4.00,
    "cross_max_frame_frac":   0.12,

    # ── M3 Red/Blue Flash (real emergency lights) ───────
    "flash_blue_lo": (100,  80,  80), "flash_blue_hi": (130, 255, 255),
    "flash_red_lo1": (  0, 100,  80), "flash_red_hi1": ( 12, 255, 255),
    "flash_red_lo2": (158, 100,  80), "flash_red_hi2": (179, 255, 255),
    "flash_rb_thresh":        0.10,   # combined red+blue fraction
    "flash_min_area":         400,
    "flash_max_area":         8000,

    # ── M4 Toy Vehicle Detector ──────────────────────────
    "toy_sat_min":            100,    # minimum HSV saturation for a toy car
    "toy_val_min":            80,     # minimum HSV value
    "toy_min_area":           60,
    "toy_max_area":           1800,
    "toy_ar_min":             0.25,
    "toy_ar_max":             5.0,
    "toy_max_frame_frac":     0.05,
    # toy ambulance specific (white+red cross at toy scale)
    "toy_ambu_white_v_min":   190,
    "toy_ambu_white_s_max":   60,
    "toy_ambu_red_thresh":    0.04,

    # ── Background subtractor ───────────────────────────
    "bg_history":             60,
    "bg_var_threshold":       50,
    "fg_dilate_px":           18,
    "fg_dilate_px_toy":       8,      # smaller dilation for toy close-up

    # ── NMS ─────────────────────────────────────────────
    "nms_iou":                0.40,

    # ── Scene auto-classification ────────────────────────
    "scene_vote_window":      30,     # rolling window of scene votes
    "scene_real_min_conf":    0.50,   # YOLO conf needed to cast a REAL vote
    "scene_toy_area_thresh":  400,    # median blob area below this → TOY vote
    "scene_toy_sat_thresh":   100,    # median blob saturation above this → TOY vote

    # ── M5 Simulation ────────────────────────────────────
    "simulate_priority":      False,  # flag via CLI --simulate
}


# ─────────────────────────────────────────────────────────
#  Utilities
# ─────────────────────────────────────────────────────────
def create_bg_subtractor() -> cv2.BackgroundSubtractor:
    """MOG2 background subtractor. Call once; pass to detector.detect() each frame."""
    return cv2.createBackgroundSubtractorMOG2(
        history=CFG["bg_history"],
        varThreshold=CFG["bg_var_threshold"],
        detectShadows=False,
    )


def load_model(model_path: str = "yolov8n.pt"):
    """Load YOLOv8 weights (auto-downloads on first call). Returns None if unavailable."""
    if not _YOLO_AVAILABLE:
        print("[Detection] ultralytics not installed — YOLO disabled.")
        return None
    print(f"[Detection] Loading model: {model_path} ...")
    m = YOLO(model_path)
    print("[Detection] Model ready.")
    return m


def _nms(dets: list[dict], iou_thresh: float = CFG["nms_iou"]) -> list[dict]:
    """Standard box NMS sorted by confidence descending."""
    if len(dets) <= 1:
        return dets
    boxes  = np.array([d["box"] for d in dets], dtype=float)
    scores = np.array([d["confidence"] for d in dets], dtype=float)
    x1, y1, x2, y2 = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]
    areas  = (x2-x1) * (y2-y1)
    order  = scores.argsort()[::-1]
    keep   = []
    while len(order):
        i = order[0]; keep.append(i)
        if len(order) == 1: break
        ix1 = np.maximum(x1[i], x1[order[1:]])
        iy1 = np.maximum(y1[i], y1[order[1:]])
        ix2 = np.minimum(x2[i], x2[order[1:]])
        iy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, ix2-ix1) * np.maximum(0, iy2-iy1)
        iou   = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        order = order[1:][iou <= iou_thresh]
    return [dets[k] for k in keep]


def _det(box, cls_id, cls_name, conf, is_priority, frame, method) -> dict:
    """Construct a detection dict with centroid auto-computed from box."""
    x1, y1, x2, y2 = box
    return {
        "box":         (x1, y1, x2, y2),
        "class_id":    cls_id,
        "class_name":  cls_name,
        "confidence":  round(float(conf), 3),
        "is_priority": is_priority,
        "centroid":    ((x1+x2)//2, (y1+y2)//2),
        "zone":        "UNKNOWN",
        "method":      method,
    }


def _cross_score(roi_hsv: np.ndarray, red_mask: np.ndarray) -> float:
    """Score the red-cross pattern strength (0–1) in a ROI."""
    rh, rw = roi_hsv.shape[:2]
    mid_y  = rh // 2
    mid_x  = rw // 2
    band   = max(2, min(rh, rw) // 8)
    h_s = red_mask[max(0, mid_y-band): mid_y+band+1, :]
    v_s = red_mask[:, max(0, mid_x-band): mid_x+band+1]
    h_f = cv2.countNonZero(h_s) / (h_s.size + 1)
    v_f = cv2.countNonZero(v_s) / (v_s.size + 1)
    return (h_f + v_f) / 2


# ─────────────────────────────────────────────────────────
#  HybridDetector
# ─────────────────────────────────────────────────────────
class HybridDetector:
    """
    Runs all 5 detection methods and merges results via NMS.

    Parameters
    ----------
    model_path : YOLOv8 weights filename.  Pass None to skip YOLO.
    scene      : "auto" | "real" | "toy"
                 "auto" uses rolling heuristics to decide each frame.
    """

    def __init__(
        self,
        model_path: str | None = "yolov8n.pt",
        scene:      str        = "auto",
    ) -> None:
        self._model       = load_model(model_path) if model_path else None
        self._scene_mode  = scene          # "auto" | "real" | "toy"
        self._scene_cur   = "unknown"      # last classified scene
        self._votes: deque = deque(maxlen=CFG["scene_vote_window"])

    # ── Public ────────────────────────────────────────────
    def set_scene(self, mode: str) -> None:
        """Override scene: "auto" | "real" | "toy"."""
        assert mode in ("auto", "real", "toy"), f"Unknown scene: {mode}"
        self._scene_mode = mode
        print(f"[Detector] Scene forced to: {mode}")

    @property
    def scene(self) -> str:
        """Current effective scene ("real" | "toy" | "unknown")."""
        if self._scene_mode != "auto":
            return self._scene_mode
        return self._scene_cur

    def detect(
        self,
        frame:       np.ndarray,
        bgsub:       cv2.BackgroundSubtractor | None = None,
        preset_zone: str | None = None,
    ) -> list[dict]:
        """
        Run all enabled methods on *frame* and return merged detections.

        Parameters
        ----------
        frame        : BGR video frame
        bgsub        : background subtractor (from create_bg_subtractor()).
                       Pass None to skip foreground masking.
        preset_zone  : when set, all detections are tagged with this zone
                       (Mode 1 multi-stream usage).
        """
        if frame is None or frame.size == 0:
            return []

        fh, fw = frame.shape[:2]

        # ── Foreground mask ───────────────────────────────
        fg_mask: np.ndarray | None = None
        if bgsub is not None:
            try:
                fg_mask = bgsub.apply(frame)
            except Exception:
                pass

        # ── M1: YOLO ──────────────────────────────────────
        yolo_dets = self._m1_yolo(frame)

        # ── Scene classification (uses M1 results) ────────
        if self._scene_mode == "auto":
            self._update_scene(frame, yolo_dets, fg_mask)
        scene = self.scene

        # ── M2: White+Red Cross ───────────────────────────
        cross_dets = self._m2_cross(frame, fg_mask, scene)

        # ── M3: Red/Blue Flash (real only) ───────────────
        flash_dets: list[dict] = []
        if scene != "toy":
            flash_dets = self._m3_flash(frame, fg_mask)

        # ── M4: Toy Vehicle Detector ──────────────────────
        toy_dets: list[dict] = []
        if scene != "real":
            toy_dets = self._m4_toy(frame)

        # ── M5: Simulation fallback ───────────────────────
        sim_dets: list[dict] = []
        if CFG["simulate_priority"]:
            sim_dets = self._m5_simulate(yolo_dets)

        # ── Merge all methods ─────────────────────────────
        all_dets = self._merge(yolo_dets, cross_dets, flash_dets,
                               toy_dets, sim_dets)

        # ── Sort: priority first, then confidence ─────────
        all_dets.sort(key=lambda d: (d["is_priority"], d["confidence"]),
                      reverse=True)
        final = _nms(all_dets)

        if preset_zone:
            for d in final:
                d["zone"] = preset_zone

        return final

    # ── M1: YOLO ──────────────────────────────────────────
    def _m1_yolo(self, frame: np.ndarray) -> list[dict]:
        if self._model is None:
            return []
        scene = self._scene_mode if self._scene_mode != "auto" else self._scene_cur
        conf_thresh = (CFG["yolo_conf_toy"] if scene == "toy"
                       else CFG["yolo_conf_real"])
        try:
            results = self._model(frame, conf=conf_thresh,
                                  iou=CFG["yolo_iou"], verbose=False)
        except Exception as exc:
            print(f"[YOLO] Error: {exc}")
            return []

        fh, fw = frame.shape[:2]
        dets: list[dict] = []
        for result in results:
            if result.boxes is None: continue
            for box in result.boxes:
                cls_id   = int(box.cls[0])
                cls_name = self._model.names.get(cls_id, "").lower()
                is_veh   = cls_id in VEHICLE_CLASSES
                is_pri   = any(kw in cls_name for kw in PRIORITY_KEYWORDS)
                if not (is_veh or is_pri): continue
                x1,y1,x2,y2 = map(int, box.xyxy[0])
                x1,y1 = max(x1,0),max(y1,0)
                x2,y2 = min(x2,fw),min(y2,fh)
                if x2<=x1 or y2<=y1: continue
                dets.append(_det(
                    (x1,y1,x2,y2),
                    cls_id if is_veh else -1,
                    cls_name if is_pri else VEHICLE_CLASSES.get(cls_id,"vehicle"),
                    float(box.conf[0]),
                    is_pri,
                    frame,
                    "yolo",
                ))
        return dets

    # ── M2: White Body + Red Cross ────────────────────────
    def _m2_cross(
        self,
        frame:   np.ndarray,
        fg_mask: np.ndarray | None,
        scene:   str,
    ) -> list[dict]:
        """
        Detect ambulances as white blobs with a red cross marking.
        Works for animated, real-world, and toy ambulances.
        Area thresholds adapt to the detected scene.
        """
        fh, fw      = frame.shape[:2]
        frame_area  = fw * fh
        hsv         = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # White body mask
        vmin = (CFG["toy_ambu_white_v_min"] if scene == "toy"
                else CFG["cross_white_v_min"])
        smax = (CFG["toy_ambu_white_s_max"] if scene == "toy"
                else CFG["cross_white_s_max"])
        white = cv2.inRange(hsv, (0, 0, vmin), (180, smax, 255))
        white = cv2.morphologyEx(
            white, cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        )

        if fg_mask is not None and scene != "toy":
            dil_sz  = CFG["fg_dilate_px_toy"] if scene == "toy" else CFG["fg_dilate_px"]
            fg_grown = cv2.dilate(
                fg_mask, cv2.getStructuringElement(
                    cv2.MORPH_RECT, (dil_sz, dil_sz)))
            white = cv2.bitwise_and(white, fg_grown)

        cnts, _ = cv2.findContours(white, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        dets: list[dict] = []

        min_a = CFG["cross_min_area_toy"] if scene == "toy" else CFG["cross_min_area_real"]
        max_a = CFG["cross_max_area_toy"] if scene == "toy" else CFG["cross_max_area_real"]

        for c in cnts:
            area = cv2.contourArea(c)
            if not (min_a <= area <= max_a): continue
            x,y,w,h = cv2.boundingRect(c)
            if w<=0 or h<=0: continue
            ar = w/h
            if not (CFG["cross_ar_min"] <= ar <= CFG["cross_ar_max"]): continue
            if (w*h) > CFG["cross_max_frame_frac"] * frame_area: continue

            x1=max(x,0); y1=max(y,0)
            x2=min(x+w,fw); y2=min(y+h,fh)
            roi = frame[y1:y2, x1:x2]
            if roi.size == 0: continue

            roi_hsv  = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            r1 = cv2.inRange(roi_hsv, np.array(CFG["cross_red_lo1"]),
                             np.array(CFG["cross_red_hi1"]))
            r2 = cv2.inRange(roi_hsv, np.array(CFG["cross_red_lo2"]),
                             np.array(CFG["cross_red_hi2"]))
            red_mask  = cv2.bitwise_or(r1, r2)
            red_ratio = cv2.countNonZero(red_mask) / (roi.shape[0]*roi.shape[1]+1)

            thresh = (CFG["toy_ambu_red_thresh"] if scene == "toy"
                      else CFG["cross_red_thresh"])
            if red_ratio < thresh: continue

            cross = _cross_score(roi_hsv, red_mask)
            conf  = float(np.clip(red_ratio*4.0 + cross*2.0, 0.10, 0.99))
            dets.append(_det((x1,y1,x2,y2), -1, "ambulance", conf, True, frame, "cross"))

        return _nms(dets)

    # ── M3: Red/Blue Flash (real emergency vehicles) ──────
    def _m3_flash(
        self,
        frame:   np.ndarray,
        fg_mask: np.ndarray | None,
    ) -> list[dict]:
        """
        Detect emergency vehicles by the presence of simultaneous red AND
        blue light regions (flashing lights on real police/ambulance).
        Skipped in toy mode — toy vehicles don't have working lights.
        """
        fh, fw = frame.shape[:2]
        hsv    = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        r1 = cv2.inRange(hsv, np.array(CFG["flash_red_lo1"]),
                         np.array(CFG["flash_red_hi1"]))
        r2 = cv2.inRange(hsv, np.array(CFG["flash_red_lo2"]),
                         np.array(CFG["flash_red_hi2"]))
        red  = cv2.bitwise_or(r1, r2)
        blue = cv2.inRange(hsv, np.array(CFG["flash_blue_lo"]),
                           np.array(CFG["flash_blue_hi"]))

        # Combine R+B before looking for blobs
        rb = cv2.bitwise_or(red, blue)
        rb = cv2.morphologyEx(rb, cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_RECT,(5,5)))

        if fg_mask is not None:
            fg_grown = cv2.dilate(
                fg_mask,
                cv2.getStructuringElement(cv2.MORPH_RECT,
                                          (CFG["fg_dilate_px"], CFG["fg_dilate_px"])))
            rb = cv2.bitwise_and(rb, fg_grown)

        cnts,_ = cv2.findContours(rb, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        dets: list[dict] = []

        for c in cnts:
            area = cv2.contourArea(c)
            if not (CFG["flash_min_area"] <= area <= CFG["flash_max_area"]): continue
            x,y,w,h = cv2.boundingRect(c)
            if w<=0 or h<=0: continue
            x1=max(x,0); y1=max(y,0)
            x2=min(x+w,fw); y2=min(y+h,fh)
            roi = frame[y1:y2, x1:x2]
            if roi.size==0: continue
            roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            tot     = roi.shape[0]*roi.shape[1]
            r_px = cv2.countNonZero(cv2.bitwise_or(
                cv2.inRange(roi_hsv, np.array(CFG["flash_red_lo1"]),
                            np.array(CFG["flash_red_hi1"])),
                cv2.inRange(roi_hsv, np.array(CFG["flash_red_lo2"]),
                            np.array(CFG["flash_red_hi2"])),
            ))
            b_px = cv2.countNonZero(
                cv2.inRange(roi_hsv, np.array(CFG["flash_blue_lo"]),
                            np.array(CFG["flash_blue_hi"])))
            rb_ratio = (r_px + b_px) / (tot + 1)
            if rb_ratio < CFG["flash_rb_thresh"]: continue
            conf = float(np.clip(rb_ratio * 5.0, 0.10, 0.99))
            dets.append(_det((x1,y1,x2,y2), -1, "emergency", conf, True, frame, "flash"))

        return _nms(dets)

    # ── M4: Toy Vehicle Detector ──────────────────────────
    def _m4_toy(self, frame: np.ndarray) -> list[dict]:
        """
        Detect toy vehicles by finding small, brightly-saturated blobs.

        Sub-pass A — toy ambulance: white body + red cross (same signature
                     as M2 but with toy-scale area limits already handled
                     there; this sub-pass adds red-body variants).
        Sub-pass B — general toy cars: any vivid-coloured blob of correct size.

        Only runs in "toy" or "auto" (non-confirmed-real) scene modes.
        """
        fh, fw     = frame.shape[:2]
        frame_area = fw * fh
        hsv        = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dets: list[dict] = []

        # ── Sub-pass A: red-body toy ambulance (red+orange with white cross) ──
        # Some toy ambulances have a red body with a white cross on top.
        red_body1  = cv2.inRange(hsv, (0,  120, 100), (15, 255, 255))
        red_body2  = cv2.inRange(hsv, (155,120, 100), (179,255, 255))
        red_body   = cv2.bitwise_or(red_body1, red_body2)
        red_body   = cv2.morphologyEx(red_body, cv2.MORPH_CLOSE,
                                      cv2.getStructuringElement(cv2.MORPH_RECT,(3,3)))
        cnts,_ = cv2.findContours(red_body, cv2.RETR_EXTERNAL,
                                  cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            area = cv2.contourArea(c)
            if not (CFG["toy_min_area"] <= area <= CFG["toy_max_area"]): continue
            x,y,w,h = cv2.boundingRect(c)
            if w<=0 or h<=0: continue
            ar = w/h
            if not (CFG["toy_ar_min"] <= ar <= CFG["toy_ar_max"]): continue
            if (w*h) > CFG["toy_max_frame_frac"] * frame_area: continue
            x1=max(x,0); y1=max(y,0)
            x2=min(x+w,fw); y2=min(y+h,fh)
            roi = frame[y1:y2,x1:x2]
            if roi.size==0: continue
            # Check for white cross on the red body
            roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            white_px = cv2.countNonZero(
                cv2.inRange(roi_hsv, (0,0,200),(180,50,255)))
            white_r  = white_px / (roi.shape[0]*roi.shape[1]+1)
            if white_r < 0.05: continue    # need at least 5% white (the cross)
            conf = float(np.clip(white_r * 6.0, 0.15, 0.90))
            dets.append(_det((x1,y1,x2,y2),-1,"toy-ambulance",conf,True,frame,"toy"))

        # ── Sub-pass B: general toy vehicles (brightly coloured blobs) ───────
        # Mask: high saturation AND reasonable value
        bright = cv2.inRange(
            hsv,
            (0,   CFG["toy_sat_min"], CFG["toy_val_min"]),
            (179, 255, 255),
        )
        # Remove red-body blobs already handled in sub-pass A
        bright = cv2.bitwise_and(bright, cv2.bitwise_not(red_body))
        bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN,
                                  cv2.getStructuringElement(cv2.MORPH_RECT,(2,2)))
        bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE,
                                  cv2.getStructuringElement(cv2.MORPH_RECT,(5,5)))

        cnts,_ = cv2.findContours(bright, cv2.RETR_EXTERNAL,
                                  cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            area = cv2.contourArea(c)
            if not (CFG["toy_min_area"] <= area <= CFG["toy_max_area"]): continue
            x,y,w,h = cv2.boundingRect(c)
            if w<=0 or h<=0: continue
            ar = w/h
            if not (CFG["toy_ar_min"] <= ar <= CFG["toy_ar_max"]): continue
            if (w*h) > CFG["toy_max_frame_frac"] * frame_area: continue
            x1=max(x,0); y1=max(y,0)
            x2=min(x+w,fw); y2=min(y+h,fh)
            roi = frame[y1:y2,x1:x2]
            if roi.size==0: continue
            roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            mean_sat = float(roi_hsv[:,:,1].mean())
            conf = float(np.clip(mean_sat / 255.0 * 0.85, 0.15, 0.85))
            # Determine dominant hue → class name
            mean_hue = float(roi_hsv[:,:,0].mean())
            if mean_hue < 15 or mean_hue > 165:
                cls_name = "toy-car-red"
            elif 15 <= mean_hue < 35:
                cls_name = "toy-car-yellow"
            elif 35 <= mean_hue < 85:
                cls_name = "toy-car-green"
            elif 85 <= mean_hue < 130:
                cls_name = "toy-car-blue"
            else:
                cls_name = "toy-car"
            dets.append(_det((x1,y1,x2,y2),-1,cls_name,conf,False,frame,"toy"))

        return _nms(dets)

    # ── M5: Simulation fallback ───────────────────────────
    def _m5_simulate(self, yolo_dets: list[dict]) -> list[dict]:
        """Flag the first detected car as priority (demo mode)."""
        for d in yolo_dets:
            if d["class_id"] == 2 and not d["is_priority"]:
                sim = dict(d)
                sim["is_priority"] = True
                sim["class_name"]  = "ambulance(sim)"
                sim["method"]      = "sim"
                return [sim]
        return []

    # ── Scene classification ──────────────────────────────
    def _update_scene(
        self,
        frame:    np.ndarray,
        yolo_dets: list[dict],
        fg_mask:   np.ndarray | None,
    ) -> None:
        """
        Cast one vote per frame and update the rolling scene classification.
        REAL votes: YOLO detects confident vehicles.
        TOY votes : small blobs OR high average saturation.
        """
        fh, fw = frame.shape[:2]

        # Vote from YOLO
        if any(d["confidence"] >= CFG["scene_real_min_conf"] for d in yolo_dets):
            self._votes.append("real")
            self._scene_cur = self._majority_vote()
            return

        # Vote from blob analysis
        source = fg_mask if fg_mask is not None else None
        gray   = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hsv    = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Get moving blobs
        if source is not None:
            thresh = source
        else:
            _, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)

        cnts,_ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                  cv2.CHAIN_APPROX_SIMPLE)
        areas = []
        sats  = []
        for c in cnts:
            a = cv2.contourArea(c)
            if a < 50: continue
            areas.append(a)
            x,y,w,h = cv2.boundingRect(c)
            roi_hsv = hsv[max(y,0):min(y+h,fh), max(x,0):min(x+w,fw)]
            if roi_hsv.size > 0:
                sats.append(float(roi_hsv[:,:,1].mean()))

        if areas:
            med_area = float(np.median(areas))
            med_sat  = float(np.median(sats)) if sats else 0.0
            is_toy   = (med_area < CFG["scene_toy_area_thresh"] or
                        med_sat  > CFG["scene_toy_sat_thresh"])
            self._votes.append("toy" if is_toy else "real")

        # Frame size heuristic
        if fw < 400 or fh < 300:
            self._votes.append("toy")

        self._scene_cur = self._majority_vote()

    def _majority_vote(self) -> str:
        if not self._votes:
            return "unknown"
        real_c = self._votes.count("real")
        toy_c  = self._votes.count("toy")
        if real_c == toy_c:
            return self._scene_cur or "unknown"
        return "real" if real_c > toy_c else "toy"

    # ── Merge helper ──────────────────────────────────────
    def _merge(self, *method_results) -> list[dict]:
        """
        Merge N method result lists, giving priority detections precedence.
        A YOLO box is suppressed if a color-method already covers it (IoU > 0.30).
        """
        # Flatten non-YOLO results first
        color_dets: list[dict] = []
        for lst in method_results[1:]:   # skip yolo_dets (index 0)
            color_dets.extend(lst)

        merged = list(color_dets)

        # Add YOLO dets that don't overlap with color dets
        for yd in method_results[0]:
            yx1,yy1,yx2,yy2 = yd["box"]
            dominated = False
            for cd in color_dets:
                cx1,cy1,cx2,cy2 = cd["box"]
                ix1=max(yx1,cx1); iy1=max(yy1,cy1)
                ix2=min(yx2,cx2); iy2=min(yy2,cy2)
                if ix2>ix1 and iy2>iy1:
                    inter   = (ix2-ix1)*(iy2-iy1)
                    area_y  = max((yx2-yx1)*(yy2-yy1),1)
                    if inter/area_y > 0.30:
                        dominated = True; break
            if not dominated:
                merged.append(yd)

        return merged


# ─────────────────────────────────────────────────────────
#  draw_detections
# ─────────────────────────────────────────────────────────
def draw_detections(
    frame:      np.ndarray,
    detections: list[dict],
    show_zone:  bool = True,
    show_method:bool = False,
) -> np.ndarray:
    """
    Annotate frame with bounding boxes, labels, and priority indicators.

    Parameters
    ----------
    show_method : if True, prints the detection method tag (yolo/cross/flash/toy)
                  in small text below the label. Useful for debugging.
    """
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        pri    = det["is_priority"]
        cid    = det["class_id"]
        zone   = det.get("zone", "?")
        method = det.get("method", "")
        color  = PRIORITY_COLOR if pri else CLASS_COLORS.get(cid, (180, 180, 180))
        thick  = 3 if pri else 2

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thick, cv2.LINE_AA)

        prefix   = "AMBULANCE: " if pri else ""
        zone_tag = f" [{zone}]" if show_zone else ""
        label    = f"{prefix}{det['class_name']} {det['confidence']:.0%}{zone_tag}"

        (tw, th), bl = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
        ly1 = max(y1 - th - bl - 6, 0)
        ly2 = ly1 + th + bl + 6

        ov = frame.copy()
        cv2.rectangle(ov, (x1, ly1), (x1+tw+8, ly2), color, -1)
        cv2.addWeighted(ov, 0.60, frame, 0.40, 0, frame)
        cv2.putText(frame, label, (x1+4, ly2-bl-2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                    (255, 255, 255), 1, cv2.LINE_AA)

        if show_method and method:
            mc = METHOD_COLORS.get(method, (180,180,180))
            cv2.putText(frame, f"[{method}]", (x1+4, y2+13),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, mc, 1, cv2.LINE_AA)

        if pri:
            cx = (x1+x2)//2; cy = (y1+y2)//2
            r  = max(x2-x1, y2-y1)//2 + 12
            cv2.circle(frame, (cx, cy), r,     PRIORITY_COLOR, 2, cv2.LINE_AA)
            cv2.circle(frame, (cx, cy), r + 5, PRIORITY_RING,  1, cv2.LINE_AA)

    return frame
