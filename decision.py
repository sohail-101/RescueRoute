"""
decision.py
Hybrid Traffic Management System
===================================
Unified Decision Engine — Mode 1 (multi-stream) + Mode 2 (top-view).

State machine
─────────────
  CYCLING  → PRIORITY (priority held ≥ trigger_hold_s)
  PRIORITY → YELLOW   (priority clears)
  YELLOW   → CYCLING  (after yellow_duration_s)

CYCLING  : Round-robin GREEN per zone every cycle_interval_s seconds.
PRIORITY : Priority zone GREEN; all others RED.
YELLOW   : All zones YELLOW for yellow_duration_s, then back to CYCLING.
"""

import time
from zone_detection import ZONE_NAMES, assign_zone

CFG = {
    "zones":              ZONE_NAMES,
    "trigger_hold_s":     1.5,
    "yellow_duration_s":  3.0,
    "cycle_interval_s":   4.0,
}

SIGNAL_BGR: dict[str, tuple] = {
    "GREEN":   (  0, 220,   0),
    "RED":     (  0,   0, 220),
    "YELLOW":  (  0, 200, 255),
    "UNKNOWN": (100, 100, 100),
}


class DecisionResult:
    __slots__ = (
        "signals","phase","priority_active",
        "priority_zone","active_zone",
        "detections","zone_counts","per_road_dets",
    )
    def __init__(self, signals, phase, priority_active,
                 priority_zone, active_zone,
                 detections, zone_counts, per_road_dets=None):
        self.signals         = signals
        self.phase           = phase
        self.priority_active = priority_active
        self.priority_zone   = priority_zone
        self.active_zone     = active_zone
        self.detections      = detections
        self.zone_counts     = zone_counts
        self.per_road_dets   = per_road_dets or {}

    def __repr__(self):
        return (f"DecisionResult(phase={self.phase}, "
                f"priority={self.priority_zone}, signals={self.signals})")


class DecisionEngine:
    def __init__(
        self,
        zones:             list[str] | None = None,
        trigger_hold_s:    float = CFG["trigger_hold_s"],
        yellow_duration_s: float = CFG["yellow_duration_s"],
        cycle_interval_s:  float = CFG["cycle_interval_s"],
    ) -> None:
        self._zones             = zones or list(CFG["zones"])
        self._trigger_hold_s    = trigger_hold_s
        self._yellow_duration_s = yellow_duration_s
        self._cycle_interval_s  = cycle_interval_s
        self._phase             = "CYCLING"
        self._cycle_idx         = 0
        self._cycle_last        = time.monotonic()
        self._priority_first:   float | None = None
        self._clear_since:      float | None = None
        self._priority_active   = False
        self._priority_zone:    str | None = None

    def update(self, detections: list[dict], zones: dict) -> DecisionResult:
        """Mode 2: assign zones spatially, then run state machine."""
        for det in detections:
            if det.get("zone") in (None, "UNKNOWN"):
                det["zone"] = assign_zone(det["centroid"], zones)
        return self._run(detections)

    def update_multi(self, per_road_dets: dict[str, list[dict]]) -> DecisionResult:
        """Mode 1: zones pre-assigned by road name."""
        all_dets: list[dict] = []
        for dets in per_road_dets.values():
            all_dets.extend(dets)
        all_dets.sort(key=lambda d: (d["is_priority"], d["confidence"]), reverse=True)
        result = self._run(all_dets)
        result.per_road_dets = per_road_dets
        return result

    def _run(self, detections: list[dict]) -> DecisionResult:
        now = time.monotonic()

        pri_det  = next((d for d in detections
                         if d["is_priority"] and
                         d.get("zone") not in (None,"UNKNOWN","INTERSECTION")), None)
        pri_zone = pri_det["zone"] if pri_det else None

        if pri_zone:
            if self._priority_first is None:
                self._priority_first = now
            if (now - self._priority_first >= self._trigger_hold_s
                    and self._phase != "PRIORITY"):
                self._phase           = "PRIORITY"
                self._priority_active = True
                self._priority_zone   = pri_zone
                self._clear_since     = None
        else:
            self._priority_first = None
            if self._phase == "PRIORITY":
                self._phase       = "YELLOW"
                self._clear_since = now
            if self._phase == "YELLOW":
                elapsed = (now - self._clear_since) if self._clear_since else 0
                if elapsed >= self._yellow_duration_s:
                    self._phase           = "CYCLING"
                    self._priority_active = False
                    self._priority_zone   = None
                    self._clear_since     = None
                    self._cycle_last      = now

        if self._phase == "CYCLING":
            if now - self._cycle_last >= self._cycle_interval_s:
                self._cycle_idx  = (self._cycle_idx + 1) % len(self._zones)
                self._cycle_last = now

        active_zone: str | None = None
        if self._phase == "CYCLING":
            active_zone = self._zones[self._cycle_idx]
            signals = {z: ("GREEN" if z == active_zone else "RED") for z in self._zones}
        elif self._phase == "PRIORITY":
            active_zone = self._priority_zone
            signals = {z: ("GREEN" if z == self._priority_zone else "RED") for z in self._zones}
        elif self._phase == "YELLOW":
            signals = {z: "YELLOW" for z in self._zones}
        else:
            signals = {z: "RED" for z in self._zones}

        zone_counts = {z: 0 for z in self._zones}
        zone_counts["INTERSECTION"] = 0
        for det in detections:
            z = det.get("zone","UNKNOWN")
            if z in zone_counts:
                zone_counts[z] += 1

        return DecisionResult(
            signals         = signals,
            phase           = self._phase,
            priority_active = self._priority_active,
            priority_zone   = self._priority_zone,
            active_zone     = active_zone,
            detections      = detections,
            zone_counts     = zone_counts,
        )
