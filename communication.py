"""
communication.py
Hybrid Traffic Management System — Serial Communication
========================================================
Sends traffic signal commands to the Arduino firmware (traffic_lights.ino)
and falls back to simulation mode automatically when no hardware is present.

TWO COMMAND PROTOCOLS
─────────────────────
Protocol A — Single character (simple, lowest latency):
    '0' → NORTH GREEN      '2' → EAST GREEN
    '1' → SOUTH GREEN      '3' → WEST GREEN
    'X' → all RED           'P' → ping

Protocol B — Full string (matches decision.py signal dict format):
    "SIGNAL:NORTH:GREEN\\n"
    "SIGNAL:SOUTH:RED\\n"
    "SIGNAL:EAST:YELLOW\\n"
    "SIGNAL:WEST:RED\\n"

Both protocols are supported by the Arduino firmware simultaneously.
Use send_active_lane() for Protocol A (1 byte per command — lowest traffic).
Use send_signals()     for Protocol B (sends only changed lanes).

ARDUINO ACKNOWLEDGEMENTS
────────────────────────
    "ACK:NORTH:GREEN\\n"  — lane switch confirmed
    "ACK:ALL:RED\\n"      — all-red confirmed
    "OK\\n"               — ping reply
    "READY\\n"            — Arduino boot complete
    "ERR:<reason>\\n"     — parse or range error

PUBLIC API
──────────
    ctrl = SerialController()           # auto-detect Arduino
    ctrl = SerialController("COM3")     # explicit port
    ctrl = SerialController(simulate=True)  # headless / no hardware

    # Protocol A — send the currently active lane (most efficient)
    ctrl.send_active_lane(active_zone)  # active_zone: "NORTH"/"SOUTH"/"EAST"/"WEST"

    # Protocol B — send full signal dict (sends only changes)
    ctrl.send_signals({"NORTH":"GREEN","SOUTH":"RED","EAST":"RED","WEST":"RED"})

    ack = ctrl.read_response()          # non-blocking ACK check

    ctrl.close()

ARDUINO PIN MAPPING (for reference)
────────────────────────────────────
    NORTH  Red→2   Yellow→3   Green→4
    SOUTH  Red→5   Yellow→6   Green→7
    EAST   Red→8   Yellow→9   Green→10
    WEST   Red→11  Yellow→12  Green→13
"""

import time
import sys
from typing import Optional

try:
    import serial
    import serial.tools.list_ports
    _SERIAL_AVAILABLE = True
except ImportError:
    _SERIAL_AVAILABLE = False


# ──────────────────────────────────────────────────────────────────
#  Configuration
# ──────────────────────────────────────────────────────────────────
CFG = {
    "baud_rate":        9600,
    "timeout_s":        1.0,
    "connect_wait_s":   2.0,    # wait for Arduino to reset after DTR toggle
    "send_interval_s":  0.25,   # minimum seconds between transmissions
    "auto_detect":      True,
    # USB descriptor substrings to recognise an Arduino port
    "keywords": ["Arduino", "USB", "CH340", "CP210", "ttyUSB", "ttyACM"],
    # Protocol A — map zone name → single-char command byte
    "zone_to_char": {
        "NORTH": b"0",
        "SOUTH": b"1",
        "EAST":  b"2",
        "WEST":  b"3",
    },
}

# Lanes recognised by both protocols
KNOWN_LANES  = {"NORTH", "SOUTH", "EAST", "WEST"}
KNOWN_STATES = {"RED", "YELLOW", "GREEN"}


# ──────────────────────────────────────────────────────────────────
#  Port auto-discovery
# ──────────────────────────────────────────────────────────────────
def find_arduino_port() -> Optional[str]:
    """
    Scan all available serial ports and return the device path of the
    first one whose description matches a known Arduino/USB-serial keyword.
    Returns None when pyserial is not installed or no match is found.
    """
    if not _SERIAL_AVAILABLE:
        return None
    for port in serial.tools.list_ports.comports():
        desc = (port.description or "").upper()
        name = (port.name       or "")
        if any(kw.upper() in desc or kw.upper() in name.upper()
               for kw in CFG["keywords"]):
            return port.device
    return None


def list_serial_ports() -> list[str]:
    """Return all available serial port device names (for diagnostics)."""
    if not _SERIAL_AVAILABLE:
        return []
    return [p.device for p in serial.tools.list_ports.comports()]


# ──────────────────────────────────────────────────────────────────
#  SerialController
# ──────────────────────────────────────────────────────────────────
class SerialController:
    """
    Manages serial communication with the Arduino traffic-light firmware.

    Gracefully degrades to simulation mode (stdout only) when:
      • pyserial is not installed
      • no Arduino is detected
      • the port fails to open
      • a write error occurs at runtime

    Parameters
    ----------
    port      : explicit serial port (e.g. "COM3" or "/dev/ttyUSB0").
                Pass None to auto-detect.
    baud_rate : must match Arduino firmware (default 9600).
    simulate  : force simulation mode regardless of hardware.
    verbose   : if True, print all transmitted and received strings.
    """

    def __init__(
        self,
        port:      Optional[str] = None,
        baud_rate: int           = CFG["baud_rate"],
        simulate:  bool          = False,
        verbose:   bool          = False,
    ) -> None:
        self._conn:       Optional["serial.Serial"] = None
        self._simulate:   bool       = simulate or not _SERIAL_AVAILABLE
        self._verbose:    bool       = verbose
        self._last_signals: dict[str, str] = {}
        self._last_active:  str | None     = None
        self._last_send_t:  float          = 0.0
        self._tx_log:       list[str]      = []
        self._rx_log:       list[str]      = []

        if self._simulate:
            print("[Serial] Simulation mode — no hardware needed.")
            return

        target = port or (find_arduino_port() if CFG["auto_detect"] else None)

        if target is None:
            print("[Serial] No Arduino detected. Available ports:",
                  list_serial_ports() or "none")
            print("[Serial] → Switching to simulation mode.")
            self._simulate = True
            return

        try:
            self._conn = serial.Serial(
                target,
                baudrate = baud_rate,
                timeout  = CFG["timeout_s"],
            )
            # Arduino resets on DTR toggle — wait for it to boot
            print(f"[Serial] Opening {target} @ {baud_rate} baud "
                  f"(waiting {CFG['connect_wait_s']:.1f}s for Arduino boot) …")
            time.sleep(CFG["connect_wait_s"])

            # Flush any boot-up garbage
            self._conn.reset_input_buffer()

            # Send a ping to confirm the firmware is running
            self._conn.write(b"P\n")
            reply = self._conn.readline().decode("ascii", errors="ignore").strip()
            if reply in ("OK", "READY"):
                print(f"[Serial] Arduino on {target} — READY.")
            else:
                print(f"[Serial] Arduino replied: '{reply}' (expected OK/READY)")

        except Exception as exc:
            print(f"[Serial] Could not open {target}: {exc}")
            print("[Serial] → Switching to simulation mode.")
            if self._conn and self._conn.is_open:
                self._conn.close()
            self._conn     = None
            self._simulate = True

    # ──────────────────────────────────────────────────────────────
    #  Protocol A — send_active_lane  (single character, most efficient)
    # ──────────────────────────────────────────────────────────────
    def send_active_lane(self, active_zone: str | None) -> None:
        """
        Send a single-character command for the currently active GREEN lane.

        Compared to send_signals(), this transmits just 1 byte instead of
        multiple "SIGNAL:..." strings.  The Arduino firmware switches the
        named lane to GREEN and all others to RED automatically.

        Parameters
        ----------
        active_zone : "NORTH" | "SOUTH" | "EAST" | "WEST"
                      None or "UNKNOWN" → sends 'X' (all RED)
        """
        now = time.time()
        if now - self._last_send_t < CFG["send_interval_s"]:
            return
        if active_zone == self._last_active:
            return   # no change — don't spam the Arduino

        zone = (active_zone or "").upper()
        cmd  = CFG["zone_to_char"].get(zone, b"X")

        self._transmit(cmd, label=f"ACTIVE_LANE:{zone}")
        self._last_active  = active_zone
        self._last_send_t  = now

    # ──────────────────────────────────────────────────────────────
    #  Protocol B — send_signals  (full per-lane state dict)
    # ──────────────────────────────────────────────────────────────
    def send_signals(self, signals: dict[str, str]) -> None:
        """
        Transmit the per-lane signal states, sending only changed lanes.

        Parameters
        ----------
        signals : dict mapping lane name → signal state
                  e.g. {"NORTH": "GREEN", "SOUTH": "RED",
                         "EAST": "RED",   "WEST": "RED"}
        """
        now = time.time()
        if now - self._last_send_t < CFG["send_interval_s"]:
            return

        changed = {
            lane: state
            for lane, state in signals.items()
            if (lane in KNOWN_LANES
                and state in KNOWN_STATES
                and self._last_signals.get(lane) != state)
        }

        if not changed:
            return

        # Always send GREEN commands first so the Arduino's yellow-transition
        # logic runs before the RED commands arrive.
        priority_order = sorted(
            changed.items(),
            key=lambda kv: (0 if kv[1] == "GREEN" else 1),
        )

        for lane, state in priority_order:
            cmd = f"SIGNAL:{lane}:{state}\n"
            self._transmit(cmd.encode("ascii"), label=cmd.strip())

        self._last_signals = dict(signals)
        self._last_send_t  = now

    # ──────────────────────────────────────────────────────────────
    #  all_red  — safe-state convenience
    # ──────────────────────────────────────────────────────────────
    def all_red(self) -> None:
        """Force all lanes RED immediately (sends 'X' command)."""
        self._transmit(b"X\n", label="ALL_RED")
        self._last_active  = None
        self._last_signals = {}

    # ──────────────────────────────────────────────────────────────
    #  ping  — connection health check
    # ──────────────────────────────────────────────────────────────
    def ping(self) -> bool:
        """
        Send a ping and wait for "OK".
        Returns True if Arduino responds within timeout.
        Always returns True in simulation mode.
        """
        if self._simulate:
            return True
        try:
            self._conn.write(b"P\n")
            self._conn.flush()
            reply = self._conn.readline().decode("ascii", errors="ignore").strip()
            return reply == "OK"
        except Exception:
            return False

    # ──────────────────────────────────────────────────────────────
    #  read_response  — non-blocking ACK reader
    # ──────────────────────────────────────────────────────────────
    def read_response(self) -> Optional[str]:
        """
        Read one pending line from Arduino (non-blocking).
        Returns None when nothing is available or in simulation mode.
        """
        if self._simulate or self._conn is None:
            return None
        try:
            if self._conn.in_waiting > 0:
                raw  = self._conn.readline()
                line = raw.decode("ascii", errors="ignore").strip()
                if line:
                    entry = f"[{time.strftime('%H:%M:%S')}] RX: {line}"
                    self._rx_log.append(entry)
                    if self._verbose:
                        print(entry)
                    return line
        except Exception:
            pass
        return None

    def drain_responses(self) -> list[str]:
        """Read all pending lines from Arduino. Returns list (may be empty)."""
        lines = []
        while True:
            line = self.read_response()
            if line is None:
                break
            lines.append(line)
        return lines

    # ──────────────────────────────────────────────────────────────
    #  Internal transmit
    # ──────────────────────────────────────────────────────────────
    def _transmit(self, data: bytes, label: str = "") -> None:
        entry = f"[{time.strftime('%H:%M:%S')}] TX: {label or data}"
        self._tx_log.append(entry)
        if self._verbose:
            print(entry)

        if self._simulate:
            print(f"[Serial SIM] {label or data}")
            return

        try:
            self._conn.write(data)
            self._conn.flush()
        except Exception as exc:
            print(f"[Serial] Write error: {exc} — switching to simulation mode.")
            self._simulate = True

    # ──────────────────────────────────────────────────────────────
    #  Diagnostics
    # ──────────────────────────────────────────────────────────────
    def get_tx_log(self) -> list[str]:
        """Return all transmitted command log entries."""
        return list(self._tx_log)

    def get_rx_log(self) -> list[str]:
        """Return all received response log entries."""
        return list(self._rx_log)

    def status(self) -> dict:
        """Return a snapshot of current controller state."""
        return {
            "simulated":      self._simulate,
            "connected":      self.is_connected,
            "last_active":    self._last_active,
            "last_signals":   dict(self._last_signals),
            "tx_count":       len(self._tx_log),
            "rx_count":       len(self._rx_log),
        }

    # ──────────────────────────────────────────────────────────────
    #  Properties
    # ──────────────────────────────────────────────────────────────
    @property
    def is_simulated(self) -> bool:
        return self._simulate

    @property
    def is_connected(self) -> bool:
        return (not self._simulate
                and self._conn is not None
                and self._conn.is_open)

    # ──────────────────────────────────────────────────────────────
    #  Lifecycle
    # ──────────────────────────────────────────────────────────────
    def close(self) -> None:
        """Send all-red command and close the serial port."""
        if self.is_connected:
            try:
                self._conn.write(b"X\n")
                self._conn.flush()
                self._conn.close()
            except Exception:
                pass
            print("[Serial] Port closed.")

    def __enter__(self): return self
    def __exit__(self, *_): self.close()


# ──────────────────────────────────────────────────────────────────
#  HUD overlay helper  (used by main.py)
# ──────────────────────────────────────────────────────────────────
def draw_comm_status(
    frame,
    ctrl:    SerialController,
    signals: dict[str, str],
) -> None:
    """
    Draw a small serial-connection status badge in the bottom-right
    corner of *frame*.
    """
    try:
        import cv2
        import numpy as np
    except ImportError:
        return

    h, w   = frame.shape[:2]
    mode   = "SIMULATED" if ctrl.is_simulated else "ARDUINO"
    color  = (0, 180, 255) if ctrl.is_simulated else (0, 220, 80)
    text   = f"Serial: {mode}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    x, y = w - tw - 16, h - 14
    cv2.putText(frame, text, (x+1, y+1),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,0), 2, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color,   1, cv2.LINE_AA)


# ──────────────────────────────────────────────────────────────────
#  Standalone demo / connection test
# ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Arduino serial connection test")
    ap.add_argument("--port",     default=None, help="Serial port (auto-detect if omitted)")
    ap.add_argument("--simulate", action="store_true", help="Force simulation mode")
    ap.add_argument("--verbose",  action="store_true", help="Print every TX/RX")
    args = ap.parse_args()

    print("=" * 50)
    print("  Arduino Serial Communication Test")
    print("=" * 50)

    with SerialController(
        port     = args.port,
        simulate = args.simulate,
        verbose  = args.verbose,
    ) as ctrl:

        print(f"\nStatus: {ctrl.status()}\n")

        lanes = ["NORTH", "SOUTH", "EAST", "WEST"]

        for lane in lanes:
            print(f"→ Activating {lane} (Protocol A) …")
            ctrl.send_active_lane(lane)
            # Drain any ACKs
            time.sleep(3.5)   # let yellow transition finish
            acks = ctrl.drain_responses()
            for a in acks:
                print(f"  ← {a}")

        print("\n→ Sending all-RED (Protocol B) …")
        ctrl.send_signals({l: "RED" for l in lanes})
        time.sleep(0.5)
        for a in ctrl.drain_responses():
            print(f"  ← {a}")

        print("\nDone.")
        print(f"TX log ({len(ctrl.get_tx_log())} entries):")
        for entry in ctrl.get_tx_log()[-8:]:
            print(f"  {entry}")
