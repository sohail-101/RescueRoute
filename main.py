"""
main.py
Hybrid Traffic Management System
===================================
Entry point — automatically selects input mode AND detection scene.

INPUT MODE (auto-selected by CLI flags)
─────────────────────────────────────────
  Mode 1  : --north/south/east/west provided → 4-stream 2×2 grid
  Mode 2  : single --source (or webcam 0)   → top-view zone overlay

SCENE MODE (detection behaviour)
─────────────────────────────────
  --auto   (default) rolling heuristic — adapts frame by frame
  --toy    force toy-vehicle mode (small blobs, bright colours, close-up)
  --real   force real-traffic mode (YOLO primary, larger blobs, BG subtraction)

HOW DETECTION ADAPTS
────────────────────
  Toy mode  : M4 (toy-vehicle detector) enabled, smaller area thresholds,
              lower YOLO confidence bar, M3 (flash) disabled.
  Real mode : M1 (YOLO) + M2 (cross) + M3 (flash) at real-scale thresholds.
  Auto mode : votes each frame; label shown in HUD and panel badge.

USAGE
─────
  python main.py                                    # webcam, auto scene
  python main.py --source intersection.mp4          # video file, auto
  python main.py --source toy_demo.mp4 --toy        # force toy mode
  python main.py --source cctv.mp4     --real       # force real mode
  python main.py --north n.mp4 --south s.mp4 \\
                 --east  e.mp4 --west  w.mp4        # Mode 1, auto scene
  python main.py --simulate --source demo.mp4       # treat cars as ambulances
  python main.py --source vid.mp4 --model yolov8s.pt --save
  python main.py --source vid.mp4 --port COM3       # explicit Arduino port
  python main.py --source vid.mp4 --no-arduino      # disable Arduino entirely

KEYBOARD CONTROLS (both modes)
───────────────────────────────
  ESC / Q   quit
  SPACE     pause / resume
  S         screenshot
  V         toggle vehicle boxes
  M         toggle method debug tags (shows [yolo]/[cross]/[toy]/[flash])
  P         toggle embedded panel overlay
  Z         toggle zone-debug window  (Mode 2 only)
  A         toggle Arduino serial output on/off at runtime
  H         print this help
"""

import argparse
import cv2
import sys
import time
import numpy as np

from vehicle_detection import (HybridDetector, create_bg_subtractor,
                                draw_detections, CFG as DET_CFG)
from multi_input        import MultiStreamReader, build_grid, ROAD_NAMES
from zone_detection     import build_zones, draw_zones
from decision           import DecisionEngine, DecisionResult
from traffic_light_ui   import (overlay_on_frame, show_panel,
                                 init_panel_window)
from communication      import SerialController, draw_comm_status

DEFAULTS = {"model": "yolov8n.pt", "out_path": "output.mp4"}


# ─────────────────────────────────────────────────────────
#  FPS counter
# ─────────────────────────────────────────────────────────
class _FPS:
    def __init__(self,n=24):
        self._t=[]; self._n=n
    def tick(self):
        self._t.append(time.perf_counter())
        if len(self._t)>self._n: self._t.pop(0)
        if len(self._t)<2: return 0.0
        return (len(self._t)-1)/(self._t[-1]-self._t[0])


# ─────────────────────────────────────────────────────────
#  Shared HUD helpers
# ─────────────────────────────────────────────────────────
def _sh(img,text,pos,scale,color,thick=1):
    cv2.putText(img,text,(pos[0]+1,pos[1]+1),
                cv2.FONT_HERSHEY_SIMPLEX,scale,(0,0,0),thick+1,cv2.LINE_AA)
    cv2.putText(img,text,pos,cv2.FONT_HERSHEY_SIMPLEX,scale,color,thick,cv2.LINE_AA)


_PHASE_COL = {"CYCLING":(0,200,80),"PRIORITY":(0,0,220),"YELLOW":(0,200,255)}
_SCENE_COL = {"toy":(0,220,128),"real":(0,200,255),"unknown":(140,140,140)}


def _top_bar(frame,fps,frame_n,result,mode,scene):
    h,w = frame.shape[:2]
    ov  = frame.copy()
    cv2.rectangle(ov,(0,0),(w,32),(12,12,12),-1)
    cv2.addWeighted(ov,0.78,frame,0.22,0,frame)
    pc  = _PHASE_COL.get(result.phase,(160,160,160))
    sc  = _SCENE_COL.get(scene,(140,140,140))
    left  = f"  Hybrid Traffic Mgmt — Mode {mode}  |  frame {frame_n}"
    right = "V M P Z = toggles  |  SPACE=pause  |  ESC=quit"
    _sh(frame,left,(6,22),0.42,(180,180,180))
    _sh(frame,f"[{result.phase}]",(len(left)*5+20,22),0.50,pc)
    scene_tag = f"[{scene.upper()}]"
    (stw,_),_=cv2.getTextSize(scene_tag,cv2.FONT_HERSHEY_SIMPLEX,0.40,1)
    (rtw,_),_=cv2.getTextSize(right,    cv2.FONT_HERSHEY_SIMPLEX,0.34,1)
    _sh(frame,scene_tag,(w-rtw-stw-18,22),0.40,sc)
    _sh(frame,right,(w-rtw-6,22),0.34,(100,100,100))


def _stats_overlay(frame,fps,result):
    total = len(result.detections)
    pri   = sum(1 for d in result.detections if d["is_priority"])

    # Detection method breakdown
    methods: dict[str,int] = {}
    for d in result.detections:
        m = d.get("method","?")
        methods[m] = methods.get(m,0)+1
    method_str = "  ".join(f"{m}:{n}" for m,n in methods.items())

    lines = [
        (f"FPS: {fps:.1f}",     (200,200,200)),
        (f"Vehicles: {total}",  (200,200,200)),
        (f"Priority: {pri}",    (0,0,220) if pri>0 else (200,200,200)),
    ]
    if result.priority_zone:
        lines.append((f"Road: {result.priority_zone}", (80,80,255)))
    elif result.active_zone:
        lines.append((f"Green: {result.active_zone}",  (0,200,80)))
    if method_str:
        lines.append((method_str, (130,130,130)))

    for i,(txt,col) in enumerate(lines):
        _sh(frame,txt,(12,50+i*22),0.52,col)


def _priority_banner(frame,result,fn):
    if not result.priority_active: return
    h,w  = frame.shape[:2]
    bw,bh= 510,38
    bx   = (w-bw)//2; by = h-bh-12
    if (fn//6)%2==0:
        ov=frame.copy()
        cv2.rectangle(ov,(bx,by),(bx+bw,by+bh),(0,0,120),-1)
        cv2.addWeighted(ov,0.80,frame,0.20,0,frame)
    msg = f"AMBULANCE IN {result.priority_zone} — GREEN CORRIDOR ACTIVE"
    (mw,_),_=cv2.getTextSize(msg,cv2.FONT_HERSHEY_SIMPLEX,0.52,1)
    _sh(frame,msg,(bx+(bw-mw)//2,by+bh-10),0.52,(70,70,255),thick=2)


def _default_result() -> DecisionResult:
    return DecisionResult(
        signals={z:"RED" for z in ROAD_NAMES},
        phase="CYCLING", priority_active=False,
        priority_zone=None, active_zone="NORTH",
        detections=[], zone_counts={z:0 for z in ROAD_NAMES+["INTERSECTION"]},
    )


# ─────────────────────────────────────────────────────────
#  Zone debug window
# ─────────────────────────────────────────────────────────
def _zone_debug(frame,zones,result):
    dbg = frame.copy()
    _ZC = {"NORTH":(200,80,0),"SOUTH":(0,160,200),"EAST":(0,200,80),"WEST":(80,0,200)}
    ov  = dbg.copy()
    for name,zone in zones.items():
        x1,y1,x2,y2=zone.rect
        cv2.rectangle(ov,(x1,y1),(x2,y2),_ZC[name],-1)
    cv2.addWeighted(ov,0.22,dbg,0.78,0,dbg)
    for det in result.detections:
        cx,cy=det["centroid"]
        col=(0,0,255) if det["is_priority"] else (0,220,255)
        cv2.circle(dbg,(cx,cy),7,col,-1,cv2.LINE_AA)
        cv2.circle(dbg,(cx,cy),7,(255,255,255),1,cv2.LINE_AA)
    cv2.imshow("Zone Debug",dbg)


# ─────────────────────────────────────────────────────────
#  MODE 1 — multi-stream 2×2 grid
# ─────────────────────────────────────────────────────────
def run_mode1(sources,model_path,save,scene_mode,arduino_port,use_arduino):
    print(f"\n[INFO] ── MODE 1  scene={scene_mode} ──────────────────")
    detector = HybridDetector(model_path, scene=scene_mode)
    engine   = DecisionEngine()
    bgsubbers= {road: create_bg_subtractor() for road in sources}

    # ── Arduino serial controller ─────────────────────────────────
    ctrl = SerialController(
        port     = arduino_port,
        simulate = not use_arduino,
    )

    win = "Intersection Monitor [Mode 1]"
    cv2.namedWindow(win,cv2.WINDOW_NORMAL)
    init_panel_window()

    tog    = {"vehicles":True,"panel":True,"method":False,"arduino":use_arduino}
    fps_c  = _FPS(); frame_n=0; paused=False; shots=0
    result = _default_result(); fps_v=0.0; writer=None

    with MultiStreamReader(sources) as reader:
        print("[INFO] ESC/Q=quit  SPACE=pause  V=vehicles  M=method  P=panel  A=arduino  S=screenshot\n")
        while True:
            if not paused:
                frames = reader.read_all()
                per_road: dict[str,list[dict]] = {}
                for road in ROAD_NAMES:
                    frame = frames.get(road)
                    if frame is None:
                        per_road[road]=[]; continue
                    dets = detector.detect(frame,
                                           bgsub=bgsubbers.get(road),
                                           preset_zone=road)
                    per_road[road] = dets
                    if tog["vehicles"]:
                        draw_detections(frame,dets,show_zone=False,
                                        show_method=tog["method"])
                    frames[road] = frame

                result  = engine.update_multi(per_road)
                frame_n += 1; fps_v = fps_c.tick()
                scene   = detector.scene

                # ── Arduino: send active lane command ─────────────
                if tog["arduino"]:
                    ctrl.send_active_lane(result.active_zone)
                    for ack in ctrl.drain_responses():
                        print(f"[Arduino] {ack}")

                grid = build_grid(frames,result.signals,cell_w=480,cell_h=360)
                if tog["panel"]:
                    overlay_on_frame(grid,result,frame_n,fps_v,
                                     scale=0.46,corner="bottom-right",scene=scene)
                _priority_banner(grid,result,frame_n)
                _stats_overlay(grid,fps_v,result)
                _top_bar(grid,fps_v,frame_n,result,1,scene)
                draw_comm_status(grid, ctrl, result.signals)

                cv2.imshow(win,grid)
                show_panel(result,frame_n,fps_v,scene)

                if writer is None and save:
                    h2,w2=grid.shape[:2]
                    fourcc=cv2.VideoWriter_fourcc(*"mp4v")
                    writer=cv2.VideoWriter(DEFAULTS["out_path"],fourcc,reader.fps,(w2,h2))
                if writer: writer.write(grid)

            key=cv2.waitKey(1)&0xFF
            if key in(27,ord("q")):   break
            elif key==ord(" "):       paused=not paused; print("[INFO]","Paused."if paused else"Resumed.")
            elif key==ord("s"):
                shots+=1; fn=f"screenshot_{shots:04d}.png"
                cv2.imwrite(fn,grid); print(f"[INFO] Saved {fn}")
            elif key==ord("v"):  tog["vehicles"]=not tog["vehicles"]; print(f"[Tog] Vehicles:{tog['vehicles']}")
            elif key==ord("m"):  tog["method"]  =not tog["method"];   print(f"[Tog] Method tags:{tog['method']}")
            elif key==ord("p"):  tog["panel"]   =not tog["panel"];    print(f"[Tog] Panel:{tog['panel']}")
            elif key==ord("a"):
                tog["arduino"] = not tog["arduino"]
                if not tog["arduino"]: ctrl.all_red()
                print(f"[Tog] Arduino output: {tog['arduino']}")
            elif key==ord("h"):  print(__doc__)

    ctrl.close()
    if writer: writer.release()
    cv2.destroyAllWindows()
    print(f"[INFO] Mode 1 — {frame_n} frames processed.")


# ─────────────────────────────────────────────────────────
#  MODE 2 — single top-view camera
# ─────────────────────────────────────────────────────────
def run_mode2(source,model_path,save,scene_mode,arduino_port,use_arduino):
    print(f"\n[INFO] ── MODE 2  scene={scene_mode} ──────────────────")
    detector = HybridDetector(model_path, scene=scene_mode)
    engine   = DecisionEngine()
    bgsub    = create_bg_subtractor()

    # ── Arduino serial controller ─────────────────────────────────
    ctrl = SerialController(
        port     = arduino_port,
        simulate = not use_arduino,
    )

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open: {source}"); sys.exit(1)

    fps_src = cap.get(cv2.CAP_PROP_FPS) or 30.0
    fw      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    delay   = max(1,int(1000/fps_src))
    zones   = build_zones(fw,fh)

    writer=None
    if save:
        writer=cv2.VideoWriter(DEFAULTS["out_path"],
                               cv2.VideoWriter_fourcc(*"mp4v"),
                               fps_src,(fw,fh))

    win="Intersection Monitor [Mode 2]"
    cv2.namedWindow(win,cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win,min(fw,1280),min(fh,720))
    init_panel_window()

    tog    = {"vehicles":True,"panel":True,"method":False,"zone_dbg":False,"arduino":use_arduino}
    fps_c  = _FPS(); frame_n=0; paused=False; shots=0
    result = _default_result(); fps_v=0.0

    print("[INFO] ESC/Q=quit  SPACE=pause  V=vehicles  M=method  P=panel  Z=zone-debug  A=arduino  S=screenshot\n")

    while True:
        if not paused:
            ret,raw = cap.read()
            if not ret:
                print("[INFO] End of stream.")
                # Loop video file
                cap.set(cv2.CAP_PROP_POS_FRAMES,0)
                ret,raw = cap.read()
                if not ret: break

            frame_n += 1
            out = raw.copy()
            scene = detector.scene

            dets   = detector.detect(raw, bgsub=bgsub)
            result = engine.update(dets,zones)

            # ── Arduino: send active lane command ─────────────────
            if tog["arduino"]:
                ctrl.send_active_lane(result.active_zone)
                for ack in ctrl.drain_responses():
                    print(f"[Arduino] {ack}")

            draw_zones(out,zones,result.signals,result.detections)
            if tog["vehicles"]:
                draw_detections(out,result.detections,show_zone=True,
                                show_method=tog["method"])

            fps_v = fps_c.tick()
            if tog["panel"]:
                overlay_on_frame(out,result,frame_n,fps_v,
                                 scale=0.48,corner="top-right",scene=scene)

            _priority_banner(out,result,frame_n)
            _stats_overlay(out,fps_v,result)
            _top_bar(out,fps_v,frame_n,result,2,scene)
            draw_comm_status(out, ctrl, result.signals)

            cv2.imshow(win,out)
            show_panel(result,frame_n,fps_v,scene)

            if tog["zone_dbg"]:
                _zone_debug(raw,zones,result)

            if writer: writer.write(out)

        key=cv2.waitKey(delay)&0xFF
        if key in(27,ord("q")):   break
        elif key==ord(" "):       paused=not paused; print("[INFO]","Paused."if paused else"Resumed.")
        elif key==ord("s"):
            shots+=1; fn=f"screenshot_{shots:04d}.png"
            cv2.imwrite(fn,out if not paused else raw); print(f"[INFO] Saved {fn}")
        elif key==ord("v"):   tog["vehicles"] =not tog["vehicles"];  print(f"[Tog] Vehicles:{tog['vehicles']}")
        elif key==ord("m"):   tog["method"]   =not tog["method"];    print(f"[Tog] Method:{tog['method']}")
        elif key==ord("p"):   tog["panel"]    =not tog["panel"];     print(f"[Tog] Panel:{tog['panel']}")
        elif key==ord("a"):
            tog["arduino"] = not tog["arduino"]
            if not tog["arduino"]: ctrl.all_red()
            print(f"[Tog] Arduino output: {tog['arduino']}")
        elif key==ord("z"):
            tog["zone_dbg"]=not tog["zone_dbg"]
            if not tog["zone_dbg"]: cv2.destroyWindow("Zone Debug")
            print(f"[Tog] Zone debug:{tog['zone_dbg']}")
        elif key==ord("h"):   print(__doc__)

    cap.release()
    ctrl.close()
    if writer: writer.release()
    cv2.destroyAllWindows()
    print(f"[INFO] Mode 2 — {frame_n} frames processed.")


# ─────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Hybrid Traffic Management System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                                          # webcam, auto scene
  python main.py --source intersection.mp4               # top-view video
  python main.py --source toy_demo.mp4 --toy             # toy setup
  python main.py --source cctv.mp4     --real            # real CCTV
  python main.py --north n.mp4 --south s.mp4 --east e.mp4 --west w.mp4
  python main.py --simulate --source demo.mp4            # demo: car = ambulance
  python main.py --source vid.mp4 --model yolov8s.pt --save
""",
    )
    # Input mode
    ap.add_argument("--north",  default=None, help="North road source (Mode 1)")
    ap.add_argument("--south",  default=None, help="South road source (Mode 1)")
    ap.add_argument("--east",   default=None, help="East road source (Mode 1)")
    ap.add_argument("--west",   default=None, help="West road source (Mode 1)")
    ap.add_argument("--source", default="0",  help="Single source (Mode 2, default: webcam 0)")

    # Scene mode (mutually exclusive)
    sg = ap.add_mutually_exclusive_group()
    sg.add_argument("--auto", dest="scene", action="store_const", const="auto",
                    default="auto", help="Auto-detect scene (default)")
    sg.add_argument("--toy",  dest="scene", action="store_const", const="toy",
                    help="Force toy-vehicle mode")
    sg.add_argument("--real", dest="scene", action="store_const", const="real",
                    help="Force real-traffic mode")

    # Common
    ap.add_argument("--model",    default=DEFAULTS["model"],
                    help="YOLOv8 weights (default: yolov8n.pt)")
    ap.add_argument("--save",     action="store_true",
                    help=f"Record output to {DEFAULTS['out_path']}")
    ap.add_argument("--simulate", action="store_true",
                    help="Treat first 'car' detection as ambulance (demo fallback)")
    ap.add_argument("--no-yolo",  action="store_true",
                    help="Disable YOLO — use color methods only (fast, no GPU needed)")
    ap.add_argument("--port",       default=None,
                    help="Arduino serial port e.g. COM3 or /dev/ttyUSB0 (auto-detect if omitted)")
    ap.add_argument("--no-arduino", action="store_true",
                    help="Disable Arduino serial output entirely (pure software mode)")
    args = ap.parse_args()

    if args.simulate:
        DET_CFG["simulate_priority"] = True
        print("[INFO] Simulation: first car → ambulance.")

    model_path = None if args.no_yolo else args.model

    def _src(s):
        return int(s) if isinstance(s,str) and s.isdigit() else s

    road_sources = {r.upper(): _src(v)
                    for r,v in [("NORTH",args.north),("SOUTH",args.south),
                                ("EAST",args.east),("WEST",args.west)]
                    if v is not None}

    scene_mode = args.scene or "auto"
    print(f"[INFO] Scene mode: {scene_mode}")

    use_arduino  = not args.no_arduino
    arduino_port = args.port

    if road_sources:
        run_mode1(road_sources, model_path, args.save, scene_mode,
                  arduino_port, use_arduino)
    else:
        run_mode2(_src(args.source), model_path, args.save, scene_mode,
                  arduino_port, use_arduino)


if __name__ == "__main__":
    main()
