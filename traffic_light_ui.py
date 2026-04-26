"""
traffic_light_ui.py
Hybrid Traffic Management System
===================================
Virtual Traffic Light Panel — 4-direction compass layout.

Panel: 480 × 440 px
  Header  : phase banner + scene mode badge
  Body    : N/S/E/W traffic lights in compass positions
  Footer  : fps | vehicles | priority | active zone

Public API
──────────
    panel = render_panel(result, frame_count, fps, scene)
    frame = overlay_on_frame(frame, result, frame_count, fps, scene)
    init_panel_window()
    show_panel(result, frame_count, fps, scene)
"""

import cv2
import math
import numpy as np
from decision import DecisionResult

PANEL_W  = 480;  PANEL_H  = 440
HEADER_H = 52;   FOOTER_H = 44
BODY_H   = PANEL_H - HEADER_H - FOOTER_H

HOUSING_W = 68;  HOUSING_H = 158;  LIGHT_R = 18

BG         = (20, 20, 20)
HOUSING_BG = (33, 33, 33)
HOUSING_BD = (58, 58, 58)
POLE_COL   = (48, 48, 48)

LAMP = {
    "RED":    {"on": (0,   0, 230), "off": (30,  8, 60)},
    "YELLOW": {"on": (0, 200, 255), "off": (18, 48, 60)},
    "GREEN":  {"on": (0, 220,   0), "off": ( 8, 52, 18)},
}

PHASE_COL = {"CYCLING":(32,32,32), "PRIORITY":(28,8,8), "YELLOW":(28,38,8)}
PHASE_FG  = {"CYCLING":(170,170,170), "PRIORITY":(80,80,255), "YELLOW":(60,200,255)}
PHASE_LBL = {
    "CYCLING":  "NORMAL TRAFFIC — CYCLING",
    "PRIORITY": "AMBULANCE — GREEN CORRIDOR",
    "YELLOW":   "CLEARING — RETURNING TO NORMAL",
}
SIG_COL = {"GREEN":(0,210,0), "RED":(0,0,210), "YELLOW":(0,190,255)}
_ARROWS  = {"NORTH":"↑N","SOUTH":"↓S","EAST":"→E","WEST":"←W"}

# Scene badge colours
_SCENE_BADGE = {
    "toy":     ((0,220,128), "TOY MODE"),
    "real":    ((0,200,255), "REAL TRAFFIC"),
    "unknown": ((100,100,100), "AUTO"),
}


def _positions(body_y: int) -> dict[str, tuple[int,int]]:
    cx = PANEL_W//2; cy = body_y + BODY_H//2; pad = 42
    return {
        "NORTH": (cx,           body_y+pad),
        "SOUTH": (cx,           body_y+BODY_H-pad),
        "WEST":  (pad,          cy),
        "EAST":  (PANEL_W-pad,  cy),
    }


def _shadow(img, text, pos, scale, color, thick=1):
    cv2.putText(img,text,(pos[0]+1,pos[1]+1),
                cv2.FONT_HERSHEY_SIMPLEX,scale,(0,0,0),thick+1,cv2.LINE_AA)
    cv2.putText(img,text,pos,cv2.FONT_HERSHEY_SIMPLEX,scale,color,thick,cv2.LINE_AA)


def _rrect(img,pt1,pt2,color,rx=8,fill=True,thick=1):
    x1,y1=pt1; x2,y2=pt2
    r=min(rx,(x2-x1)//2,(y2-y1)//2)
    if fill:
        cv2.rectangle(img,(x1+r,y1),(x2-r,y2),color,-1)
        cv2.rectangle(img,(x1,y1+r),(x2,y2-r),color,-1)
        for p in[(x1+r,y1+r),(x2-r,y1+r),(x1+r,y2-r),(x2-r,y2-r)]:
            cv2.circle(img,p,r,color,-1)
    else:
        for p in[(x1+r,y1+r),(x2-r,y1+r),(x1+r,y2-r),(x2-r,y2-r)]:
            cv2.ellipse(img,p,(r,r),0,0,360,color,thick)
        cv2.line(img,(x1+r,y1),(x2-r,y1),color,thick)
        cv2.line(img,(x1+r,y2),(x2-r,y2),color,thick)
        cv2.line(img,(x1,y1+r),(x1,y2-r),color,thick)
        cv2.line(img,(x2,y1+r),(x2,y2-r),color,thick)


def _glow(img,centre,radius,color,layers=4):
    for i in range(layers,0,-1):
        ov=img.copy()
        cv2.circle(ov,centre,radius+i*4,color,-1,cv2.LINE_AA)
        cv2.addWeighted(ov,0.08*i/layers,img,1-0.08*i/layers,0,img)


def _draw_light(panel,cx,cy,zone,signal,is_priority,frame_count):
    horiz = zone in ("NORTH","SOUTH")
    hw,hh = (HOUSING_H,HOUSING_W) if horiz else (HOUSING_W,HOUSING_H)
    hx1,hy1 = cx-hw//2, cy-hh//2
    hx2,hy2 = cx+hw//2, cy+hh//2

    poles = {
        "NORTH": ((cx-3,hy2),(cx+3,hy2+26)),
        "SOUTH": ((cx-3,hy1-26),(cx+3,hy1)),
        "WEST":  ((hx2,cy-3),(hx2+26,cy+3)),
        "EAST":  ((hx1-26,cy-3),(hx1,cy+3)),
    }
    if zone in poles:
        cv2.rectangle(panel,poles[zone][0],poles[zone][1],POLE_COL,-1)

    _rrect(panel,(hx1,hy1),(hx2,hy2),HOUSING_BG,rx=8,fill=True)
    if is_priority and signal=="GREEN":
        pulse = int(abs(math.sin(frame_count*0.14))*200)+55
        _rrect(panel,(hx1-3,hy1-3),(hx2+3,hy2+3),(0,0,pulse),rx=11,fill=False,thick=3)
    _rrect(panel,(hx1,hy1),(hx2,hy2),HOUSING_BD,rx=8,fill=False,thick=2)

    lamps = ["RED","YELLOW","GREEN"]
    gap   = (hw if horiz else hh)//4
    lc_list = ([(hx1+gap,cy),(hx1+2*gap,cy),(hx1+3*gap,cy)] if horiz
               else [(cx,hy1+gap),(cx,hy1+2*gap),(cx,hy1+3*gap)])

    for lamp_name,lc in zip(lamps,lc_list):
        active = (lamp_name==signal)
        lc_    = LAMP[lamp_name]["on" if active else "off"]
        if active: _glow(panel,lc,LIGHT_R,lc_)
        cv2.circle(panel,lc,LIGHT_R,lc_,-1,cv2.LINE_AA)
        cv2.circle(panel,lc,LIGHT_R,(70,70,70),1,cv2.LINE_AA)
        if active:
            cv2.circle(panel,(lc[0]-LIGHT_R//3,lc[1]-LIGHT_R//3),
                       LIGHT_R//4,(255,255,255),-1,cv2.LINE_AA)

    sc = SIG_COL.get(signal,(150,150,150))
    if horiz:
        _shadow(panel,f"{_ARROWS[zone]} {signal}",(cx-22,hy2+16),0.38,sc)
    else:
        lx = hx2+6 if zone=="EAST" else hx1-46
        _shadow(panel,f"{_ARROWS[zone]} {signal}",(lx,cy+6),0.38,sc)

    if is_priority:
        bx1,by1,bx2,by2 = hx1+4,hy2-22,hx2-4,hy2-4
        cv2.rectangle(panel,(bx1,by1),(bx2,by2),(0,0,160),-1)
        cv2.putText(panel,"AMB",(bx1+4,by2-3),cv2.FONT_HERSHEY_SIMPLEX,
                    0.30,(255,255,255),1,cv2.LINE_AA)


def render_panel(
    result:      DecisionResult,
    frame_count: int   = 0,
    fps:         float = 0.0,
    scene:       str   = "unknown",
) -> np.ndarray:
    panel = np.full((PANEL_H,PANEL_W,3),BG,dtype=np.uint8)
    phase = result.phase

    # Header
    cv2.rectangle(panel,(0,0),(PANEL_W,HEADER_H),PHASE_COL.get(phase,(32,32,32)),-1)
    if phase=="PRIORITY" and (frame_count//7)%2==0:
        cv2.rectangle(panel,(0,0),(PANEL_W,HEADER_H),(0,0,28),-1)

    lbl = PHASE_LBL.get(phase,phase)
    fc  = PHASE_FG.get(phase,(170,170,170))
    (tw,_),_ = cv2.getTextSize(lbl,cv2.FONT_HERSHEY_SIMPLEX,0.55,1)
    _shadow(panel,lbl,((PANEL_W-tw)//2,HEADER_H-16),0.55,fc)

    if result.priority_zone:
        sub = f"Ambulance → {result.priority_zone} road"
        (sw,_),_ = cv2.getTextSize(sub,cv2.FONT_HERSHEY_SIMPLEX,0.38,1)
        _shadow(panel,sub,((PANEL_W-sw)//2,HEADER_H-2),0.38,(110,110,255))

    # Scene badge (top-right of header)
    s_col, s_txt = _SCENE_BADGE.get(scene, _SCENE_BADGE["unknown"])
    badge = f"◉ {s_txt}"
    (bw,_),_ = cv2.getTextSize(badge,cv2.FONT_HERSHEY_SIMPLEX,0.35,1)
    _shadow(panel,badge,(PANEL_W-bw-8,14),0.35,s_col)

    cv2.line(panel,(0,HEADER_H),(PANEL_W,HEADER_H),(55,55,55),1)

    # Traffic lights
    body_y = HEADER_H
    pos    = _positions(body_y)
    pri_zones = {d.get("zone") for d in result.detections if d.get("is_priority")}

    cx_p = PANEL_W//2; cy_p = body_y+BODY_H//2
    cv2.circle(panel,(cx_p,cy_p),24,(38,38,38),-1)
    cv2.circle(panel,(cx_p,cy_p),24,(56,56,56),1)
    _shadow(panel,"⊕",(cx_p-9,cy_p+7),0.55,(80,80,80))

    for zone_name,(lx,ly) in pos.items():
        sig    = result.signals.get(zone_name,"RED")
        is_pri = zone_name in pri_zones
        _draw_light(panel,lx,ly,zone_name,sig,is_pri,frame_count)

    # Footer
    footer_y = PANEL_H-FOOTER_H
    cv2.line(panel,(0,footer_y),(PANEL_W,footer_y),(55,55,55),1)
    total = len(result.detections)
    pri_n = sum(1 for d in result.detections if d.get("is_priority"))
    az    = result.active_zone or "—"
    items = [
        (f"FPS {fps:.0f}",     (130,130,130)),
        (f"Vehicles {total}",  (130,130,130)),
        (f"Priority {pri_n}",  (60,60,220) if pri_n>0 else (130,130,130)),
        (f"Green: {az}",       SIG_COL.get("GREEN",(0,200,0)) if az!="—" else (130,130,130)),
    ]
    iw = PANEL_W//len(items)
    for i,(txt,col) in enumerate(items):
        _shadow(panel,txt,(i*iw+6,PANEL_H-12),0.38,col)

    cv2.rectangle(panel,(0,0),(PANEL_W-1,PANEL_H-1),(65,65,65),1)
    return panel


def overlay_on_frame(
    frame:       np.ndarray,
    result:      DecisionResult,
    frame_count: int   = 0,
    fps:         float = 0.0,
    scale:       float = 0.50,
    corner:      str   = "top-right",
    scene:       str   = "unknown",
) -> np.ndarray:
    panel  = render_panel(result, frame_count, fps, scene)
    pw     = int(PANEL_W*scale); ph = int(PANEL_H*scale)
    small  = cv2.resize(panel,(pw,ph),interpolation=cv2.INTER_AREA)
    fh,fw  = frame.shape[:2]; pad=10; y_off=34
    corners= {
        "top-right":    (fw-pw-pad, y_off),
        "top-left":     (pad,       y_off),
        "bottom-right": (fw-pw-pad, fh-ph-pad),
        "bottom-left":  (pad,       fh-ph-pad),
    }
    x0,y0 = corners.get(corner,corners["top-right"])
    x0=max(0,min(x0,fw-pw)); y0=max(0,min(y0,fh-ph))
    roi    = frame[y0:y0+ph,x0:x0+pw]
    blended= cv2.addWeighted(small,0.90,roi,0.10,0)
    frame[y0:y0+ph,x0:x0+pw] = blended
    cv2.rectangle(frame,(x0,y0),(x0+pw,y0+ph),(65,65,65),1)
    return frame


_WIN = "Traffic Light Controller"

def init_panel_window() -> None:
    cv2.namedWindow(_WIN,cv2.WINDOW_NORMAL)
    cv2.resizeWindow(_WIN,PANEL_W,PANEL_H)

def show_panel(result, frame_count=0, fps=0.0, scene="unknown") -> None:
    cv2.imshow(_WIN, render_panel(result,frame_count,fps,scene))
