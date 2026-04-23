#!/usr/bin/env python3
"""
detector.py — STARKVIS Core Intelligence Module
Stark Threat Analysis and Recognition Knowledge Vision Intelligence System — Stark Industries
YOLOv8 anomaly detection + Iron Man HUD rendering
"""

import cv2
import numpy as np
import random
import math
from datetime import datetime
from ultralytics import YOLO

# ── Model singleton ───────────────────────────────────────────────────────────
_model = None

def get_model(weights: str = "yolov8n.pt") -> YOLO:
    global _model
    if _model is None:
        _model = YOLO(weights)
    return _model

# ── Iron Man / Arc Reactor color palette (BGR) ────────────────────────────────
ARC_BLUE     = (255, 210,  30)   # arc reactor blue-white
STARK_GOLD   = (0,   180, 255)   # Iron Man gold
DANGER_RED   = (30,   30, 220)   # alert red
WARN_ORANGE  = (0,   140, 255)   # warning orange
HUD_TEAL     = (200, 230,  80)   # HUD teal-green
DIM_BLUE     = (120, 100,  40)   # dimmed arc blue
WHITE        = (240, 240, 255)

SEVERITY_BGR = {
    "low":      STARK_GOLD,
    "medium":   WARN_ORANGE,
    "high":     DANGER_RED,
    "critical": (0, 0, 255),
}

# ── Anomaly classification ────────────────────────────────────────────────────
CRITICAL_OBJECTS = {"knife", "gun", "pistol", "rifle", "fire"}
HIGH_OBJECTS     = {"scissors", "smoke", "baseball bat"}
MEDIUM_OBJECTS   = {"cell phone", "remote"}
EXOTIC_OBJECTS   = {"bear", "elephant", "zebra", "giraffe", "horse", "cow", "sheep"}

def classify(label: str) -> tuple:
    """Returns (is_anomaly, severity, reason, threat_code)"""
    l = label.lower()
    if l in CRITICAL_OBJECTS:
        return True, "critical", f"Weapons-class object in frame: {label}", "WEAPONS-LOCK"
    if l in HIGH_OBJECTS:
        return True, "high", f"Potential threat object: {label}", "THREAT-ALPHA"
    if l in MEDIUM_OBJECTS:
        return True, "medium", f"Unauthorized device detected: {label}", "SIGNAL-TRACE"
    if l in EXOTIC_OBJECTS:
        return True, "medium", f"Anomalous entity — unexpected fauna: {label}", "BIO-ANOMALY"
    return False, "low", None, None

# ── YOLOv8 detection ──────────────────────────────────────────────────────────
def run_detection(image_path: str, weights: str = "yolov8n.pt",
                  conf: float = 0.35) -> dict:
    model   = get_model(weights)
    results = model(image_path, conf=conf, verbose=False)
    r       = results[0]
    img     = cv2.imread(image_path)
    h, w    = img.shape[:2]

    objects = []
    for i, box in enumerate(r.boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        confidence      = round(float(box.conf[0]) * 100)
        label           = r.names[int(box.cls[0])]
        is_anom, sev, reason, code = classify(label)

        objects.append({
            "id":          f"SID-{i+1:03d}",
            "name":        label,
            "is_anomaly":  is_anom,
            "reason":      reason,
            "severity":    sev,
            "threat_code": code,
            "confidence":  confidence,
            "bbox_pct": {
                "x": round(x1 / w * 100, 2),
                "y": round(y1 / h * 100, 2),
                "w": round((x2 - x1) / w * 100, 2),
                "h": round((y2 - y1) / h * 100, 2),
            },
        })

    anomalies    = [o for o in objects if o["is_anomaly"]]
    threat_index = min(100, len(anomalies) * 28 + (5 if objects else 0))
    status       = "THREAT CONFIRMED" if anomalies else "ALL CLEAR"

    return {
        "objects":      objects,
        "anomalies":    anomalies,
        "threat_index": threat_index,
        "status":       status,
        "object_count": len(objects),
        "anomaly_count":len(anomalies),
    }

# ── Iron Man HUD rendering ────────────────────────────────────────────────────

def _draw_arc_circle(img, cx, cy, radius, color, thickness=1, alpha=0.6):
    """Draw partial arc circle segments — STARKVIS HUD style."""
    overlay = img.copy()
    # Draw 4 arc segments with gaps
    for start_angle in [20, 110, 200, 290]:
        cv2.ellipse(overlay, (cx, cy), (radius, radius), 0,
                    start_angle, start_angle + 70, color, thickness)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


def _draw_hex_marker(img, cx, cy, size, color):
    """Draw a small hexagonal marker."""
    pts = []
    for i in range(6):
        angle = math.radians(60 * i - 30)
        pts.append([int(cx + size * math.cos(angle)),
                    int(cy + size * math.sin(angle))])
    cv2.polylines(img, [np.array(pts)], True, color, 1, cv2.LINE_AA)


def _draw_scan_grid(img):
    """Subtle perspective grid on lower half — Iron Man HUD floor projection."""
    h, w = img.shape[:2]
    overlay = img.copy()
    mid_x = w // 2
    horizon = int(h * 0.55)
    grid_color = (60, 50, 20)

    # Radiating lines from vanishing point
    for i in range(-8, 9):
        end_x = mid_x + i * (w // 14)
        cv2.line(overlay, (mid_x, horizon), (end_x, h), grid_color, 1)

    # Horizontal depth lines
    for j in range(6):
        t = (j + 1) / 7
        y = int(horizon + t * (h - horizon))
        x_spread = int(t * w * 0.55)
        cv2.line(overlay, (mid_x - x_spread, y),
                 (mid_x + x_spread, y), grid_color, 1)

    cv2.addWeighted(overlay, 0.25, img, 0.75, 0, img)


def apply_stark_grade(img: np.ndarray) -> np.ndarray:
    """Iron Man color grading — warm amber tones, slight blue in shadows."""
    b, g, r = cv2.split(img.astype(np.float32))

    # Warm highlights, cool shadows
    r = np.clip(r * 1.08 + 5, 0, 255)
    g = np.clip(g * 1.02,     0, 255)
    b = np.clip(b * 0.94,     0, 255)

    img = cv2.merge([b.astype(np.uint8),
                     g.astype(np.uint8),
                     r.astype(np.uint8)])

    # Subtle vignette
    rows, cols = img.shape[:2]
    X, Y = np.meshgrid(np.linspace(-1, 1, cols),
                       np.linspace(-1, 1, rows))
    vignette = np.clip(1.0 - (X**2 + Y**2) * 0.45, 0.35, 1.0)
    img = (img * vignette[:, :, np.newaxis]).astype(np.uint8)
    return img


def draw_starkvis_bbox(img, x1, y1, x2, y2, color, is_anomaly, label, sid, reason=""):
    """Draw Iron Man targeting reticle style bounding box."""
    h, w = img.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    bx   = x2 - x1
    by   = y2 - y1

    # Solid border — no glow copy (saves time, no lag)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

    # Corner L-brackets (solid, thick)
    cl = max(12, min(22, bx // 5, by // 5))
    ct = 3
    for cx2, cy2, dx, dy in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
        cv2.line(img, (cx2, cy2), (cx2 + dx*cl, cy2), color, ct, cv2.LINE_AA)
        cv2.line(img, (cx2, cy2), (cx2, cy2 + dy*cl), color, ct, cv2.LINE_AA)

    # Center crosshair
    cx3, cy3 = (x1 + x2) // 2, (y1 + y2) // 2
    cl2 = 7
    cv2.line(img, (cx3 - cl2, cy3), (cx3 + cl2, cy3), color, 1, cv2.LINE_AA)
    cv2.line(img, (cx3, cy3 - cl2), (cx3, cy3 + cl2), color, 1, cv2.LINE_AA)

    # ── Label at bottom-right corner of box ──
    icon = "[!]" if is_anomaly else "[+]"
    txt  = f"{icon} {label.upper()}"
    (lw, lh), _ = cv2.getTextSize(txt, font, 0.42, 1)

    # Anchor to bottom-right, clamp to image bounds
    lx = max(0, x2 - lw - 8)
    ly = min(y2 - 4, h - 6)

    # Dark backing rectangle
    cv2.rectangle(img, (lx - 2, ly - lh - 4), (lx + lw + 4, ly + 4),
                  (0, 0, 0), -1)
    cv2.putText(img, txt, (lx, ly), font, 0.42, color, 1, cv2.LINE_AA)

    return img


def draw_starkvis_hud(img: np.ndarray, result: dict) -> np.ndarray:
    """Full STARKVIS HUD overlay — arc reactor aesthetic."""
    h, w   = img.shape[:2]
    font   = cv2.FONT_HERSHEY_SIMPLEX
    ts     = datetime.now().strftime("%Y.%m.%d  //  %H:%M:%S.") + \
             f"{datetime.now().microsecond // 10000:02d}"

    threat = result.get("threat_index", 0)
    n_obj  = result.get("object_count", 0)
    n_anom = result.get("anomaly_count", 0)
    status = result.get("status", "STANDBY")

    t_color = DANGER_RED if threat >= 70 else WARN_ORANGE if threat >= 40 else ARC_BLUE

    # ── Perspective grid ──
    _draw_scan_grid(img)

    # ── Top bar ──
    cv2.rectangle(img, (0, 0), (w, 52), (8, 6, 4), -1)
    cv2.line(img, (0, 52), (w, 52), ARC_BLUE, 1)

    # STARKVIS title
    cv2.putText(img, "STARKVIS", (14, 22), font, 0.65, ARC_BLUE, 2, cv2.LINE_AA)
    cv2.putText(img, "STARK INDUSTRIES  //  THREAT ANALYSIS MODULE",
                (14, 42), font, 0.32, DIM_BLUE, 1, cv2.LINE_AA)

    # Timestamp (center top)
    (tw, _), _ = cv2.getTextSize(ts, font, 0.36, 1)
    cv2.putText(img, ts, (w//2 - tw//2, 22), font, 0.36, ARC_BLUE, 1, cv2.LINE_AA)

    # Status badge (top right)
    s_color = DANGER_RED if "THREAT" in status else HUD_TEAL
    (bw, bh), _ = cv2.getTextSize(f"  {status}  ", font, 0.42, 2)
    bx = w - bw - 14
    cv2.rectangle(img, (bx-4, 8), (w-8, 44), s_color, -1)
    cv2.putText(img, status, (bx, 32), font, 0.42, (0,0,0), 2, cv2.LINE_AA)

    # ── Left panel ──
    panel_h = 160
    cv2.rectangle(img, (0, 52), (220, 52 + panel_h), (10, 8, 5), -1)
    cv2.line(img, (220, 52), (220, 52 + panel_h), ARC_BLUE, 1)

    def hud_line(label, value, y, vc=ARC_BLUE):
        cv2.putText(img, label, (10, y), font, 0.32, DIM_BLUE, 1, cv2.LINE_AA)
        cv2.putText(img, str(value), (120, y), font, 0.38, vc, 1, cv2.LINE_AA)

    hud_line("OBJECTS SCANNED :", n_obj,  80)
    hud_line("ANOMALIES FOUND :", n_anom, 100, DANGER_RED if n_anom else HUD_TEAL)
    hud_line("THREAT INDEX    :", f"{threat}%", 120, t_color)
    hud_line("SCAN MODE       :", "ACTIVE", 140)
    hud_line("SYSTEM          :", "ONLINE", 160, HUD_TEAL)
    hud_line("AI CORE         :", "YOLOv8n", 180)

    # Hex decorations left panel
    for hy in range(60, 52 + panel_h, 20):
        _draw_hex_marker(img, 205, hy, 4, DIM_BLUE)

    # ── Threat bar (right side) ──
    bar_x = w - 28
    bar_top = 60
    bar_bot = h - 60
    bar_len = bar_bot - bar_top
    fill_len = int(threat / 100 * bar_len)

    cv2.rectangle(img, (bar_x - 2, bar_top - 2),
                  (bar_x + 18, bar_bot + 2), (20, 16, 10), -1)
    cv2.rectangle(img, (bar_x - 2, bar_top - 2),
                  (bar_x + 18, bar_bot + 2), ARC_BLUE, 1)

    # Segmented fill
    seg_h = bar_len // 20
    for i in range(20):
        seg_y2 = bar_bot - i * seg_h
        seg_y1 = seg_y2 - seg_h + 2
        if i < int(threat / 5):
            c = DANGER_RED if i > 13 else WARN_ORANGE if i > 7 else ARC_BLUE
            cv2.rectangle(img, (bar_x, seg_y1), (bar_x+14, seg_y2), c, -1)
        else:
            cv2.rectangle(img, (bar_x, seg_y1), (bar_x+14, seg_y2),
                          (30, 24, 14), -1)

    cv2.putText(img, "THREAT", (bar_x - 10, bar_top - 10),
                font, 0.28, ARC_BLUE, 1, cv2.LINE_AA)
    cv2.putText(img, f"{threat}%", (bar_x - 6, bar_bot + 16),
                font, 0.30, t_color, 1, cv2.LINE_AA)

    # ── Arc reactor corner decoration (top-right) ──
    arc_cx, arc_cy = w - 80, 28
    for r_size in [18, 13, 8]:
        alpha = 0.3 if r_size == 18 else 0.5 if r_size == 13 else 0.9
        _draw_arc_circle(img, arc_cx, arc_cy, r_size, ARC_BLUE, 1, alpha)
    cv2.circle(img, (arc_cx, arc_cy), 3, ARC_BLUE, -1)

    # ── Bottom bar ──
    cv2.rectangle(img, (0, h-32), (w, h), (8, 6, 4), -1)
    cv2.line(img, (0, h-32), (w, h-32), ARC_BLUE, 1)

    footer = f"STARK INDUSTRIES  //  VISION OS v3.7  //  {n_obj} OBJECTS  //  {n_anom} ANOMALIES"
    cv2.putText(img, footer, (14, h-12), font, 0.30, DIM_BLUE, 1, cv2.LINE_AA)

    # Corner frame brackets
    bl = 45
    for cx4, cy4, dx, dy in [(0,0,1,1),(w-1,0,-1,1),(0,h-1,1,-1),(w-1,h-1,-1,-1)]:
        cv2.line(img, (cx4, cy4), (cx4+dx*bl, cy4), ARC_BLUE, 2, cv2.LINE_AA)
        cv2.line(img, (cx4, cy4), (cx4, cy4+dy*bl), ARC_BLUE, 2, cv2.LINE_AA)

    return img


def render_output(image_path: str, result: dict, output_path: str) -> str:
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot read: {image_path}")
    h, w = img.shape[:2]

    img = apply_stark_grade(img)

    for obj in result.get("objects", []):
        b = obj.get("bbox_pct", {})
        if not b:
            continue
        x1 = int(b["x"] / 100 * w);  y1 = int(b["y"] / 100 * h)
        x2 = x1 + int(b["w"] / 100 * w); y2 = y1 + int(b["h"] / 100 * h)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w-1, x2), min(h-1, y2)

        is_anom = obj["is_anomaly"]
        color   = SEVERITY_BGR.get(obj["severity"], DANGER_RED) if is_anom else STARK_GOLD
        reason  = obj.get("reason") or ""

        img = draw_starkvis_bbox(img, x1, y1, x2, y2, color,
                               is_anom, obj["name"], obj["id"], reason)

    img = draw_starkvis_hud(img, result)
    cv2.imwrite(output_path, img)
    return output_path
