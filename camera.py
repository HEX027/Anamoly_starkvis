#!/usr/bin/env python3
"""
camera.py — STARKVIS Vision System
Jetson USB Camera — smooth live feed, inference in background thread
"""

import sys
import cv2
import time
import argparse
import threading
from pathlib import Path
from datetime import datetime

from detector import (
    run_detection, render_output,
    draw_starkvis_hud,
    ARC_BLUE, DANGER_RED, WARN_ORANGE, HUD_TEAL,
    DIM_BLUE, STARK_GOLD, SEVERITY_BGR,
)

# ── ANSI colors ───────────────────────────────────────────────────────────────
R  = "\033[0m"
Y  = "\033[93m"
RD = "\033[91m"
GR = "\033[92m"
CY = "\033[96m"
DM = "\033[2;37m"
BD = "\033[1m"
LN = "\033[4m"

BANNER = f"""{Y}{BD}
     ██╗ █████╗ ██████╗ ██╗   ██╗██╗███████╗
     ██║██╔══██╗██╔══██╗██║   ██║██║██╔════╝
     ██║███████║██████╔╝██║   ██║██║███████╗
██   ██║██╔══██║██╔══██╗╚██╗ ██╔╝██║╚════██║
╚█████╔╝██║  ██║██║  ██║ ╚████╔╝ ██║███████║
 ╚════╝ ╚═╝  ╚═╝╚═╝  ╚═╝  ╚═══╝  ╚═╝╚══════╝
{R}{CY}  STARK INDUSTRIES  //  STARKVIS v3.7  //  YOLOv8{R}
"""

def print_banner():
    print(BANNER)

def hr(c="─", n=60, col=Y):
    print(f"{col}{c*n}{R}")

def print_results(result, elapsed, path):
    objects   = result.get("objects", [])
    anomalies = result.get("anomalies", [])
    normals   = [o for o in objects if not o["is_anomaly"]]
    threat    = result.get("threat_index", 0)
    status    = result.get("status", "—")
    avg_conf  = round(sum(o["confidence"] for o in objects) / max(len(objects),1))
    tc        = RD if threat>=70 else Y if threat>=40 else GR

    print(); hr("═")
    print(f"{Y}{BD}  STARKVIS  //  SCAN COMPLETE{R}")
    hr("═")
    print(f"\n  {BD}{CY}{len(objects)}{R} objects  "
          f"{BD}{RD if anomalies else GR}{len(anomalies)}{R} anomalies  "
          f"{BD}{tc}{threat}%{R} threat  "
          f"{BD}{GR}{avg_conf}%{R} conf  "
          f"{BD}{CY}{elapsed:.2f}s{R}\n")
    sc = RD if "THREAT" in status else GR
    print(f"  STATUS: {BD}{sc}{status}{R}\n")

    if anomalies:
        hr("─", col=RD)
        print(f"  {RD}{BD}⚠ {len(anomalies)} ANOMALY DETECTED{R}")
        hr("─", col=RD)
        for o in anomalies:
            sev = o["severity"].upper()
            sc2 = RD if sev in ("CRITICAL","HIGH") else Y
            print(f"  {CY}{o['id']}{R}  {BD}{o['name'].upper()}{R}  "
                  f"{sc2}{sev}{R}  {GR}{o['confidence']}%{R}  "
                  f"{DM}{o.get('reason','')}{R}")
    else:
        print(f"  {GR}{BD}✓ ALL CLEAR{R}")

    if normals:
        print(f"\n  {DM}Non-hostile: " +
              "  ".join(f"{o['name'].upper()}({o['confidence']}%)" for o in normals) + R)

    print(); hr()
    print(f"  {Y}◈ SAVED:{R} {LN}{path}{R}")
    hr(); print()


# ── Camera ────────────────────────────────────────────────────────────────────

def list_cameras(n=6):
    found = []
    for i in range(n):
        c = cv2.VideoCapture(i)
        if c.isOpened():
            found.append(i)
            c.release()
    return found


def open_camera(index, width=640, height=480):
    cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
    if not cap.isOpened():
        cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open /dev/video{index}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    # warm up
    for _ in range(5): cap.read()
    aw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    ah = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"  {Y}◈ CAMERA:{R}  /dev/video{index}  {aw}×{ah}")
    return cap


# ── Frame reader thread — keeps buffer fresh always ──────────────────────────

class FrameReader:
    """Reads camera in a tight loop so the buffer never goes stale."""
    def __init__(self, cap):
        self.cap   = cap
        self.frame = None
        self.lock  = threading.Lock()
        self._stop = False
        t = threading.Thread(target=self._run, daemon=True)
        t.start()

    def _run(self):
        while not self._stop:
            ret, frame = self.cap.read()
            if ret and frame is not None:
                with self.lock:
                    self.frame = frame
            else:
                time.sleep(0.01)

    def get(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self):
        self._stop = True


# ── Lightweight live overlay (drawn every frame) ──────────────────────────────

def draw_live_overlay(frame, scanning, scan_count, interval,
                      last_scan_t, last_result):
    h, w  = frame.shape[:2]
    font  = cv2.FONT_HERSHEY_SIMPLEX

    # Top bar
    cv2.rectangle(frame, (0,0), (w,44), (8,6,4), -1)
    cv2.line(frame, (0,44), (w,44), ARC_BLUE, 1)
    cv2.putText(frame, "STARKVIS  VISION OS",
                (12,20), font, 0.50, ARC_BLUE, 2, cv2.LINE_AA)
    ts = datetime.now().strftime("%H:%M:%S")
    cv2.putText(frame, f"STARK INDUSTRIES  //  {ts}",
                (12,36), font, 0.30, DIM_BLUE, 1, cv2.LINE_AA)

    # Status badge top-right
    status = "SCANNING..." if scanning else "STANDBY"
    s_col  = DANGER_RED if scanning else HUD_TEAL
    (bw, bh), _ = cv2.getTextSize(f" {status} ", font, 0.40, 2)
    cv2.rectangle(frame, (w-bw-16,8), (w-8,38), s_col, -1)
    cv2.putText(frame, status, (w-bw-10,30), font, 0.40, (0,0,0), 2, cv2.LINE_AA)

    # Left info
    cv2.rectangle(frame, (0,44), (190,155), (10,8,5), -1)
    cv2.line(frame, (190,44), (190,155), ARC_BLUE, 1)

    def info(lbl, val, y, vc=ARC_BLUE):
        cv2.putText(frame, lbl, (8,y), font, 0.28, DIM_BLUE, 1, cv2.LINE_AA)
        cv2.putText(frame, str(val), (108,y), font, 0.33, vc, 1, cv2.LINE_AA)

    info("SCANS :", scan_count, 64)
    info("MODE  :", "AUTO" if interval>0 else "MANUAL", 82)
    if interval > 0 and not scanning:
        rem = max(0, interval - int(time.time() - last_scan_t))
        info("NEXT  :", f"{rem}s", 100, WARN_ORANGE)
    if last_result:
        threat = last_result.get("threat_index", 0)
        n_anom = last_result.get("anomaly_count", 0)
        tc     = DANGER_RED if threat>=70 else WARN_ORANGE if threat>=40 else ARC_BLUE
        info("THREAT:", f"{threat}%", 118, tc)
        info("ANOMAL:", n_anom, 136, DANGER_RED if n_anom else HUD_TEAL)

    # Last scan bboxes on live feed
    if last_result:
        for obj in last_result.get("objects", []):
            b = obj.get("bbox_pct", {})
            if not b: continue
            x1=int(b["x"]/100*w); y1=int(b["y"]/100*h)
            x2=x1+int(b["w"]/100*w); y2=y1+int(b["h"]/100*h)
            x1,y1=max(0,x1),max(0,y1)
            x2,y2=min(w-1,x2),min(h-1,y2)
            is_a  = obj["is_anomaly"]
            color = SEVERITY_BGR.get(obj["severity"], DANGER_RED) if is_a else STARK_GOLD
            cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
            lbl = f"{'[!]' if is_a else '[+]'} {obj['name'].upper()}"
            (lw,lh),_ = cv2.getTextSize(lbl, font, 0.38, 1)
            lx = max(0, x2-lw-6)
            ly = min(y2-4, h-6)
            cv2.rectangle(frame,(lx-2,ly-lh-3),(lx+lw+4,ly+4),(0,0,0),-1)
            cv2.putText(frame,lbl,(lx,ly),font,0.38,color,1,cv2.LINE_AA)

    # Corner brackets
    bl = 35
    for cx,cy,dx,dy in [(0,0,1,1),(w-1,0,-1,1),(0,h-1,1,-1),(w-1,h-1,-1,-1)]:
        cv2.line(frame,(cx,cy),(cx+dx*bl,cy),ARC_BLUE,2,cv2.LINE_AA)
        cv2.line(frame,(cx,cy),(cx,cy+dy*bl),ARC_BLUE,2,cv2.LINE_AA)

    # Bottom help
    cv2.rectangle(frame,(0,h-24),(w,h),(8,6,4),-1)
    cv2.line(frame,(0,h-24),(w,h-24),ARC_BLUE,1)
    cv2.putText(frame,"SPACE: scan    S: save    Q: quit",
                (10,h-8),font,0.32,DIM_BLUE,1,cv2.LINE_AA)

    return frame


# ── Main live loop ────────────────────────────────────────────────────────────

class StarkvisVision:
    def __init__(self, cap, output_dir, weights, interval):
        self.reader     = FrameReader(cap)
        self.cap        = cap
        self.output_dir = Path(output_dir)
        self.weights    = weights
        self.interval   = interval
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._last_result     = None
        self._last_ann        = None
        self._scanning        = False
        self._lock            = threading.Lock()
        self._last_scan_t     = 0.0
        self._scan_count      = 0
        self._show_ann        = False
        self._ann_shown_at    = 0.0

    def _scan_worker(self, frame):
        # Runs in background — never blocks camera loop
        with self._lock:
            if self._scanning: return
            self._scanning = True
        try:
            ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
            raw_path = str(self.output_dir / f"raw_{ts}.jpg")
            ann_path = str(self.output_dir / f"starkvis_{ts}.jpg")

            cv2.imwrite(raw_path, frame)
            print(f"\n  {Y}◈ SCANNING…{R}  [{ts}]")

            t0      = time.time()
            result  = run_detection(raw_path, weights=self.weights)
            render_output(raw_path, result, ann_path)
            elapsed = time.time() - t0

            ann = cv2.imread(ann_path)
            if ann is not None:
                self._last_ann     = ann

            self._last_result  = result
            self._scan_count  += 1
            self._last_scan_t  = time.time()
            self._show_ann     = True
            self._ann_shown_at = time.time()

            print_results(result, elapsed, ann_path)
            Path(raw_path).unlink(missing_ok=True)

        except Exception as ex:
            print(f"  {RD}◈ ERROR:{R} {ex}")
        finally:
            self._scanning = False

    def _trigger(self, frame):
        threading.Thread(target=self._scan_worker,
                         args=(frame.copy(),), daemon=True).start()

    def run(self):
        print(f"\n  {Y}{BD}◈ STARKVIS LIVE — FEED ACTIVE{R}")
        print(f"  {DM}SPACE → scan  │  S → save  │  Q → quit{R}")
        print(f"  {DM}Auto-scan: {'every '+str(self.interval)+'s' if self.interval else 'OFF'}{R}\n")

        win = "STARKVIS — STARK INDUSTRIES"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, 800, 600)

        try:
            while True:
                # Get latest frame from reader thread — NEVER blocks
                frame = self.reader.get()
                if frame is None:
                    time.sleep(0.01)
                    continue

                # Auto-scan trigger
                if (self.interval > 0 and not self._scanning
                        and time.time() - self._last_scan_t >= self.interval):
                    self._trigger(frame)

                # Display: show annotated result for 4s, then live feed
                if (self._show_ann and self._last_ann is not None
                        and time.time() - self._ann_shown_at < 4.0):
                    fh, fw = frame.shape[:2]
                    display = cv2.resize(self._last_ann, (fw, fh))
                else:
                    self._show_ann = False
                    display = frame.copy()
                    display = draw_live_overlay(
                        display, self._scanning, self._scan_count,
                        self.interval, self._last_scan_t, self._last_result)

                cv2.imshow(win, display)
                key = cv2.waitKey(1) & 0xFF

                if key in (ord('q'), 27):
                    break
                elif key == ord(' '):
                    if not self._scanning:
                        self._trigger(frame)
                    else:
                        print(f"  {DM}Scan in progress…{R}")
                elif key == ord('s'):
                    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
                    out = self.output_dir / f"frame_{ts}.jpg"
                    cv2.imwrite(str(out), frame)
                    print(f"  {Y}◈ SAVED:{R} {out}")

        finally:
            self.reader.stop()
            cv2.destroyAllWindows()
            self.cap.release()
            print(f"\n  {DM}[STARKVIS] Offline.{R}\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="STARKVIS — Jetson Camera")
    parser.add_argument("mode",       nargs="?", default="live",
                        choices=["live","capture"])
    parser.add_argument("--camera",   "-c", type=int, default=0)
    parser.add_argument("--interval", "-i", type=int, default=10)
    parser.add_argument("--output",   "-o")
    parser.add_argument("--outdir",   "-d", default="starkvis_output")
    parser.add_argument("--weights",  "-w", default="yolov8n.pt")
    parser.add_argument("--width",          type=int, default=640)
    parser.add_argument("--height",         type=int, default=480)
    parser.add_argument("--headless",       action="store_true")
    parser.add_argument("--list",           action="store_true")
    parser.add_argument("--no-banner",      action="store_true")
    args = parser.parse_args()

    if not args.no_banner:
        print_banner()

    if args.list:
        print(f"  {Y}◈ PROBING CAMERAS…{R}")
        for i in list_cameras():
            print(f"  {Y}/dev/video{i}{R}  ← available")
        return

    cap = open_camera(args.camera, args.width, args.height)

    if args.headless:
        reader  = FrameReader(cap)
        out_dir = Path(args.outdir)
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"  {Y}◈ HEADLESS — every {args.interval}s{R}")
        while True:
            frame = reader.get()
            if frame is None:
                time.sleep(0.1); continue
            ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
            raw_path = str(out_dir / f"raw_{ts}.jpg")
            ann_path = str(out_dir / f"starkvis_{ts}.jpg")
            cv2.imwrite(raw_path, frame)
            t0     = time.time()
            result = run_detection(raw_path, weights=args.weights)
            render_output(raw_path, result, ann_path)
            Path(raw_path).unlink(missing_ok=True)
            print_results(result, time.time()-t0, ann_path)
            time.sleep(args.interval)
    else:
        StarkvisVision(cap, args.outdir, args.weights, args.interval).run()


if __name__ == "__main__":
    main()
