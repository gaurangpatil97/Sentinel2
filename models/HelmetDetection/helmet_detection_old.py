import cv2
import numpy as np
import os
import easyocr
import pandas as pd
import shutil
import torch
from ultralytics import YOLO
import supervision as sv
from collections import deque
from datetime import datetime
import time

# ─────────────────────────────────────────────────────────────────────────────
# SENTINEL v6  —  optical-flow speed, no reference lines
# ─────────────────────────────────────────────────────────────────────────────
#
#  HOW SPEED WORKS HERE
#  ─────────────────────
#  Standard pixel-displacement fails on front-facing cameras because vehicles
#  move mostly in Z (depth), not X/Y.  Optical flow measures the ACTUAL pixel
#  motion of the vehicle's own feature points — including the radial expansion
#  caused by Z-approach — and converts that to real speed using a perspective
#  scale that varies with the vehicle's Y position in frame (closer = bigger
#  pixels per metre).
#
#  Per-vehicle pipeline every frame:
#    1. Crop the vehicle's bounding box from the previous and current frame.
#    2. Run Lucas-Kanade sparse optical flow on Shi-Tomasi corner points.
#    3. Compute the MAGNITUDE of the median flow vector (px/frame).
#    4. Convert px/frame → m/s using SCALE_AT_Y(bot_y):
#         scale(y) = SCALE_TOP + (SCALE_BOT - SCALE_TOP) * (y / H)
#       SCALE_TOP = metres-per-pixel when vehicle is at top of frame (far).
#       SCALE_BOT = metres-per-pixel when vehicle is at bottom (close).
#    5. Convert m/s → km/h, smooth with a tight EMA.
#
#  CALIBRATION (two numbers only):
#    SCALE_TOP  — at y=0 (far end), one pixel ≈ ? metres.
#                 For a camera 5-6 m high looking 30 m ahead: ~0.15–0.25 m/px
#    SCALE_BOT  — at y=H (near), one pixel ≈ ? metres.
#                 Same camera: ~0.02–0.05 m/px
#    Start with defaults.  Drive past at a known speed, adjust until it matches.
#
# ─────────────────────────────────────────────────────────────────────────────

# ── PATHS ─────────────────────────────────────────────────────────────────────
MODEL_PATH    = r"C:\Users\gaura\Desktop\Sentinel\models\HelmetDetection\best_old.pt"
INPUT_VIDEO   = r"C:\Users\gaura\Desktop\Sentinel\models\HelmetDetection\14934614_2160_3840_30fps.mp4"
OUTPUT_VIDEO  = r"C:\Users\gaura\Desktop\Sentinel\models\HelmetDetection\output_v6.mp4"
VIOLATION_DIR = "violations_v6"

# ── PERSPECTIVE SCALE ─────────────────────────────────────────────────────────
# metres-per-pixel at top of frame (far) and bottom of frame (close).
# These are the ONLY two numbers you need to calibrate.
SCALE_TOP = 0.20    # m/px when vehicle is at y=0  (far, small in frame)
SCALE_BOT = 0.03    # m/px when vehicle is at y=H  (close, large in frame)

# ── OPTICAL FLOW PARAMS ───────────────────────────────────────────────────────
# Shi-Tomasi corner detection
OF_MAX_CORNERS  = 30     # max feature points per vehicle crop
OF_QUALITY      = 0.20   # minimum quality of corners to track
OF_MIN_DIST     = 5      # min pixel distance between corners

# Lucas-Kanade params
LK_WIN    = (15, 15)
LK_LEVELS = 2
LK_CRIT   = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)

# Speed smoothing (EMA alpha — lower = smoother but slower to react)
SPEED_ALPHA = 0.20

# ── VIOLATION THRESHOLDS ──────────────────────────────────────────────────────
SPEED_VIOLATION_KMH = 60
MAX_PLAUSIBLE_KMH   = 150   # discard optical flow readings above this
MIN_FLOW_PX         = 0.3   # ignore sub-pixel flow (stationary / noise)

# ── DETECTION ─────────────────────────────────────────────────────────────────
DETECTION_CONF          = 0.45
TRACK_ACTIVATION_THRESH = 0.40
LOG_COOLDOWN_S          = 5.0

# ── HELMET KEYWORDS ───────────────────────────────────────────────────────────
NO_HELMET_KEYWORDS = {"no-helmet", "no_helmet", "without_helmet",
                      "bare_head", "nohelmet", "without helmet"}
HELMET_KEYWORDS    = {"helmet", "with_helmet", "with helmet"}

# ── LABEL STYLE ───────────────────────────────────────────────────────────────
FONT   = cv2.FONT_HERSHEY_DUPLEX
FONT_L = 0.50
FONT_S = 0.42
THICK  = 1
PAD    = 5

CLR_GREEN  = (34,  197,  94)
CLR_ORANGE = (0,   140, 255)
CLR_RED    = (39,   39, 220)
CLR_WHITE  = (255, 255, 255)
CLR_DARK   = (15,   15,  15)


# ─────────────────────────────────────────────────────────────────────────────
# PERSPECTIVE SCALE  (linear interpolation with Y position)
# ─────────────────────────────────────────────────────────────────────────────

def scale_at_y(y, H):
    """metres-per-pixel at a given y row, linearly interpolated."""
    t = np.clip(y / H, 0.0, 1.0)
    return SCALE_TOP + (SCALE_BOT - SCALE_TOP) * t


# ─────────────────────────────────────────────────────────────────────────────
# OPTICAL FLOW  — sparse LK on vehicle crop
# ─────────────────────────────────────────────────────────────────────────────

def flow_speed_kmh(prev_gray, curr_gray,
                   prev_box, curr_box,
                   bot_y, H, fps):
    """
    Compute vehicle speed in km/h using sparse optical flow.

    prev_gray, curr_gray : full-frame grayscale images
    prev_box, curr_box   : (x1,y1,x2,y2) ints for previous and current frame
    bot_y                : current bottom-centre Y pixel (for scale lookup)
    H                    : frame height
    fps                  : video fps
    Returns km/h or None if flow could not be computed.
    """
    px1, py1, px2, py2 = prev_box

    # guard — crop must be non-trivial
    if px2 <= px1 + 8 or py2 <= py1 + 8:
        return None

    prev_crop = prev_gray[py1:py2, px1:px2]

    # detect corners in previous crop
    corners = cv2.goodFeaturesToTrack(
        prev_crop,
        maxCorners=OF_MAX_CORNERS,
        qualityLevel=OF_QUALITY,
        minDistance=OF_MIN_DIST,
    )
    if corners is None or len(corners) < 3:
        return None

    # shift corner coords to full-frame space
    corners_full = corners + np.array([[[px1, py1]]], dtype=np.float32)

    # LK optical flow
    next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
        prev_gray, curr_gray,
        corners_full, None,
        winSize=LK_WIN,
        maxLevel=LK_LEVELS,
        criteria=LK_CRIT,
    )

    if next_pts is None:
        return None

    good_prev = corners_full[status == 1]
    good_next = next_pts[status == 1]

    if len(good_prev) < 3:
        return None

    # flow vectors (px/frame)
    flow = good_next - good_prev                    # shape (N, 2)
    magnitudes = np.linalg.norm(flow, axis=1)       # px/frame per point

    # median magnitude — robust to outlier points on background
    med_px_per_frame = float(np.median(magnitudes))

    if med_px_per_frame < MIN_FLOW_PX:
        return None

    # convert to real-world speed
    mpp   = scale_at_y(bot_y, H)                   # metres per pixel at this depth
    mps   = med_px_per_frame * mpp * fps            # m/s
    kmh   = mps * 3.6

    return kmh if kmh <= MAX_PLAUSIBLE_KMH else None


# ─────────────────────────────────────────────────────────────────────────────
# DRAWING HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def draw_label(frame, x1, y1, x2, box_bgr, vid, speed_str, helmet_str):
    row1 = f"ID {vid}   {helmet_str}"
    row2 = speed_str
    (w1, h1), _ = cv2.getTextSize(row1, FONT, FONT_L, THICK)
    (w2, h2), _ = cv2.getTextSize(row2, FONT, FONT_S, THICK)

    pill_w = max(w1, w2) + PAD * 2
    pill_h = h1 + h2 + PAD * 3

    rx1 = x1
    ry2 = y1
    rx2 = rx1 + pill_w
    ry1 = max(0, ry2 - pill_h)

    cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), box_bgr, -1)
    border = tuple(max(0, c - 55) for c in box_bgr)
    cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), border, 1)
    cv2.putText(frame, row1, (rx1+PAD, ry1+PAD+h1),
                FONT, FONT_L, CLR_WHITE, THICK, cv2.LINE_AA)
    dy = ry1 + PAD + h1 + PAD//2
    cv2.line(frame, (rx1+PAD, dy), (rx2-PAD, dy), (200,200,200), 1)
    cv2.putText(frame, row2, (rx1+PAD, dy+PAD+h2),
                FONT, FONT_S, CLR_WHITE, THICK, cv2.LINE_AA)


def draw_osd(frame, cs, ch, cb, total):
    panel = frame.copy()
    cv2.rectangle(panel, (8,8), (318,172), CLR_DARK, -1)
    cv2.addWeighted(panel, 0.50, frame, 0.50, 0, frame)
    rows = [
        ("SPEED VIOLATIONS",  f"{cs+cb}", CLR_RED),
        ("HELMET VIOLATIONS", f"{ch+cb}", CLR_ORANGE),
        ("BOTH",              f"{cb}",    (170,170,170)),
        ("TOTAL ENFORCED",    f"{total}", CLR_WHITE),
    ]
    for idx, (lbl, val, col) in enumerate(rows):
        y = 40 + idx * 32
        cv2.putText(frame, lbl, (16, y),  FONT, 0.50, col, 1, cv2.LINE_AA)
        cv2.putText(frame, val, (290, y), FONT, 0.50, col, 1, cv2.LINE_AA)


# ─────────────────────────────────────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def resolve_class_ids(model):
    h_ids, nh_ids = set(), set()
    for cid, cname in model.names.items():
        lo = cname.lower().strip()
        if lo in HELMET_KEYWORDS:        h_ids.add(cid)
        elif lo in NO_HELMET_KEYWORDS:   nh_ids.add(cid)
    print(f"[Sentinel] helmet_ids={h_ids}  no_helmet_ids={nh_ids}")
    if not h_ids and not nh_ids:
        print("  WARNING: no helmet classes matched")
    return h_ids, nh_ids


def preprocess_ocr(crop):
    h, w  = crop.shape[:2]
    up    = cv2.resize(crop, (w*3, h*3), interpolation=cv2.INTER_CUBIC)
    gray  = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    gray  = clahe.apply(gray)
    blur  = cv2.GaussianBlur(gray, (0,0), 3)
    return cv2.addWeighted(gray, 1.8, blur, -0.8, 0)

def read_plate(crop, reader):
    if crop is None or crop.size == 0: return "NO_CROP"
    res = reader.readtext(preprocess_ocr(crop), detail=1, paragraph=False)
    if not res: return "NOT_READABLE"
    return max(res, key=lambda r: r[2])[1].upper().replace(" ","") or "NOT_READABLE"

def img_sharpness(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim==3 else img
    return float(cv2.Laplacian(g, cv2.CV_64F).var())

def clamp_box(x1, y1, x2, y2, W, H):
    return (max(0,x1), max(0,y1), min(W,x2), min(H,y2))


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def run_enforcement():
    os.makedirs(VIOLATION_DIR, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    reader = easyocr.Reader(["en"], gpu=(device=="cuda"))
    model  = YOLO(MODEL_PATH)
    HELMET_IDS, NO_HELMET_IDS = resolve_class_ids(model)

    byte_track = sv.ByteTrack(track_activation_threshold=TRACK_ACTIVATION_THRESH)

    cap    = cv2.VideoCapture(INPUT_VIDEO)
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(
        OUTPUT_VIDEO, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    print(f"[Sentinel] FPS={fps:.1f}  Frame={W}x{H}")
    print(f"[Sentinel] Scale: {SCALE_TOP} m/px (top) → {SCALE_BOT} m/px (bottom)")

    # previous-frame grayscale and bounding boxes (needed for optical flow)
    prev_gray = None
    prev_boxes = {}   # tid → (x1,y1,x2,y2) from last frame

    # per-vehicle state
    smooth_spd   = {}   # tid → smoothed km/h
    helmet_votes = {}
    best_crop    = {}
    last_log_t   = {}

    violation_records = []
    logged_ids = set()
    cs = ch = cb = 0

    cv2.namedWindow("Sentinel v6", cv2.WINDOW_NORMAL)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        curr_gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        results    = model.predict(frame, conf=DETECTION_CONF, verbose=False)[0]
        detections = byte_track.update_with_detections(
            sv.Detections.from_ultralytics(results))

        out = frame.copy()
        now = time.time()

        if len(detections) == 0:
            draw_osd(out, cs, ch, cb, len(logged_ids))
            prev_gray = curr_gray.copy()
            prev_boxes.clear()
            cv2.imshow("Sentinel v6", out)
            writer.write(out)
            if cv2.waitKey(1) & 0xFF == ord("q"): break
            continue

        bot_centres = detections.get_anchors_coordinates(
            anchor=sv.Position.BOTTOM_CENTER)

        curr_boxes = {}   # fill this frame's boxes

        for i, tid in enumerate(detections.tracker_id):
            if tid is None: continue

            # ── init ──────────────────────────────────────────────────────────
            if tid not in smooth_spd:
                smooth_spd[tid]   = 0.0
                helmet_votes[tid] = deque(maxlen=20)
                best_crop[tid]    = (0.0, None)
                last_log_t[tid]   = 0.0

            x1, y1, x2, y2 = detections.xyxy[i].astype(int)
            box = clamp_box(x1, y1, x2, y2, W, H)
            curr_boxes[tid] = box
            bot_y = int(bot_centres[i][1])

            # ── OPTICAL FLOW SPEED ────────────────────────────────────────────
            if prev_gray is not None and tid in prev_boxes:
                raw_kmh = flow_speed_kmh(
                    prev_gray, curr_gray,
                    prev_boxes[tid], box,
                    bot_y, H, fps
                )
                if raw_kmh is not None:
                    # EMA smoothing
                    smooth_spd[tid] = (SPEED_ALPHA * raw_kmh
                                       + (1 - SPEED_ALPHA) * smooth_spd[tid])

            display_spd = int(smooth_spd[tid])
            is_speeding = display_spd > SPEED_VIOLATION_KMH

            # ── HELMET ────────────────────────────────────────────────────────
            cid = int(detections.class_id[i]) \
                  if detections.class_id is not None else -1
            if cid in HELMET_IDS:       helmet_votes[tid].append(1)
            elif cid in NO_HELMET_IDS:  helmet_votes[tid].append(0)

            helmet_worn = (
                sum(helmet_votes[tid]) / len(helmet_votes[tid]) >= 0.5
                if helmet_votes[tid] else True)
            is_nohelmet = not helmet_worn

            # ── box colour ────────────────────────────────────────────────────
            if is_speeding:   box_bgr = CLR_RED
            elif is_nohelmet: box_bgr = CLR_ORANGE
            else:             box_bgr = CLR_GREEN

            # ── best crop ─────────────────────────────────────────────────────
            crop = frame[box[1]:box[3], box[0]:box[2]]
            if crop.size > 0:
                sh = img_sharpness(crop)
                if sh > best_crop[tid][0]:
                    best_crop[tid] = (sh, crop.copy())

            # ── VIOLATION LOG ─────────────────────────────────────────────────
            cooldown_ok = (now - last_log_t[tid]) > LOG_COOLDOWN_S
            if (is_speeding or is_nohelmet) and cooldown_ok \
                    and tid not in logged_ids:
                last_log_t[tid] = now
                logged_ids.add(tid)

                if is_speeding and is_nohelmet: vtype="SPEEDING+NO_HELMET"; cb+=1
                elif is_speeding:               vtype="SPEEDING";           cs+=1
                else:                           vtype="NO_HELMET";          ch+=1

                save_img = best_crop[tid][1]
                plate    = read_plate(save_img, reader)
                ts       = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                fname    = f"ID{tid}_{vtype}_{plate}_{display_spd}kmh.jpg"
                if save_img is not None and save_img.size > 0:
                    cv2.imwrite(os.path.join(VIOLATION_DIR, fname), save_img)
                violation_records.append({
                    "Timestamp": ts, "Vehicle_ID": tid,
                    "Violation_Type": vtype, "Speed_KMH": display_spd,
                    "Helmet_Worn": helmet_worn, "License_Plate": plate,
                    "Snapshot": fname,
                    "Sharpness": round(best_crop[tid][0], 2),
                })

            # ── DRAW ──────────────────────────────────────────────────────────
            cv2.rectangle(out, (x1, y1), (x2, y2), box_bgr, 2)

            helmet_str = "HELMET" if helmet_worn else "NO HELMET"
            prefix     = "!  " if is_speeding else "   "
            spd_str    = f"{prefix}{display_spd} km/h"

            draw_label(out, x1, y1, x2, box_bgr, tid, spd_str, helmet_str)

        # ── end of frame: store for next iteration ────────────────────────────
        prev_gray  = curr_gray.copy()
        prev_boxes = curr_boxes

        draw_osd(out, cs, ch, cb, len(logged_ids))
        cv2.imshow("Sentinel v6", out)
        writer.write(out)
        if cv2.waitKey(1) & 0xFF == ord("q"): break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    if violation_records:
        df = pd.DataFrame(violation_records)
        df.to_csv("violations_report_v6.csv", index=False)
        shutil.make_archive("enforcement_evidence_v6", "zip", VIOLATION_DIR)
        print("\n" + "="*52)
        print("  SENTINEL v6  —  FINAL REPORT")
        print("="*52)
        print(f"  Speed      : {cs}")
        print(f"  No helmet  : {ch}")
        print(f"  Both       : {cb}")
        print(f"  Total      : {len(violation_records)}")
        print("="*52)
        print(df[["Timestamp","Vehicle_ID","Violation_Type",
                   "Speed_KMH","Helmet_Worn","License_Plate"]].to_string())
    else:
        print("No violations detected.")
    print(f"\nOutput → {OUTPUT_VIDEO}")


if __name__ == "__main__":
    run_enforcement()