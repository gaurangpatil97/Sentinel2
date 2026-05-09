import cv2
import numpy as np
import os
import pandas as pd
import shutil
import torch
from ultralytics import YOLO
import supervision as sv
from collections import deque, defaultdict
from datetime import datetime
import sys

# ─────────────────────────────────────────────
# 1. INITIALIZATION
# ─────────────────────────────────────────────
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"[Sentinel] Running on: {device.upper()}")

VIOLATION_DIR = 'violations'
os.makedirs(VIOLATION_DIR, exist_ok=True)
violation_records = []
logged_ids = set()

# ─────────────────────────────────────────────
# 2. CONFIGURATION
# ─────────────────────────────────────────────
# Real-world dimensions for each zone (metres)
# Tune these to match the actual road section you draw
AWAY_WIDTH_M     = 12
AWAY_LENGTH_M    = 80
TOWARDS_WIDTH_M  = 12
TOWARDS_LENGTH_M = 80

SMOOTHING_ALPHA     = 0.15   # EMA speed smoothing
VIOLATION_THRESHOLD = 80     # Speed limit km/h
MIN_TRACK_SECONDS   = 1.5    # Minimum tracking time before violation logged
CONF_THRESHOLD      = 0.3    # YOLO confidence threshold
IOU_THRESHOLD       = 0.7    # YOLO IOU threshold

VEHICLE_CLASSES = [2, 3, 5, 7]  # car, motorbike, bus, truck

def make_target(width, length):
    return np.array(
        [[0, 0], [width - 1, 0], [width - 1, length - 1], [0, length - 1]],
        dtype=np.float32
    )


# ─────────────────────────────────────────────
# 3. PERSPECTIVE TRANSFORM
# ─────────────────────────────────────────────
class ViewTransformer:
    def __init__(self, source: np.ndarray, target: np.ndarray):
        self.m = cv2.getPerspectiveTransform(
            source.astype(np.float32),
            target.astype(np.float32)
        )

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        if points.size == 0:
            return points
        return cv2.perspectiveTransform(
            points.reshape(-1, 1, 2).astype(np.float32), self.m
        ).reshape(-1, 2)


# ─────────────────────────────────────────────
# 4. INTERACTIVE POLYGON SELECTOR
# ─────────────────────────────────────────────
poly_points = []

def mouse_callback(event, x, y, flags, param):
    global poly_points
    if event == cv2.EVENT_LBUTTONDOWN and len(poly_points) < 4:
        poly_points.append((x, y))

def select_polygon(frame, zone_label, color):
    global poly_points
    poly_points = []

    clone  = frame.copy()
    window = f'Sentinel - Zone {zone_label}'

    print(f"\n[Sentinel] Define {zone_label} zone:")
    print("  Click 4 corners: top-left > top-right > bottom-right > bottom-left")
    print("  ENTER to confirm | R to reset\n")

    instructions = [
        f"Zone: {zone_label} (vehicles going {zone_label.lower()} camera)",
        "top-left > top-right > bottom-right > bottom-left",
        "ENTER to confirm  |  R to reset",
    ]

    cv2.imshow(window, clone)
    cv2.waitKey(1)
    cv2.setMouseCallback(window, mouse_callback)

    while True:
        display = clone.copy()

        for i, line in enumerate(instructions):
            cv2.putText(display, line, (20, 30 + i * 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

        for i, pt in enumerate(poly_points):
            cv2.circle(display, pt, 7, color, -1)
            cv2.putText(display, str(i + 1), (pt[0] + 10, pt[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        if len(poly_points) > 1:
            for i in range(len(poly_points) - 1):
                cv2.line(display, poly_points[i], poly_points[i + 1], color, 2)
        if len(poly_points) == 4:
            cv2.line(display, poly_points[3], poly_points[0], color, 2)
            cv2.putText(display, "Press ENTER to confirm",
                        (20, display.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow(window, display)
        key = cv2.waitKey(1) & 0xFF

        if key == 13 and len(poly_points) == 4:
            break
        elif key == ord('r'):
            poly_points = []
            print(f"[Sentinel] {zone_label} points reset.")

    cv2.destroyWindow(window)
    result = np.array(poly_points, dtype=np.float32)
    print(f"[Sentinel] {zone_label} zone locked: {poly_points}\n")
    return result


# ─────────────────────────────────────────────
# 5. HUD + REPORT
# ─────────────────────────────────────────────
def _draw_hud(frame, tracked_count, fps_actual=0):
    cv2.putText(frame, f"Violations: {len(logged_ids)}",
                (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    cv2.putText(frame, f"Tracked: {tracked_count}",
                (30, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Limit: {VIOLATION_THRESHOLD} km/h",
                (30, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    if fps_actual > 0:
        cv2.putText(frame, f"FPS: {fps_actual:.1f}",
                    (30, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 2)

def _generate_report():
    print(f"\n{'='*40}")
    if violation_records:
        df = pd.DataFrame(violation_records)
        df.to_csv('violations_report.csv', index=False)
        shutil.make_archive('enforcement_evidence', 'zip', VIOLATION_DIR)
        print("SENTINEL ENFORCEMENT REPORT")
        print(f"Total Violations : {len(violation_records)}")
        print(f"Report           : violations_report.csv")
        print(f"Evidence archive : enforcement_evidence.zip")
        print('='*40)
        print(df.to_string(index=False))
    else:
        print("No violations detected in this session.")
    print('='*40)


# ─────────────────────────────────────────────
# 6. MAIN ENFORCEMENT LOOP
# ─────────────────────────────────────────────
def run_enforcement(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        return

    video_info = sv.VideoInfo.from_video_path(video_path)
    fps        = video_info.fps or 30
    print(f"[Sentinel] FPS: {fps:.1f} | Resolution: {video_info.resolution_wh} | Limit: {VIOLATION_THRESHOLD} km/h")

    # Auto-scale annotation thickness and text to video resolution
    thickness  = sv.calculate_optimal_line_thickness(resolution_wh=video_info.resolution_wh)
    text_scale = sv.calculate_optimal_text_scale(resolution_wh=video_info.resolution_wh)

    ret, first_frame = cap.read()
    if not ret:
        print("[ERROR] Could not read first frame.")
        cap.release()
        return

    # Draw both zones
    poly_away    = select_polygon(first_frame, "AWAY",    color=(0, 255, 255))
    poly_towards = select_polygon(first_frame, "TOWARDS", color=(255, 0, 255))

    vt_away    = ViewTransformer(poly_away,    make_target(AWAY_WIDTH_M,    AWAY_LENGTH_M))
    vt_towards = ViewTransformer(poly_towards, make_target(TOWARDS_WIDTH_M, TOWARDS_LENGTH_M))

    # Supervision zone triggers — cleaner than manual pointPolygonTest
    zone_away    = sv.PolygonZone(polygon=poly_away.astype(int))
    zone_towards = sv.PolygonZone(polygon=poly_towards.astype(int))

    # Annotators
    box_annotator   = sv.BoxAnnotator(thickness=thickness)
    label_annotator = sv.LabelAnnotator(
        text_scale=text_scale,
        text_thickness=thickness,
        text_position=sv.Position.BOTTOM_CENTER
    )
    trace_annotator = sv.TraceAnnotator(
        thickness=thickness,
        trace_length=int(fps * 2),
        position=sv.Position.BOTTOM_CENTER
    )

    # Model + tracker
    model      = YOLO("yolov8n.pt")
    byte_track = sv.ByteTrack(
        frame_rate=int(fps),
        track_activation_threshold=CONF_THRESHOLD
    )

    # Per-vehicle state
    coordinates  = defaultdict(lambda: deque(maxlen=int(fps)))
    zone_history = {}
    speed_store  = {}

    # Output video path
    base        = os.path.splitext(os.path.basename(video_path))[0]
    output_path = f"{base}_sentinel_output.mp4"

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frame_count  = 0
    t_prev       = cv2.getTickCount()
    display_fps  = 0.0

    cv2.namedWindow('Sentinel - Speed Enforcement', cv2.WINDOW_NORMAL)

    with sv.VideoSink(output_path, video_info) as sink:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # ── YOLO + ByteTrack ──
            result     = model(frame, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD,
                               verbose=False, classes=VEHICLE_CLASSES)[0]
            detections = sv.Detections.from_ultralytics(result)
            detections = byte_track.update_with_detections(detections)

            # ── Zone filtering — keep only vehicles inside either zone ──
            in_away    = zone_away.trigger(detections)
            in_towards = zone_towards.trigger(detections)
            in_either  = in_away | in_towards
            detections = detections[in_either]

            # Update zone_history for each tracked vehicle
            for i, tid in enumerate(detections.tracker_id):
                if tid not in zone_history:
                    if in_away[in_either][i]:
                        zone_history[tid] = 'away'
                    else:
                        zone_history[tid] = 'towards'

            # ── Speed estimation ──
            bottom_centers = detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)

            for tid, bc in zip(detections.tracker_id, bottom_centers):
                zone = zone_history.get(tid, 'away')
                if zone == 'away':
                    transformed = vt_away.transform_points(bc.reshape(1, 2)).astype(int)
                else:
                    transformed = vt_towards.transform_points(bc.reshape(1, 2)).astype(int)

                # Store Y coordinate only (stable for linear road motion)
                coordinates[tid].append(transformed[0][1])

            # ── Build labels + detect violations ──
            labels = []
            colors = []

            for i, tid in enumerate(detections.tracker_id):
                if len(coordinates[tid]) < fps / 2:
                    speed_store[tid] = 0
                    labels.append(f"#{tid}")
                    colors.append((200, 200, 200))
                    continue

                y_start = coordinates[tid][-1]
                y_end   = coordinates[tid][0]
                dist    = abs(y_start - y_end)
                time    = len(coordinates[tid]) / fps
                speed   = dist / time * 3.6

                # EMA smoothing
                if tid not in speed_store:
                    speed_store[tid] = speed
                else:
                    speed_store[tid] = SMOOTHING_ALPHA * speed + (1 - SMOOTHING_ALPHA) * speed_store[tid]

                current_speed = int(speed_store[tid])
                zone          = zone_history.get(tid, 'away')
                is_speeding   = current_speed > VIOLATION_THRESHOLD

                print(f"[TRACK] ID:{tid} | {zone.upper()} | {current_speed} km/h")

                # Violation logging
                track_secs = len(coordinates[tid]) / fps
                if is_speeding and tid not in logged_ids and track_secs >= MIN_TRACK_SECONDS:
                    x1, y1, x2, y2 = detections.xyxy[i].astype(int)
                    crop = frame[
                        max(0, y1):min(frame.shape[0], y2),
                        max(0, x1):min(frame.shape[1], x2)
                    ]
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    direction = zone.upper()
                    file_name = f"ID_{tid}_{direction}_{current_speed}kmh_f{frame_count}.jpg"
                    cv2.imwrite(os.path.join(VIOLATION_DIR, file_name), crop)

                    violation_records.append({
                        "Timestamp":     timestamp,
                        "Vehicle_ID":    tid,
                        "Direction":     direction,
                        "Speed_KMH":     current_speed,
                        "Snapshot_File": file_name
                    })
                    logged_ids.add(tid)
                    print(f"[VIOLATION] ID:{tid} | {direction} | {current_speed} km/h | {file_name}")

                labels.append(f"#{tid} {current_speed} km/h {'[!]' if is_speeding else ''}")
                colors.append((0, 0, 255) if is_speeding else (0, 255, 0))

            # ── Annotate frame ──
            annotated = frame.copy()

            # Draw zone polygons
            cv2.polylines(annotated, [poly_away.astype(np.int32)],    True, (0, 255, 255), thickness)
            cv2.polylines(annotated, [poly_towards.astype(np.int32)], True, (255, 0, 255), thickness)
            cv2.putText(annotated, "AWAY",    tuple(poly_away[0].astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, text_scale, (0, 255, 255), thickness)
            cv2.putText(annotated, "TOWARDS", tuple(poly_towards[0].astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, text_scale, (255, 0, 255), thickness)

            if len(detections) > 0:
                annotated = trace_annotator.annotate(scene=annotated, detections=detections)
                annotated = box_annotator.annotate(scene=annotated, detections=detections)
                annotated = label_annotator.annotate(scene=annotated, detections=detections, labels=labels)

            # Compute live FPS
            t_now       = cv2.getTickCount()
            display_fps = cv2.getTickFrequency() / (t_now - t_prev)
            t_prev      = t_now

            _draw_hud(annotated, len(detections), display_fps)

            sink.write_frame(annotated)
            cv2.imshow('Sentinel - Speed Enforcement', annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n[Sentinel] Output saved: {output_path}")
    _generate_report()


# ─────────────────────────────────────────────
# 7. ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    video = sys.argv[1] if len(sys.argv) > 1 else "speed1.mp4"
    run_enforcement(video)