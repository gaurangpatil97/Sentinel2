import cv2
import torch
import numpy as np
from ultralytics import YOLO
import time
import os
from collections import defaultdict

def run_master_helmet_system(video_source):
    # 1. SETUP & PATHS
    model_path = r"C:\Users\gaura\Downloads\Sentinel_Final_Package\weights\best3.pt"
    
    if not os.path.exists(model_path):
        print(f"❌ CRITICAL ERROR: Weights not found at {model_path}")
        return

    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_source)

    actual_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    time_per_frame = 1.0 / actual_fps

    # DISPLAY CANVAS SETTINGS
    CANVAS_W, CANVAS_H = 1280, 720            
    
    id_map = {}                          
    tracker_history = defaultdict(int)   
    next_clean_id = 1                    

    print(f"🚀 SYSTEM ONLINE | Processing: {video_source}")

    while cap.isOpened():
        start_time = time.perf_counter()
        ret, frame = cap.read()
        if not ret: break

        # 2. RUN TRACKING
        results = model.track(frame, persist=True, device=0, conf=0.3, verbose=False)[0]
        annotated_frame = frame.copy()

        if results.boxes.id is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            raw_ids = results.boxes.id.cpu().numpy().astype(int)
            classes = results.boxes.cls.cpu().numpy().astype(int)
            
            for box, raw_tid, cls_idx in zip(boxes, raw_ids, classes):
                if raw_tid not in id_map:
                    tracker_history[raw_tid] += 1
                    if tracker_history[raw_tid] >= 3:
                        id_map[raw_tid] = next_clean_id
                        status_name = model.names[cls_idx]
                        print(f"🚨 CONFIRMED | ID {next_clean_id} is [{status_name}]")
                        next_clean_id += 1

                if raw_tid in id_map:
                    clean_id = id_map[raw_tid]
                    status_name = model.names[cls_idx]
                    label = f"ID:{clean_id} {status_name}"
                    x1, y1, x2, y2 = map(int, box)
                    color = (0, 0, 255) if "no-helmet" in status_name else (0, 255, 0)
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 4)
                    cv2.putText(annotated_frame, label, (x1, y1 - 15), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

        # --- 5. UPDATED UNIVERSAL LETTERBOX ENGINE ---
        h, w = annotated_frame.shape[:2]
        
        # Calculate scale to fit within BOTH width and height
        scale = min(CANVAS_W / w, CANVAS_H / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        # Resize frame
        resized_f = cv2.resize(annotated_frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Create black canvas
        display_frame = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)
        
        # Center horizontally AND vertically
        x_offset = (CANVAS_W - new_w) // 2
        y_offset = (CANVAS_H - new_h) // 2
        
        # Safe placement
        display_frame[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_f

        # TOP DASHBOARD UI
        cv2.rectangle(display_frame, (0, 0), (450, 60), (0, 0, 0), -1)
        cv2.putText(display_frame, f"UNIQUE RIDERS: {next_clean_id - 1}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow("KJSCE AI Guardian - Master Build", display_frame)

        elapsed = time.perf_counter() - start_time
        wait_ms = max(1, int((time_per_frame - elapsed) * 1000))
        if cv2.waitKey(wait_ms) & 0xFF == ord('q'): 
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = "helmet2.mp4"
    run_master_helmet_system(video_path)