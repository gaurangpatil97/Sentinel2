import cv2
import numpy as np
from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO(r"C:\Users\gaura\Desktop\Sentinel\models\HelmetDetection\best.pt")

    results = model(r"C:\Users\gaura\Desktop\Sentinel\models\HelmetDetection\image.png", conf=0.25)

    img = cv2.imread(r"C:\Users\gaura\Desktop\Sentinel\models\HelmetDetection\image.png")

    detections = []
    for box in results[0].boxes:
        cls = results[0].names[int(box.cls)]
        conf = float(box.conf)
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        detections.append((cls, conf, x1, y1, x2, y2))

    # Track IDs per class
    driver_id = 1
    bicycle_id = 1

    # First pass — find helmet/no-helmet boxes to pair with drivers
    helmet_boxes = [(x1, y1, x2, y2) for cls, conf, x1, y1, x2, y2 in detections if cls == "helmet"]
    no_helmet_boxes = [(x1, y1, x2, y2) for cls, conf, x1, y1, x2, y2 in detections if cls == "no-helmet"]

    def is_overlapping(box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        return x1 < x2 and y1 < y2

    no_helmet_count = 0
    helmet_count = 0
    bicycle_count = 0

    for cls, conf, x1, y1, x2, y2 in detections:
        if cls == "helmet" or cls == "no-helmet":
            continue  # skip, handled via driver

        if cls == "driver":
            driver_box = (x1, y1, x2, y2)
            has_helmet = any(is_overlapping(driver_box, hb) for hb in helmet_boxes)
            has_no_helmet = any(is_overlapping(driver_box, nb) for nb in no_helmet_boxes)

            if has_no_helmet:
                color = (0, 0, 255)  # red
                label = f"D{driver_id} | NO HELMET"
                no_helmet_count += 1
            elif has_helmet:
                color = (0, 255, 0)  # green
                label = f"D{driver_id} | HELMET"
                helmet_count += 1
            else:
                color = (255, 165, 0)  # orange - uncertain
                label = f"D{driver_id} | UNKNOWN"

            driver_id += 1

        elif cls == "bicyclist":
            color = (255, 255, 0)  # yellow
            label = f"B{bicycle_id} | BICYCLIST"
            bicycle_id += 1
            bicycle_count += 1

        # Draw box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        # Draw label background
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(img, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Summary overlay
    summary = f"Helmet: {helmet_count}  |  No Helmet: {no_helmet_count}  |  Bicyclists: {bicycle_count}"
    cv2.rectangle(img, (0, 0), (len(summary) * 11, 35), (0, 0, 0), -1)
    cv2.putText(img, summary, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    if no_helmet_count > 0:
        alert = f"VIOLATION: {no_helmet_count} rider(s) without helmet!"
        cv2.putText(img, alert, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    output_path = r"C:\Users\gaura\Desktop\Sentinel\models\HelmetDetection\output5.jpg"
    cv2.imwrite(output_path, img)
    print(f"Helmet: {helmet_count} | No Helmet: {no_helmet_count} | Bicyclists: {bicycle_count}")
    if no_helmet_count > 0:
        print(f"🚨 VIOLATION — {no_helmet_count} rider(s) without helmet!")
    print(f"Saved to {output_path}")