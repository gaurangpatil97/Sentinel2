from __future__ import annotations

import base64
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from ultralytics import YOLO

MODEL_PATH = Path(r"C:\Users\gaura\Desktop\Sentinel\models\HelmetDetection\best.pt")
SUMMARY_BAR_HEIGHT = 74

model: YOLO | None = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    global model

    if not MODEL_PATH.exists():
        raise RuntimeError(f"YOLO model not found at {MODEL_PATH}")

    model = YOLO(str(MODEL_PATH))
    yield


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def normalize_class_name(name: str) -> str:
    normalized = name.strip().lower().replace("_", "-")
    if normalized == "nohelmet":
        return "no-helmet"
    return normalized


def is_overlapping(box1: tuple[int, int, int, int], box2: tuple[int, int, int, int]) -> bool:
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    return x1 < x2 and y1 < y2


def draw_labeled_box(
    image: np.ndarray,
    box: tuple[int, int, int, int],
    label: str,
    color: tuple[int, int, int],
) -> None:
    x1, y1, x2, y2 = box
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
    label_top = max(0, y1 - text_height - baseline - 8)
    cv2.rectangle(image, (x1, label_top), (x1 + text_width + 8, y1), color, -1)
    cv2.putText(
        image,
        label,
        (x1 + 4, y1 - 5),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


def make_summary_canvas(
    image: np.ndarray,
    helmet_count: int,
    no_helmet_count: int,
    bicyclist_count: int,
    violation: bool,
) -> np.ndarray:
    height, width = image.shape[:2]
    canvas = np.zeros((height + SUMMARY_BAR_HEIGHT, width, 3), dtype=np.uint8)
    canvas[:] = (8, 12, 20)
    canvas[SUMMARY_BAR_HEIGHT:, :width] = image

    cv2.rectangle(canvas, (0, 0), (width, SUMMARY_BAR_HEIGHT), (10, 10, 10), -1)
    cv2.line(canvas, (0, SUMMARY_BAR_HEIGHT - 1), (width, SUMMARY_BAR_HEIGHT - 1), (45, 55, 72), 1)

    summary = f"Helmet: {helmet_count}   No Helmet: {no_helmet_count}   Bicyclists: {bicyclist_count}"
    cv2.putText(
        canvas,
        summary,
        (16, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (245, 245, 245),
        2,
        cv2.LINE_AA,
    )

    if violation:
        violation_text = f"VIOLATION: {no_helmet_count} rider(s) without helmet"
        cv2.putText(
            canvas,
            violation_text,
            (16, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    return canvas


def encode_image_to_base64(image: np.ndarray) -> str:
    success, buffer = cv2.imencode(".png", image)
    if not success:
        raise ValueError("Failed to encode annotated image.")
    return base64.b64encode(buffer.tobytes()).decode("utf-8")


@app.post("/detect")
async def detect(image: UploadFile = File(...)) -> JSONResponse:
    if model is None:
        raise HTTPException(status_code=503, detail="Model failed to load at startup.")

    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    try:
        raw_bytes = await image.read()
        if not raw_bytes:
            raise HTTPException(status_code=400, detail="No image data was provided.")

        np_bytes = np.frombuffer(raw_bytes, dtype=np.uint8)
        decoded_image = cv2.imdecode(np_bytes, cv2.IMREAD_COLOR)
        if decoded_image is None:
            raise HTTPException(status_code=400, detail="Unable to decode the uploaded image.")

        results = model(decoded_image, conf=0.25, verbose=False)
        result = results[0]

        detections: list[tuple[str, float, int, int, int, int]] = []
        for box in result.boxes:
            class_id = int(box.cls.item())
            class_name = normalize_class_name(result.names[class_id])
            confidence = float(box.conf.item())
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            detections.append((class_name, confidence, x1, y1, x2, y2))

        helmet_boxes = [(x1, y1, x2, y2) for class_name, _, x1, y1, x2, y2 in detections if class_name == "helmet"]
        no_helmet_boxes = [(x1, y1, x2, y2) for class_name, _, x1, y1, x2, y2 in detections if class_name == "no-helmet"]

        annotated = decoded_image.copy()
        helmet_count = 0
        no_helmet_count = 0
        bicyclist_count = 0
        driver_count = 0
        driver_id = 1
        bicycle_id = 1

        for class_name, _, x1, y1, x2, y2 in detections:
            if class_name in {"helmet", "no-helmet"}:
                continue

            if class_name == "driver":
                driver_count += 1
                driver_box = (x1, y1, x2, y2)
                has_helmet = any(is_overlapping(driver_box, helmet_box) for helmet_box in helmet_boxes)
                has_no_helmet = any(is_overlapping(driver_box, no_helmet_box) for no_helmet_box in no_helmet_boxes)

                if has_no_helmet:
                    color = (0, 0, 255)
                    label = f"D{driver_id} | NO HELMET"
                    no_helmet_count += 1
                elif has_helmet:
                    color = (0, 255, 0)
                    label = f"D{driver_id} | HELMET"
                    helmet_count += 1
                else:
                    color = (0, 165, 255)
                    label = f"D{driver_id} | UNKNOWN"

                draw_labeled_box(annotated, driver_box, label, color)
                driver_id += 1
                continue

            if class_name == "bicyclist":
                bicyclist_count += 1
                bicycle_box = (x1, y1, x2, y2)
                label = f"B{bicycle_id} | BICYCLIST"
                draw_labeled_box(annotated, bicycle_box, label, (0, 255, 255))
                bicycle_id += 1

        violation = no_helmet_count > 0
        output_image = make_summary_canvas(annotated, helmet_count, no_helmet_count, bicyclist_count, violation)
        output_b64 = encode_image_to_base64(output_image)

        return JSONResponse(
            status_code=200,
            content={
                "output_image": output_b64,
                "helmet_count": helmet_count,
                "no_helmet_count": no_helmet_count,
                "bicyclist_count": bicyclist_count,
                "violation": violation,
                "driver_count": driver_count,
            },
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Detection failed: {exc}") from exc


@app.exception_handler(HTTPException)
async def http_exception_handler(_: Any, exc: HTTPException) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content={"error": exc.detail})


@app.exception_handler(Exception)
async def unhandled_exception_handler(_: Any, exc: Exception) -> JSONResponse:
    return JSONResponse(status_code=500, content={"error": f"Internal server error: {exc}"})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=5000)
