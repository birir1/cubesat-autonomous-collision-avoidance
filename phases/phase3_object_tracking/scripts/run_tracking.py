# run_tracking.py
"""
Full Object Tracking Pipeline

Pipeline:
Images -> YOLO Detection -> Deep SORT Tracking -> State Vector Output

Outputs:
results/tracking_outputs/tracked_objects.csv
results/figures/tracking_visualization.mp4
"""

import os
import cv2
import glob
import torch
import pandas as pd
from tqdm import tqdm

from ultralytics import YOLO

from phases.phase3_object_tracking.models.deep_sort_tracker import DeepSortTracker


# ------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------

IMAGE_FOLDER = "data/raw/debris-detection/train"
OUTPUT_VIDEO = "results/figures/tracking_visualization.mp4"
OUTPUT_CSV = "results/tracking_outputs/tracked_objects.csv"

YOLO_MODEL = "yolov8n.pt"

CONF_THRESHOLD = 0.25

os.makedirs("results/tracking_outputs", exist_ok=True)
os.makedirs("results/figures", exist_ok=True)


# ------------------------------------------------------
# LOAD YOLO MODEL
# ------------------------------------------------------

def load_detector():

    print("Loading YOLO model...")

    model = YOLO(YOLO_MODEL)

    return model


# ------------------------------------------------------
# DETECTION FUNCTION
# ------------------------------------------------------

def run_detection(model, frame):

    results = model(frame)[0]

    detections = []

    for box in results.boxes:

        conf = float(box.conf[0])

        if conf < CONF_THRESHOLD:
            continue

        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

        detections.append({
            "bbox": [x1, y1, x2, y2],
            "confidence": conf
        })

    return detections


# ------------------------------------------------------
# DRAW TRACKS
# ------------------------------------------------------

def draw_tracks(frame, tracks):

    for t in tracks:

        x, y = t.get_position()

        track_id = t.track_id

        cv2.circle(frame, (int(x), int(y)), 5, (0,255,0), -1)

        cv2.putText(
            frame,
            f"ID {track_id}",
            (int(x)+5, int(y)+5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0,255,0),
            1
        )

    return frame


# ------------------------------------------------------
# LOAD IMAGE SEQUENCE
# ------------------------------------------------------

def load_images():

    images = sorted(
        glob.glob(os.path.join(IMAGE_FOLDER, "*.jpg"))
    )

    return images


# ------------------------------------------------------
# MAIN PIPELINE
# ------------------------------------------------------

def run_pipeline():

    detector = load_detector()

    tracker = DeepSortTracker()

    image_paths = load_images()

    if len(image_paths) == 0:
        raise RuntimeError("No images found.")

    first_frame = cv2.imread(image_paths[0])
    h, w, _ = first_frame.shape

    video_writer = cv2.VideoWriter(
        OUTPUT_VIDEO,
        cv2.VideoWriter_fourcc(*"mp4v"),
        10,
        (w, h)
    )

    tracking_data = []

    print("Running tracking pipeline...")

    for frame_idx, img_path in enumerate(tqdm(image_paths)):

        frame = cv2.imread(img_path)

        detections = run_detection(detector, frame)

        tracks = tracker.update(frame, detections)

        frame = draw_tracks(frame, tracks)

        video_writer.write(frame)

        # store tracking data
        for t in tracks:

            px, py = t.get_position()
            vx, vy = t.get_velocity()

            tracking_data.append({

                "frame": frame_idx,
                "track_id": t.track_id,

                "pos_x": float(px),
                "pos_y": float(py),

                "vel_x": float(vx),
                "vel_y": float(vy)
            })

    video_writer.release()

    df = pd.DataFrame(tracking_data)

    df.to_csv(OUTPUT_CSV, index=False)

    print("Tracking complete.")
    print("Saved:", OUTPUT_CSV)
    print("Video:", OUTPUT_VIDEO)


# ------------------------------------------------------

if __name__ == "__main__":
    run_pipeline()