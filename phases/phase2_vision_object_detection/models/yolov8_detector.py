# yolov8_detector.py
"""
YOLOv8 Space Object Detection Model

Implements training and inference for detecting
satellites and debris in CubeSat camera imagery.

Features:
- YOLOv8 training pipeline
- inference pipeline
- bounding box postprocessing
- dataset integration
- model export
"""

import os
import cv2
import torch
import numpy as np
from ultralytics import YOLO


# -------------------------------------------------------
# Detector Class
# -------------------------------------------------------

class YOLOv8SpaceObjectDetector:

    def __init__(
        self,
        model_name="yolov8m.pt",
        device=None
    ):
        """
        Initialize YOLOv8 detector.

        model_name:
            yolov8n.pt
            yolov8s.pt
            yolov8m.pt
            yolov8l.pt
        """

        self.device = device if device else (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model = YOLO(model_name)

        self.model.to(self.device)

    # ---------------------------------------------------

    def train(
        self,
        data_config,
        epochs=100,
        batch=16,
        imgsz=640,
        project="results/detection_training",
        name="yolov8_space_objects"
    ):
        """
        Train YOLOv8 model.
        """

        self.model.train(
            data=data_config,
            epochs=epochs,
            batch=batch,
            imgsz=imgsz,
            project=project,
            name=name,
            device=self.device
        )

    # ---------------------------------------------------

    def predict_image(
        self,
        image_path,
        conf=0.25,
        iou=0.5,
        save=False
    ):
        """
        Run inference on a single image.
        """

        results = self.model.predict(
            source=image_path,
            conf=conf,
            iou=iou,
            save=save,
            device=self.device
        )

        detections = []

        for r in results:

            boxes = r.boxes.xyxy.cpu().numpy()
            scores = r.boxes.conf.cpu().numpy()
            classes = r.boxes.cls.cpu().numpy()

            for box, score, cls in zip(boxes, scores, classes):

                detections.append({
                    "bbox": box.tolist(),
                    "confidence": float(score),
                    "class_id": int(cls)
                })

        return detections

    # ---------------------------------------------------

    def predict_frame(self, frame):

        results = self.model.predict(
            frame,
            device=self.device
        )

        detections = []

        for r in results:

            boxes = r.boxes.xyxy.cpu().numpy()
            scores = r.boxes.conf.cpu().numpy()
            classes = r.boxes.cls.cpu().numpy()

            for box, score, cls in zip(boxes, scores, classes):

                detections.append({
                    "bbox": box.tolist(),
                    "confidence": float(score),
                    "class_id": int(cls)
                })

        return detections

    # ---------------------------------------------------

    def predict_video(
        self,
        video_path,
        output_path="results/detection_video.mp4"
    ):
        """
        Run detection on video stream.
        """

        cap = cv2.VideoCapture(video_path)

        width = int(cap.get(3))
        height = int(cap.get(4))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        out = cv2.VideoWriter(
            output_path,
            fourcc,
            fps,
            (width, height)
        )

        while cap.isOpened():

            ret, frame = cap.read()

            if not ret:
                break

            detections = self.predict_frame(frame)

            for det in detections:

                x1, y1, x2, y2 = map(int, det["bbox"])

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                label = f"{det['class_id']} {det['confidence']:.2f}"

                cv2.putText(
                    frame,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2
                )

            out.write(frame)

        cap.release()
        out.release()

    # ---------------------------------------------------

    def export_model(self, format="onnx"):

        """
        Export trained model to deployment format.

        Supported:
        onnx
        torchscript
        openvino
        """

        self.model.export(format=format)

    # ---------------------------------------------------

    def evaluate(self, data_config):

        """
        Evaluate model on validation dataset.
        """

        metrics = self.model.val(data=data_config)

        return metrics


# -------------------------------------------------------
# Dataset Configuration Generator
# -------------------------------------------------------

def generate_dataset_yaml(
    train_images,
    val_images,
    test_images,
    class_names,
    output_path="dataset.yaml"
):
    """
    Generate YOLO dataset config file.
    """

    yaml_content = f"""
train: {train_images}
val: {val_images}
test: {test_images}

nc: {len(class_names)}
names: {class_names}
"""

    with open(output_path, "w") as f:
        f.write(yaml_content)

    print(f"Dataset YAML created at {output_path}")