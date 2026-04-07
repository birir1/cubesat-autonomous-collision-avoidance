"""
YOLOv8 Satellite Object Detector

Implements YOLOv8-based object detection for satellite imagery
and space situational awareness applications.
"""

import torch
import torch.nn as nn
import numpy as np
import cv2
from pathlib import Path
import logging
from typing import List, Dict, Tuple, Optional

try:
    import ultralytics
    from ultralytics import YOLO
except ImportError:
    print("Warning: ultralytics not installed. Install with: pip install ultralytics")
    YOLO = None

class YOLOv8Detector:
    """
    YOLOv8-based detector for satellite object detection.
    """

    def __init__(self, model_path: Optional[str] = None,
                 confidence_threshold: float = 0.5,
                 device: str = 'cuda'):
        """
        Initialize YOLOv8 detector.

        Args:
            model_path: Path to trained model weights (.pt file)
            confidence_threshold: Detection confidence threshold
            device: Device to run model on
        """
        self.confidence_threshold = confidence_threshold
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.logger = logging.getLogger(__name__)

        # Model parameters
        self.num_classes = 2  # satellite, debris
        self.input_size = 640  # YOLOv8 default

        # Initialize model
        self.model = self._load_model(model_path)

    def _load_model(self, model_path: Optional[str]):
        """Load YOLOv8 model."""
        if YOLO is None:
            self.logger.warning("YOLOv8 not available, using fallback detector")
            return FallbackYOLO()

        if model_path and Path(model_path).exists():
            try:
                model = YOLO(model_path)
                self.logger.info(f"Loaded YOLOv8 model from {model_path}")
                return model
            except Exception as e:
                self.logger.warning(f"Failed to load model: {e}")

        # Load default YOLOv8 model
        try:
            model = YOLO('yolov8n.pt')  # nano model
            self.logger.info("Loaded default YOLOv8 nano model")
            return model
        except Exception as e:
            self.logger.warning(f"Failed to load default model: {e}")
            return FallbackYOLO()

    def detect(self, image: np.ndarray) -> List[Dict]:
        """
        Detect objects in image.

        Args:
            image: Input image (H, W, C) or path to image

        Returns:
            List of detections with format:
            {'bbox': [x1, y1, x2, y2], 'confidence': float, 'class': int}
        """
        if isinstance(image, str):
            # Load image from path
            image = cv2.imread(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Run inference
        results = self.model(image, conf=self.confidence_threshold, device=self.device)

        # Process results
        detections = []
        if hasattr(results[0], 'boxes'):
            boxes = results[0].boxes
            for box in boxes:
                # Get box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

                # Get confidence and class
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])

                detection = {
                    'bbox': [float(x1), float(y1), float(x2), float(y2)],
                    'confidence': confidence,
                    'class': class_id
                }
                detections.append(detection)

        return detections

    def detect_batch(self, images: List[np.ndarray]) -> List[List[Dict]]:
        """
        Detect objects in batch of images.

        Args:
            images: List of input images

        Returns:
            List of detection lists for each image
        """
        results = self.model(images, conf=self.confidence_threshold, device=self.device)

        batch_detections = []
        for result in results:
            detections = []
            if hasattr(result, 'boxes'):
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])

                    detection = {
                        'bbox': [float(x1), float(y1), float(x2), float(y2)],
                        'confidence': confidence,
                        'class': class_id
                    }
                    detections.append(detection)
            batch_detections.append(detections)

        return batch_detections

    def get_class_names(self) -> List[str]:
        """Get class names."""
        if hasattr(self.model, 'names'):
            return list(self.model.names.values())
        return ['satellite', 'debris']

    def train(self, data_yaml: str, epochs: int = 100, batch_size: int = 16,
              img_size: int = 640, **kwargs):
        """
        Train YOLOv8 model.

        Args:
            data_yaml: Path to data YAML file
            epochs: Number of training epochs
            batch_size: Batch size
            img_size: Image size
            **kwargs: Additional training arguments
        """
        if not hasattr(self.model, 'train'):
            self.logger.error("Model does not support training")
            return

        self.logger.info(f"Starting training for {epochs} epochs")

        results = self.model.train(
            data=data_yaml,
            epochs=epochs,
            batch=batch_size,
            imgsz=img_size,
            device=self.device,
            **kwargs
        )

        return results

    def validate(self, data_yaml: str):
        """
        Validate model on dataset.

        Args:
            data_yaml: Path to data YAML file

        Returns:
            Validation metrics
        """
        if not hasattr(self.model, 'val'):
            self.logger.error("Model does not support validation")
            return None

        results = self.model.val(data=data_yaml, device=self.device)
        return results

    def export(self, format: str = 'onnx', **kwargs):
        """
        Export model to different formats.

        Args:
            format: Export format ('onnx', 'torchscript', etc.)
            **kwargs: Additional export arguments
        """
        if not hasattr(self.model, 'export'):
            self.logger.error("Model does not support export")
            return None

        export_path = self.model.export(format=format, **kwargs)
        self.logger.info(f"Model exported to {export_path}")
        return export_path

    def __str__(self):
        return f"YOLOv8Detector(conf_threshold={self.confidence_threshold}, device={self.device})"


class FallbackYOLO:
    """
    Fallback detector when YOLOv8 is not available.
    """

    def __init__(self):
        self.names = {0: 'satellite', 1: 'debris'}

    def __call__(self, image, conf=0.5, device='cpu'):
        """Mock detection results."""
        # Generate random detections
        n_detections = np.random.randint(0, 5)
        detections = []

        h, w = image.shape[:2] if isinstance(image, np.ndarray) else (640, 640)

        for _ in range(n_detections):
            x1 = np.random.uniform(0, w * 0.8)
            y1 = np.random.uniform(0, h * 0.8)
            x2 = x1 + np.random.uniform(20, 100)
            y2 = y1 + np.random.uniform(20, 100)

            confidence = np.random.uniform(conf, 1.0)
            class_id = np.random.randint(0, 2)

            detection = {
                'bbox': [float(x1), float(y1), float(x2), float(y2)],
                'confidence': float(confidence),
                'class': int(class_id)
            }
            detections.append(detection)

        # Mock result object
        class MockResult:
            def __init__(self, detections):
                self.detections = detections

            @property
            def boxes(self):
                class MockBoxes:
                    def __init__(self, detections):
                        self.detections = detections

                    def __iter__(self):
                        for det in self.detections:
                            yield MockBox(det)

                return MockBoxes(self.detections)

        class MockBox:
            def __init__(self, detection):
                self.detection = detection

            @property
            def xyxy(self):
                return [torch.tensor(self.detection['bbox'])]

            @property
            def conf(self):
                return [torch.tensor(self.detection['confidence'])]

            @property
            def cls(self):
                return [torch.tensor(self.detection['class'])]

        return [MockResult(detections)]


class SatelliteDetectionTrainer:
    """
    Trainer class for YOLOv8 satellite detection models.
    """

    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize trainer.

        Args:
            model_path: Path to base model
        """
        self.model_path = model_path or 'yolov8n.pt'
        self.logger = logging.getLogger(__name__)

    def create_data_yaml(self, train_path: str, val_path: str,
                        class_names: List[str], output_path: str):
        """
        Create YOLO data YAML file.

        Args:
            train_path: Path to training images
            val_path: Path to validation images
            class_names: List of class names
            output_path: Output YAML path
        """
        data_config = {
            'train': train_path,
            'val': val_path,
            'nc': len(class_names),
            'names': class_names
        }

        import yaml
        with open(output_path, 'w') as f:
            yaml.dump(data_config, f, default_flow_style=False)

        self.logger.info(f"Data YAML created at {output_path}")

    def train_model(self, data_yaml: str, project_name: str = 'satellite_detection',
                   epochs: int = 100, batch_size: int = 16):
        """
        Train satellite detection model.

        Args:
            data_yaml: Path to data configuration
            project_name: Name for training project
            epochs: Number of training epochs
            batch_size: Training batch size
        """
        model = YOLO(self.model_path)

        results = model.train(
            data=data_yaml,
            epochs=epochs,
            batch=batch_size,
            project=project_name,
            name=f'train_{epochs}epochs',
            save=True,
            save_period=10,
            cache=False,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )

        self.logger.info("Training completed")
        return results


if __name__ == "__main__":
    # Example usage
    detector = YOLOv8Detector()

    # Create dummy image
    dummy_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

    # Run detection
    detections = detector.detect(dummy_image)

    print(f"Found {len(detections)} detections")
    for i, det in enumerate(detections):
        print(f"Detection {i+1}: {det}")

    # Print class names
    print(f"Classes: {detector.get_class_names()}")