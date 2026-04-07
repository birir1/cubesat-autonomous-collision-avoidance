"""
Vision Processor for CubeSat Collision Avoidance

Handles image preprocessing, object detection, and feature extraction.
"""

import numpy as np
import cv2
import torch
import torchvision.transforms as transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
import logging
import yaml
from PIL import Image

class VisionProcessor:
    """
    Vision processing pipeline for satellite imagery and object detection.
    """

    def __init__(self, config_path: str = 'configs/data_config.yaml'):
        """
        Initialize vision processor.

        Args:
            config_path: Path to configuration file
        """
        self.logger = logging.getLogger(__name__)

        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.vision_config = self.config['vision']

        # Image preprocessing parameters
        self.target_size = tuple(self.vision_config['target_size'])
        self.normalize_mean = self.vision_config['normalize_mean']
        self.normalize_std = self.vision_config['normalize_std']

        # Object detection parameters
        self.confidence_threshold = self.vision_config['confidence_threshold']
        self.iou_threshold = self.vision_config['iou_threshold']

        # Initialize transforms
        self._setup_transforms()

        # Initialize detection model (lazy loading)
        self.detection_model = None

        # Class labels for satellite objects
        self.class_labels = {
            1: 'satellite',
            2: 'debris',
            3: 'spacecraft',
            4: 'rocket_body'
        }

    def _setup_transforms(self) -> None:
        """Setup image preprocessing transforms."""
        self.transform = transforms.Compose([
            transforms.Resize(self.target_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=self.normalize_mean,
                std=self.normalize_std
            )
        ])

        # Raw image transform (no normalization for visualization)
        self.raw_transform = transforms.Compose([
            transforms.Resize(self.target_size),
            transforms.ToTensor()
        ])

    def _load_detection_model(self) -> None:
        """Lazy load object detection model."""
        if self.detection_model is None:
            self.logger.info("Loading Faster R-CNN detection model...")
            self.detection_model = fasterrcnn_resnet50_fpn(pretrained=True)
            self.detection_model.eval()

            # Move to GPU if available
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.detection_model.to(self.device)
            self.logger.info(f"Detection model loaded on {self.device}")

    def preprocess_image(self, image: Union[np.ndarray, str, Path],
                        return_raw: bool = False) -> Tuple[torch.Tensor, Optional[np.ndarray]]:
        """
        Preprocess image for model input.

        Args:
            image: Image as numpy array, file path, or PIL Image
            return_raw: Whether to return raw resized image

        Returns:
            Tuple of (processed_tensor, raw_image_array)
        """
        # Load image if path provided
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            # Convert numpy array to PIL Image
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
            image = Image.fromarray(image)
        elif not isinstance(image, Image.Image):
            raise ValueError("Image must be numpy array, file path, or PIL Image")

        # Apply transforms
        processed = self.transform(image)

        if return_raw:
            raw = self.raw_transform(image)
            raw_array = raw.numpy().transpose(1, 2, 0)  # CHW to HWC
            return processed, raw_array

        return processed, None

    def detect_objects(self, image: torch.Tensor) -> List[Dict[str, Any]]:
        """
        Perform object detection on preprocessed image.

        Args:
            image: Preprocessed image tensor (C, H, W)

        Returns:
            List of detection dictionaries
        """
        self._load_detection_model()

        # Add batch dimension
        if image.dim() == 3:
            image = image.unsqueeze(0)

        # Move to device
        image = image.to(self.device)

        with torch.no_grad():
            predictions = self.detection_model(image)

        # Process predictions
        detections = []
        pred = predictions[0]  # First (and only) image

        # Filter by confidence
        keep = pred['scores'] > self.confidence_threshold
        boxes = pred['boxes'][keep].cpu().numpy()
        scores = pred['scores'][keep].cpu().numpy()
        labels = pred['labels'][keep].cpu().numpy()

        # Apply non-maximum suppression
        if len(boxes) > 0:
            indices = self._nms(boxes, scores, self.iou_threshold)

            for idx in indices:
                detection = {
                    'bbox': boxes[idx].tolist(),
                    'confidence': float(scores[idx]),
                    'class_id': int(labels[idx]),
                    'class_name': self.class_labels.get(int(labels[idx]), 'unknown'),
                    'area': float((boxes[idx][2] - boxes[idx][0]) * (boxes[idx][3] - boxes[idx][1]))
                }
                detections.append(detection)

        return detections

    def _nms(self, boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> List[int]:
        """
        Non-maximum suppression for bounding boxes.

        Args:
            boxes: Bounding boxes [x1, y1, x2, y2]
            scores: Confidence scores
            iou_threshold: IoU threshold for suppression

        Returns:
            Indices of boxes to keep
        """
        if len(boxes) == 0:
            return []

        # Sort by confidence score
        indices = np.argsort(scores)[::-1]

        keep = []
        while len(indices) > 0:
            # Keep the box with highest score
            current = indices[0]
            keep.append(current)

            if len(indices) == 1:
                break

            # Calculate IoU with remaining boxes
            remaining = indices[1:]
            ious = self._calculate_iou(boxes[current], boxes[remaining])

            # Keep boxes with IoU below threshold
            indices = remaining[ious < iou_threshold]

        return keep

    def _calculate_iou(self, box1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
        """
        Calculate IoU between one box and multiple boxes.

        Args:
            box1: Single bounding box [x1, y1, x2, y2]
            boxes2: Multiple bounding boxes (N, 4)

        Returns:
            IoU values (N,)
        """
        # Ensure boxes2 is 2D
        if boxes2.ndim == 1:
            boxes2 = boxes2.reshape(1, -1)

        # Calculate intersection coordinates
        x1 = np.maximum(box1[0], boxes2[:, 0])
        y1 = np.maximum(box1[1], boxes2[:, 1])
        x2 = np.minimum(box1[2], boxes2[:, 2])
        y2 = np.minimum(box1[3], boxes2[:, 3])

        # Calculate intersection area
        intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

        # Calculate union area
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        boxes2_area = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        union = box1_area + boxes2_area - intersection

        # Calculate IoU
        iou = intersection / np.maximum(union, 1e-6)

        return iou

    def extract_detection_features(self, detections: List[Dict[str, Any]],
                                 image_shape: Tuple[int, int]) -> Dict[str, Any]:
        """
        Extract high-level features from object detections.

        Args:
            detections: List of detection dictionaries
            image_shape: Image shape (height, width)

        Returns:
            Dictionary of detection features
        """
        features = {
            'num_detections': len(detections),
            'detection_confidences': [],
            'detection_areas': [],
            'detection_centers': [],
            'class_counts': {label: 0 for label in self.class_labels.values()},
            'largest_detection_area': 0.0,
            'total_detection_area': 0.0,
            'detection_density': 0.0
        }

        image_area = image_shape[0] * image_shape[1]

        for detection in detections:
            confidence = detection['confidence']
            area = detection['area']
            bbox = detection['bbox']

            # Basic features
            features['detection_confidences'].append(confidence)
            features['detection_areas'].append(area)

            # Center coordinates
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2
            features['detection_centers'].append([center_x, center_y])

            # Class counts
            class_name = detection['class_name']
            if class_name in features['class_counts']:
                features['class_counts'][class_name] += 1

            # Area statistics
            features['largest_detection_area'] = max(features['largest_detection_area'], area)
            features['total_detection_area'] += area

        # Calculate density
        if image_area > 0:
            features['detection_density'] = features['total_detection_area'] / image_area

        # Convert lists to numpy arrays for easier processing
        features['detection_confidences'] = np.array(features['detection_confidences'])
        features['detection_areas'] = np.array(features['detection_areas'])
        features['detection_centers'] = np.array(features['detection_centers'])

        return features

    def simulate_satellite_detection(self, num_satellites: int = 3,
                                   image_shape: Tuple[int, int] = (224, 224),
                                   noise_level: float = 0.1) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Simulate satellite detections for testing (when real images not available).

        Args:
            num_satellites: Number of satellites to simulate
            image_shape: Image dimensions
            noise_level: Random noise in detection parameters

        Returns:
            Tuple of (simulated_image, detections)
        """
        # Create blank image
        image = np.zeros((3, image_shape[0], image_shape[1]), dtype=np.float32)

        detections = []

        for i in range(num_satellites):
            # Random position and size
            center_x = np.random.uniform(0.2, 0.8) * image_shape[1]
            center_y = np.random.uniform(0.2, 0.8) * image_shape[0]
            size = np.random.uniform(20, 60)

            # Add noise
            center_x += np.random.normal(0, noise_level * image_shape[1])
            center_y += np.random.normal(0, noise_level * image_shape[0])
            size *= (1 + np.random.normal(0, noise_level))

            # Ensure within bounds
            center_x = np.clip(center_x, size/2, image_shape[1] - size/2)
            center_y = np.clip(center_y, size/2, image_shape[0] - size/2)

            # Create bounding box
            x1 = center_x - size/2
            y1 = center_y - size/2
            x2 = center_x + size/2
            y2 = center_y + size/2

            # Random class and confidence
            class_id = np.random.choice(list(self.class_labels.keys()))
            confidence = np.random.uniform(0.7, 0.95)

            detection = {
                'bbox': [float(x1), float(y1), float(x2), float(y2)],
                'confidence': float(confidence),
                'class_id': int(class_id),
                'class_name': self.class_labels[class_id],
                'area': float((x2 - x1) * (y2 - y1))
            }
            detections.append(detection)

            # Add visual marker to image (simple rectangle)
            image[:, int(y1):int(y2), int(x1):int(x2)] = np.random.uniform(0.5, 1.0, (3, 1, 1))

        return image, detections

    def process_image_batch(self, images: List[Union[np.ndarray, str, Path]],
                           perform_detection: bool = True) -> List[Dict[str, Any]]:
        """
        Process a batch of images.

        Args:
            images: List of images
            perform_detection: Whether to perform object detection

        Returns:
            List of processing results
        """
        results = []

        for image in images:
            try:
                # Preprocess image
                processed_tensor, raw_image = self.preprocess_image(image, return_raw=True)

                result = {
                    'processed_tensor': processed_tensor,
                    'raw_image': raw_image,
                    'detections': [],
                    'detection_features': {}
                }

                if perform_detection:
                    # Perform detection
                    detections = self.detect_objects(processed_tensor)
                    result['detections'] = detections

                    # Extract features
                    if raw_image is not None:
                        detection_features = self.extract_detection_features(
                            detections, raw_image.shape[:2]
                        )
                        result['detection_features'] = detection_features

                results.append(result)

            except Exception as e:
                self.logger.error(f"Error processing image: {e}")
                results.append({
                    'processed_tensor': None,
                    'raw_image': None,
                    'detections': [],
                    'detection_features': {},
                    'error': str(e)
                })

        return results

    def save_processed_data(self, results: List[Dict[str, Any]], output_dir: str) -> None:
        """
        Save processed vision data.

        Args:
            results: Processing results
            output_dir: Output directory
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save tensors
        tensors = []
        detections_list = []
        features_list = []

        for result in results:
            if result['processed_tensor'] is not None:
                tensors.append(result['processed_tensor'])
            detections_list.append(result['detections'])
            features_list.append(result['detection_features'])

        if tensors:
            # Stack tensors
            tensor_stack = torch.stack(tensors)
            torch.save(tensor_stack, output_path / 'vision_tensors.pt')

        # Save metadata
        metadata = {
            'detections': detections_list,
            'features': features_list,
            'num_samples': len(results),
            'config': self.vision_config
        }

        import json
        with open(output_path / 'vision_metadata.json', 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            json_metadata = json.loads(json.dumps(metadata, default=str))
            json.dump(json_metadata, f, indent=2)

        self.logger.info(f"Saved vision data to {output_path}")

    def load_processed_data(self, input_dir: str) -> Tuple[torch.Tensor, List[Dict], List[Dict]]:
        """
        Load processed vision data.

        Args:
            input_dir: Input directory

        Returns:
            Tuple of (tensors, detections, features)
        """
        input_path = Path(input_dir)

        # Load tensors
        tensor_file = input_path / 'vision_tensors.pt'
        if tensor_file.exists():
            tensors = torch.load(tensor_file)
        else:
            tensors = None

        # Load metadata
        metadata_file = input_path / 'vision_metadata.json'
        if metadata_file.exists():
            import json
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            detections = metadata['detections']
            features = metadata['features']
        else:
            detections = []
            features = []

        return tensors, detections, features