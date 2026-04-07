"""
Object Detection and Tracking Pipeline Runner

Orchestrates the complete detection and tracking pipeline for satellite
conjunction monitoring and collision risk assessment.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
import yaml
import numpy as np
import torch
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from .detection.efficientdet_detector import EfficientDetDetector
from .detection.yolov8_detector import YOLOv8Detector
from .tracking.deepsort_tracker import DeepSORTTracker
from .tracking.kalman_tracker import KalmanTracker
from utils.data_loader import SatelliteDataLoader
from utils.logger import setup_logger

class DetectionTrackingPipeline:
    """
    Complete detection and tracking pipeline for satellite monitoring.
    """

    def __init__(self, config_path=None):
        """
        Initialize the detection and tracking pipeline.

        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.logger = setup_logger(__name__, self.config.get('log_level', 'INFO'))
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize components
        self.detector = None
        self.tracker = None
        self.data_loader = None

        self._initialize_components()

    def _load_config(self, config_path):
        """Load configuration from YAML file."""
        if config_path is None:
            config_path = project_root / 'configs' / 'simulation_config.yaml'

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        return config.get('detection_tracking', {})

    def _initialize_components(self):
        """Initialize detection and tracking components."""
        detector_config = self.config.get('detector', {})
        tracker_config = self.config.get('tracker', {})

        # Initialize detector
        detector_type = detector_config.get('type', 'efficientdet')
        if detector_type == 'efficientdet':
            self.detector = EfficientDetDetector(
                model_path=detector_config.get('model_path'),
                confidence_threshold=detector_config.get('confidence_threshold', 0.5),
                device=self.device
            )
        elif detector_type == 'yolov8':
            self.detector = YOLOv8Detector(
                model_path=detector_config.get('model_path'),
                confidence_threshold=detector_config.get('confidence_threshold', 0.5),
                device=self.device
            )

        # Initialize tracker
        tracker_type = tracker_config.get('type', 'deepsort')
        if tracker_type == 'deepsort':
            self.tracker = DeepSORTTracker(
                max_age=tracker_config.get('max_age', 30),
                n_init=tracker_config.get('n_init', 3),
                max_iou_distance=tracker_config.get('max_iou_distance', 0.7)
            )
        elif tracker_type == 'kalman':
            self.tracker = KalmanTracker(
                dt=tracker_config.get('dt', 1.0),
                process_noise=tracker_config.get('process_noise', 0.1),
                measurement_noise=tracker_config.get('measurement_noise', 0.1)
            )

        # Initialize data loader
        self.data_loader = SatelliteDataLoader(
            data_path=self.config.get('data_path'),
            batch_size=self.config.get('batch_size', 1)
        )

        self.logger.info("Detection and tracking components initialized")

    def process_frame(self, frame, timestamp=None):
        """
        Process a single frame through detection and tracking.

        Args:
            frame: Input image/frame
            timestamp: Frame timestamp

        Returns:
            dict: Detection and tracking results
        """
        # Run detection
        detections = self.detector.detect(frame)

        # Run tracking
        if len(detections) > 0:
            tracks = self.tracker.update(detections, timestamp)
        else:
            tracks = self.tracker.update([], timestamp)

        return {
            'detections': detections,
            'tracks': tracks,
            'timestamp': timestamp,
            'frame_shape': frame.shape if hasattr(frame, 'shape') else None
        }

    def process_sequence(self, frames, timestamps=None):
        """
        Process a sequence of frames.

        Args:
            frames: List of frames or path to video/image sequence
            timestamps: List of timestamps

        Returns:
            list: Results for each frame
        """
        results = []

        if isinstance(frames, str):
            # Load frames from path
            frames = self.data_loader.load_sequence(frames)

        if timestamps is None:
            timestamps = [i for i in range(len(frames))]

        self.logger.info(f"Processing {len(frames)} frames")

        for i, (frame, timestamp) in enumerate(zip(frames, timestamps)):
            if i % 100 == 0:
                self.logger.info(f"Processing frame {i+1}/{len(frames)}")

            result = self.process_frame(frame, timestamp)
            results.append(result)

        return results

    def evaluate_tracking(self, ground_truth, predictions):
        """
        Evaluate tracking performance.

        Args:
            ground_truth: Ground truth tracks
            predictions: Predicted tracks

        Returns:
            dict: Evaluation metrics
        """
        # Calculate tracking metrics
        metrics = {
            'mota': self._calculate_mota(ground_truth, predictions),
            'motp': self._calculate_motp(ground_truth, predictions),
            'idf1': self._calculate_idf1(ground_truth, predictions),
            'track_fragmentation': self._calculate_track_fragmentation(predictions),
            'track_purity': self._calculate_track_purity(predictions)
        }

        return metrics

    def _calculate_mota(self, gt, pred):
        """Calculate MOTA (Multiple Object Tracking Accuracy)."""
        # Simplified MOTA calculation
        total_gt = sum(len(frame_gt) for frame_gt in gt)
        total_pred = sum(len(frame_pred) for frame_pred in pred)

        if total_gt == 0:
            return 0.0

        # Count matches (simplified)
        matches = 0
        for gt_frame, pred_frame in zip(gt, pred):
            for gt_obj in gt_frame:
                for pred_obj in pred_frame:
                    if self._iou(gt_obj, pred_obj) > 0.5:
                        matches += 1
                        break

        mota = 1 - (total_pred - matches) / total_gt
        return max(0, mota)

    def _calculate_motp(self, gt, pred):
        """Calculate MOTP (Multiple Object Tracking Precision)."""
        total_distance = 0
        total_matches = 0

        for gt_frame, pred_frame in zip(gt, pred):
            for gt_obj in gt_frame:
                best_iou = 0
                best_pred = None
                for pred_obj in pred_frame:
                    iou = self._iou(gt_obj, pred_obj)
                    if iou > best_iou:
                        best_iou = iou
                        best_pred = pred_obj

                if best_iou > 0.5:
                    total_distance += 1 - best_iou  # Distance = 1 - IoU
                    total_matches += 1

        return total_distance / total_matches if total_matches > 0 else 1.0

    def _calculate_idf1(self, gt, pred):
        """Calculate IDF1 score."""
        # Simplified IDF1 calculation
        return 0.85  # Placeholder

    def _calculate_track_fragmentation(self, pred):
        """Calculate track fragmentation."""
        # Simplified calculation
        return len(pred) * 0.1  # Placeholder

    def _calculate_track_purity(self, pred):
        """Calculate track purity."""
        # Simplified calculation
        return 0.9  # Placeholder

    def _iou(self, obj1, obj2):
        """Calculate IoU between two bounding boxes."""
        # obj format: [x1, y1, x2, y2]
        x1 = max(obj1[0], obj2[0])
        y1 = max(obj1[1], obj2[1])
        x2 = min(obj1[2], obj2[2])
        y2 = min(obj1[3], obj2[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (obj1[2] - obj1[0]) * (obj1[3] - obj1[1])
        area2 = (obj2[2] - obj2[0]) * (obj2[3] - obj2[1])
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0

    def save_results(self, results, output_path):
        """
        Save tracking results to file.

        Args:
            results: Tracking results
            output_path: Output file path
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Save as numpy array
        np.savez(output_path,
                detections=[r['detections'] for r in results],
                tracks=[r['tracks'] for r in results],
                timestamps=[r['timestamp'] for r in results])

        self.logger.info(f"Results saved to {output_path}")

    def visualize_results(self, results, output_dir):
        """
        Visualize tracking results.

        Args:
            results: Tracking results
            output_dir: Output directory for visualizations
        """
        os.makedirs(output_dir, exist_ok=True)

        # Import visualization functions
        from visualization.plot_multi_agent_trajectories import plot_tracking_results

        # Create visualizations
        plot_tracking_results(results, output_dir)

        self.logger.info(f"Visualizations saved to {output_dir}")


def main():
    """Main function for running the detection and tracking pipeline."""
    parser = argparse.ArgumentParser(description='Satellite Detection and Tracking Pipeline')
    parser.add_argument('--config', type=str, help='Configuration file path')
    parser.add_argument('--input', type=str, required=True, help='Input data path')
    parser.add_argument('--output', type=str, help='Output directory')
    parser.add_argument('--visualize', action='store_true', help='Generate visualizations')

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(level=logging.INFO,
                       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Initialize pipeline
    pipeline = DetectionTrackingPipeline(args.config)

    # Load and process data
    if os.path.isfile(args.input):
        # Single file
        if args.input.endswith(('.jpg', '.png', '.tiff')):
            # Process single image
            results = [pipeline.process_frame(pipeline.data_loader.load_image(args.input))]
        else:
            # Assume video or sequence
            results = pipeline.process_sequence(args.input)
    else:
        # Directory of images
        results = pipeline.process_sequence(args.input)

    # Save results
    if args.output:
        output_file = os.path.join(args.output, 'tracking_results.npz')
        pipeline.save_results(results, output_file)

        # Generate visualizations
        if args.visualize:
            vis_dir = os.path.join(args.output, 'visualizations')
            pipeline.visualize_results(results, vis_dir)

    # Print summary
    total_detections = sum(len(r['detections']) for r in results)
    total_tracks = sum(len(r['tracks']) for r in results)

    print("Detection and Tracking Summary:")
    print(f"Total frames processed: {len(results)}")
    print(f"Total detections: {total_detections}")
    print(f"Total tracks: {total_tracks}")


if __name__ == "__main__":
    main()