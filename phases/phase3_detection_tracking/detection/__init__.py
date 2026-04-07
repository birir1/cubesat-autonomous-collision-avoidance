"""
Detection Module

Provides object detection capabilities for satellite imagery
and space situational awareness data.
"""

from .efficientdet_detector import EfficientDetDetector, SatelliteDetectionDataset
from .yolov8_detector import YOLOv8Detector, SatelliteDetectionTrainer

__all__ = [
    'EfficientDetDetector',
    'SatelliteDetectionDataset',
    'YOLOv8Detector',
    'SatelliteDetectionTrainer'
]