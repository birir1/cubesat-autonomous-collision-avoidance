"""
Phase 3: Detection and Tracking

Implements satellite object detection and multi-object tracking
for space situational awareness and collision avoidance.
"""

from .run_tracking import DetectionTrackingPipeline
from .detection import EfficientDetDetector, YOLOv8Detector
from .tracking import DeepSORTTracker, KalmanTracker

__all__ = [
    'DetectionTrackingPipeline',
    'EfficientDetDetector',
    'YOLOv8Detector',
    'DeepSORTTracker',
    'KalmanTracker'
]