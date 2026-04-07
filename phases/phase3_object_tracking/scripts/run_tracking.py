"""
Run object tracking with either Kalman or DeepSORT-inspired trackers.
"""

import argparse
import json
import os

from phases.phase3_object_tracking.models.kalman_tracker import KalmanTracker
from phases.phase3_object_tracking.models.deep_sort_tracker import DeepSORTTracker


def parse_args():
    parser = argparse.ArgumentParser(description='Run object tracking for phase 3.')
    parser.add_argument('--tracker', choices=['kalman', 'deepsort'], default='kalman', help='Tracker type to use')
    parser.add_argument('--detections', type=str, default='', help='Path to JSON detections file')
    parser.add_argument('--output', type=str, default='tracking_output.json', help='Output file for tracks')
    return parser.parse_args()


def load_detections(path):
    if not path or not os.path.exists(path):
        return []
    with open(path, 'r') as handle:
        return json.load(handle)


def main():
    args = parse_args()
    detections = load_detections(args.detections)

    if args.tracker == 'deepsort':
        tracker = DeepSORTTracker()
    else:
        tracker = KalmanTracker()

    tracks = tracker.track(detections)

    with open(args.output, 'w') as handle:
        json.dump({'tracks': tracks}, handle, indent=2)

    print(f"Saved tracking output to {args.output}")


if __name__ == '__main__':
    main()
