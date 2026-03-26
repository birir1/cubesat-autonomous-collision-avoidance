# kalman_tracker.py
"""
Kalman Filter Multi-Object Tracker

Tracks detected satellites/debris across frames and estimates
object motion (position + velocity).

State Model (constant velocity):

state = [x, y, vx, vy]

Used by:
- trajectory prediction
- orbital state estimation
- collision risk assessment
"""

import numpy as np
from scipy.optimize import linear_sum_assignment


# --------------------------------------------------------
# Kalman Filter for a single track
# --------------------------------------------------------

class KalmanTrack:

    def __init__(self, bbox, track_id):

        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2

        # State vector
        self.x = np.array([
            cx,
            cy,
            0.0,
            0.0
        ])

        # Covariance
        self.P = np.eye(4) * 100

        # Motion model
        self.F = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        # Measurement model
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])

        self.R = np.eye(2) * 5
        self.Q = np.eye(4) * 0.01

        self.track_id = track_id
        self.age = 0
        self.missed = 0

        self.last_bbox = bbox

    # ----------------------------------------------------

    def predict(self):

        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

        self.age += 1

    # ----------------------------------------------------

    def update(self, bbox):

        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2

        z = np.array([cx, cy])

        y = z - (self.H @ self.x)

        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y

        I = np.eye(4)
        self.P = (I - K @ self.H) @ self.P

        self.last_bbox = bbox
        self.missed = 0

    # ----------------------------------------------------

    def mark_missed(self):
        self.missed += 1

    # ----------------------------------------------------

    def get_position(self):
        return self.x[:2]

    # ----------------------------------------------------

    def get_velocity(self):
        return self.x[2:]


# --------------------------------------------------------
# IOU Computation
# --------------------------------------------------------

def compute_iou(boxA, boxB):

    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_area = max(0, xB - xA) * max(0, yB - yA)

    boxA_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxB_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    union = boxA_area + boxB_area - inter_area

    if union == 0:
        return 0

    return inter_area / union


# --------------------------------------------------------
# Tracker Manager
# --------------------------------------------------------

class MultiObjectTracker:

    def __init__(
        self,
        max_missed=10,
        iou_threshold=0.3
    ):

        self.tracks = []
        self.next_id = 0
        self.max_missed = max_missed
        self.iou_threshold = iou_threshold

    # ----------------------------------------------------

    def update(self, detections):

        for track in self.tracks:
            track.predict()

        if len(self.tracks) == 0:

            for det in detections:
                self._start_track(det["bbox"])

            return self.tracks

        cost_matrix = np.zeros((len(self.tracks), len(detections)))

        for t, track in enumerate(self.tracks):
            for d, det in enumerate(detections):

                cost_matrix[t, d] = 1 - compute_iou(
                    track.last_bbox,
                    det["bbox"]
                )

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        assigned_tracks = set()
        assigned_dets = set()

        for r, c in zip(row_ind, col_ind):

            if cost_matrix[r, c] > 1 - self.iou_threshold:
                continue

            self.tracks[r].update(detections[c]["bbox"])

            assigned_tracks.add(r)
            assigned_dets.add(c)

        for t, track in enumerate(self.tracks):

            if t not in assigned_tracks:
                track.mark_missed()

        for d, det in enumerate(detections):

            if d not in assigned_dets:
                self._start_track(det["bbox"])

        self.tracks = [
            t for t in self.tracks
            if t.missed < self.max_missed
        ]

        return self.tracks

    # ----------------------------------------------------

    def _start_track(self, bbox):

        track = KalmanTrack(
            bbox=bbox,
            track_id=self.next_id
        )

        self.next_id += 1
        self.tracks.append(track)

    # ----------------------------------------------------

    def get_active_tracks(self):

        return [
            {
                "id": t.track_id,
                "position": t.get_position(),
                "velocity": t.get_velocity()
            }
            for t in self.tracks
        ]