# deep_sort_tracker.py
"""
Deep SORT Tracker for Space Object Tracking

Combines:
- Kalman motion model
- Appearance embeddings
- Hungarian matching

Used after YOLO detections to produce consistent object IDs
and motion estimates for trajectory prediction.
"""

import numpy as np
import torch
import torch.nn as nn
import cv2
from scipy.optimize import linear_sum_assignment

from phases.phase3_object_tracking.models.kalman_tracker import KalmanTrack


# -------------------------------------------------------
# Appearance Embedding Network
# -------------------------------------------------------

class AppearanceEncoder(nn.Module):
    """
    Lightweight CNN for appearance feature extraction
    used in object re-identification.
    """

    def __init__(self, embedding_dim=128):
        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((1,1))
        )

        self.fc = nn.Linear(128, embedding_dim)

    def forward(self, x):

        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        x = torch.nn.functional.normalize(x, dim=1)

        return x


# -------------------------------------------------------
# Track Object
# -------------------------------------------------------

class DeepSortTrack(KalmanTrack):

    def __init__(self, bbox, track_id, embedding):

        super().__init__(bbox, track_id)

        self.embedding = embedding
        self.embedding_history = [embedding]

    def update_embedding(self, embedding):

        self.embedding_history.append(embedding)

        if len(self.embedding_history) > 30:
            self.embedding_history.pop(0)

        self.embedding = np.mean(self.embedding_history, axis=0)


# -------------------------------------------------------
# Cosine Distance
# -------------------------------------------------------

def cosine_distance(a, b):

    return 1 - np.dot(a, b) / (
        np.linalg.norm(a) * np.linalg.norm(b)
    )


# -------------------------------------------------------
# Deep SORT Tracker
# -------------------------------------------------------

class DeepSortTracker:

    def __init__(
        self,
        max_missed=15,
        iou_weight=0.4,
        appearance_weight=0.6,
        device=None
    ):

        self.tracks = []
        self.next_id = 0

        self.max_missed = max_missed

        self.iou_weight = iou_weight
        self.appearance_weight = appearance_weight

        self.device = device if device else (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.encoder = AppearanceEncoder().to(self.device)
        self.encoder.eval()

    # ---------------------------------------------------

    def extract_embedding(self, frame, bbox):

        x1, y1, x2, y2 = map(int, bbox)

        crop = frame[y1:y2, x1:x2]

        if crop.size == 0:
            return np.zeros(128)

        crop = cv2.resize(crop, (64,64))

        crop = crop.astype(np.float32) / 255.0

        crop = torch.tensor(crop).permute(2,0,1).unsqueeze(0)

        crop = crop.to(self.device)

        with torch.no_grad():
            emb = self.encoder(crop).cpu().numpy()[0]

        return emb

    # ---------------------------------------------------

    def compute_cost_matrix(self, detections, embeddings):

        n_tracks = len(self.tracks)
        n_dets = len(detections)

        cost_matrix = np.zeros((n_tracks, n_dets))

        for t, track in enumerate(self.tracks):

            for d, det in enumerate(detections):

                bbox = det["bbox"]

                # IOU similarity
                iou_score = self.compute_iou(
                    track.last_bbox,
                    bbox
                )

                iou_cost = 1 - iou_score

                # Appearance similarity
                appearance_cost = cosine_distance(
                    track.embedding,
                    embeddings[d]
                )

                cost = (
                    self.iou_weight * iou_cost +
                    self.appearance_weight * appearance_cost
                )

                cost_matrix[t, d] = cost

        return cost_matrix

    # ---------------------------------------------------

    def compute_iou(self, boxA, boxB):

        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        inter = max(0, xB-xA) * max(0, yB-yA)

        areaA = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
        areaB = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])

        union = areaA + areaB - inter

        if union == 0:
            return 0

        return inter / union

    # ---------------------------------------------------

    def update(self, frame, detections):

        embeddings = [
            self.extract_embedding(frame, det["bbox"])
            for det in detections
        ]

        for track in self.tracks:
            track.predict()

        if len(self.tracks) == 0:

            for det, emb in zip(detections, embeddings):
                self._start_track(det["bbox"], emb)

            return self.tracks

        cost_matrix = self.compute_cost_matrix(
            detections,
            embeddings
        )

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        assigned_tracks = set()
        assigned_dets = set()

        for r, c in zip(row_ind, col_ind):

            if cost_matrix[r, c] > 0.7:
                continue

            self.tracks[r].update(detections[c]["bbox"])
            self.tracks[r].update_embedding(embeddings[c])

            assigned_tracks.add(r)
            assigned_dets.add(c)

        # missed tracks
        for i, track in enumerate(self.tracks):

            if i not in assigned_tracks:
                track.mark_missed()

        # new tracks
        for d, det in enumerate(detections):

            if d not in assigned_dets:
                self._start_track(det["bbox"], embeddings[d])

        # remove lost tracks
        self.tracks = [
            t for t in self.tracks
            if t.missed < self.max_missed
        ]

        return self.tracks

    # ---------------------------------------------------

    def _start_track(self, bbox, embedding):

        track = DeepSortTrack(
            bbox=bbox,
            track_id=self.next_id,
            embedding=embedding
        )

        self.next_id += 1

        self.tracks.append(track)

    # ---------------------------------------------------

    def get_active_tracks(self):

        results = []

        for t in self.tracks:

            results.append({
                "id": t.track_id,
                "position": t.get_position(),
                "velocity": t.get_velocity()
            })

        return results