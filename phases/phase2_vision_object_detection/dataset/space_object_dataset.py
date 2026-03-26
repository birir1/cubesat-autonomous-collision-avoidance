# space_object_dataset.py
"""
Space Object Detection Dataset

Loads CubeSat space imagery datasets for training object detection models.

Supports:
- Satellite detection
- Space debris detection
- Unknown object detection

Compatible with:
- YOLO training
- PyTorch DataLoader
- EfficientDet
"""

import os
import cv2
import torch
import numpy as np
import pandas as pd

from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


# ---------------------------------------------------------
# Class Mapping
# ---------------------------------------------------------

CLASS_MAP = {
    "satellite": 0,
    "debris": 1,
    "unknown": 2
}

NUM_CLASSES = len(CLASS_MAP)


# ---------------------------------------------------------
# Dataset Class
# ---------------------------------------------------------

class SpaceObjectDataset(Dataset):

    def __init__(
        self,
        image_dir,
        annotation_file,
        img_size=640,
        augment=True
    ):
        """
        Parameters

        image_dir : directory containing images
        annotation_file : CSV or JSON annotations
        img_size : input image size
        augment : enable augmentations
        """

        self.image_dir = image_dir
        self.annotations = pd.read_csv(annotation_file)
        self.img_size = img_size
        self.augment = augment

        self.image_ids = self.annotations["image_id"].unique()

        self.transforms = self._build_transforms()

    # -----------------------------------------------------

    def _build_transforms(self):

        transforms = []

        if self.augment:
            transforms.extend([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                A.GaussNoise(p=0.2),
                A.MotionBlur(p=0.1),
                A.Rotate(limit=10, p=0.3),
            ])

        transforms.extend([
            A.Resize(self.img_size, self.img_size),
            A.Normalize(),
            ToTensorV2()
        ])

        return A.Compose(
            transforms,
            bbox_params=A.BboxParams(
                format="pascal_voc",
                label_fields=["labels"]
            )
        )

    # -----------------------------------------------------

    def __len__(self):
        return len(self.image_ids)

    # -----------------------------------------------------

    def __getitem__(self, idx):

        image_id = self.image_ids[idx]

        records = self.annotations[
            self.annotations["image_id"] == image_id
        ]

        image_path = os.path.join(self.image_dir, image_id)

        image = cv2.imread(image_path)

        if image is None:
            raise ValueError(f"Image not found: {image_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        boxes = []
        labels = []

        for _, row in records.iterrows():

            xmin = row["xmin"]
            ymin = row["ymin"]
            xmax = row["xmax"]
            ymax = row["ymax"]

            boxes.append([xmin, ymin, xmax, ymax])

            labels.append(CLASS_MAP[row["class"]])

        boxes = np.array(boxes)
        labels = np.array(labels)

        transformed = self.transforms(
            image=image,
            bboxes=boxes,
            labels=labels
        )

        image = transformed["image"]
        boxes = torch.tensor(transformed["bboxes"], dtype=torch.float32)
        labels = torch.tensor(transformed["labels"], dtype=torch.long)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx])
        }

        return image, target


# ---------------------------------------------------------
# Collate Function
# ---------------------------------------------------------

def collate_fn(batch):
    """
    Required for object detection DataLoader.
    """

    images = []
    targets = []

    for img, tgt in batch:
        images.append(img)
        targets.append(tgt)

    return images, targets


# ---------------------------------------------------------
# DataLoader Builder
# ---------------------------------------------------------

def build_dataloader(
    image_dir,
    annotation_file,
    batch_size=8,
    shuffle=True,
    num_workers=4
):

    dataset = SpaceObjectDataset(
        image_dir=image_dir,
        annotation_file=annotation_file
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn
    )

    return loader


# ---------------------------------------------------------
# Dataset Statistics
# ---------------------------------------------------------

def compute_dataset_statistics(annotation_file):

    df = pd.read_csv(annotation_file)

    stats = df["class"].value_counts()

    print("Dataset Distribution")

    for cls, count in stats.items():
        print(f"{cls} : {count}")

    return stats