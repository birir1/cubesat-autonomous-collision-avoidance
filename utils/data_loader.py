import os
import cv2
import numpy as np


def load_space_images(image_folder):

    images = []

    if not os.path.exists(image_folder):
        print("Image folder not found:", image_folder)
        return images

    for file in os.listdir(image_folder):

        if file.endswith(".jpg") or file.endswith(".png"):

            path = os.path.join(image_folder, file)

            img = cv2.imread(path)

            if img is None:
                continue

            img = cv2.resize(img, (640, 640))

            img = img / 255.0

            images.append(img)

    return np.array(images)