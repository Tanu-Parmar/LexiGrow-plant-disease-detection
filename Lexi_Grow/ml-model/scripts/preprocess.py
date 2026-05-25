import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split

IMG_SIZE = 128
DATASET_PATH = "../../database/"

def load_data():
    images = []
    labels = []
    classes = sorted(os.listdir(DATASET_PATH))

    print("Classes:", classes)

    for label_index, class_name in enumerate(classes):
        class_folder = os.path.join(DATASET_PATH, class_name)

        for img_name in os.listdir(class_folder):
            img_path = os.path.join(class_folder, img_name)

            img = cv2.imread(img_path)
            if img is None:
                continue

            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

            images.append(img)
            labels.append(label_index)

    images = np.array(images) / 255.0
    labels = np.array(labels)

    x_train, x_test, y_train, y_test = train_test_split(
        images, labels, test_size=0.2, random_state=42
    )

    return x_train, x_test, y_train, y_test, classes
