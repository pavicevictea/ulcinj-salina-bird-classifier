import os
import shutil
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from config import DATASET_PATH, TEST_DATA_PATH, IMG_SIZE

def load_images_from_folder(base_path):
    images = []
    labels = []
    class_names = sorted([d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))])
    class_to_idx = {name: i for i, name in enumerate(class_names)}

    for bird_class in class_names:
        class_dir = os.path.join(base_path, bird_class)
        for img_name in sorted(os.listdir(class_dir)):
            img_path = os.path.join(class_dir, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            images.append(img)
            labels.append(class_to_idx[bird_class])
    return np.array(images), np.array(labels), class_names

def copy_test_images(idx_test, all_image_paths):
    if os.path.exists(TEST_DATA_PATH):
        return
    for idx in idx_test:
        bird_class, img_name = all_image_paths[idx]
        dest_dir = os.path.join(TEST_DATA_PATH, bird_class)
        os.makedirs(dest_dir, exist_ok=True)
        shutil.copy(
            os.path.join(DATASET_PATH, bird_class, img_name),
            os.path.join(dest_dir, img_name)
        )

def prepare_and_split():
    X, y, class_names = load_images_from_folder(DATASET_PATH)
    all_image_paths = []
    for bird_class in class_names:
        class_dir = os.path.join(DATASET_PATH, bird_class)
        for img_name in sorted(os.listdir(class_dir)):
            all_image_paths.append((bird_class, img_name))

    X_train_val, X_test, y_train_val, y_test, idx_train_val, idx_test = train_test_split(X, y, np.arange(len(y)), test_size=0.15, stratify=y, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.15 / 0.85, stratify=y_train_val, random_state=42)
    copy_test_images(idx_test, all_image_paths)
    return X_train, X_val, X_test, y_train, y_val, y_test, class_names

def load_test_data():
    X, y, class_names = load_images_from_folder(TEST_DATA_PATH)
    return X, y, class_names