import os
import cv2 # type: ignore
import numpy as np # type: ignore
import pickle
from sklearn.preprocessing import LabelEncoder # type: ignore

# def load_images(image_dir):
#     images = []
#     labels = []
#     for filename in os.listdir(image_dir):
#         if filename.endswith(".jpg") or filename.endswith(".png"):
#             image_path = os.path.join(image_dir, filename)
#             image = cv2.imread(image_path)
#             if image is not None:
#                 images.append(image)
#                 labels.append(filename.split('_')[0])  # Exemple d'étiquetage par le prénom
#     return images, labels

IMAGE_DIR = '/data/raw/'
PROCESSED_DIR = '/data/processed/'
if not os.path.exists(PROCESSED_DIR):
    os.makedirs(PROCESSED_DIR)

def load_images(image_dir):
    images = []
    labels = []
    for filename in os.listdir(image_dir):
        if filename.endswith('.jpg'):
            label = filename.split('_')[0]
            image_path = os.path.join(image_dir, filename)
            img = cv2.imread(image_path)
            if img is not None:
                images.append(img)
                labels.append(label)
            else:
                print(f"Image non chargée: {filename}")
    return images, labels



def preprocess_images(images):
    processed_images = []
    for img in images:
        img = cv2.resize(img, (128, 64))
        img = img / 255.0
        img = (img * 255).astype(np.uint8)
        processed_images.append(img)
    return np.array(processed_images)

def encode_labels(labels):
    le = LabelEncoder()
    encoded_labels = le.fit_transform(labels)
    with open(os.path.join(PROCESSED_DIR, 'label_encoder.pkl'), 'wb') as f:
        pickle.dump(le, f)
    return encoded_labels


# def preprocess_images(images):
#     processed_images = []
#     for image in images:
#         processed_image = cv2.resize(image, (100, 100))  # Redimensionner pour simplification
#         processed_images.append(processed_image)
#     return np.array(processed_images)

# def encode_labels(labels):
#     le = LabelEncoder()
#     encoded_labels = le.fit_transform(labels)
#     with open('D:/Projet/detection/facial_recognition_system2/src/data/processed/label_encoder.pkl', 'wb') as f:
#         pickle.dump(le, f)
#     return encoded_labels
