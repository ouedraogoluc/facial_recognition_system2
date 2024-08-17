import os
import cv2
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder

def load_images(image_dir):
    images = []
    labels = []
    for filename in os.listdir(image_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(image_dir, filename)
            image = cv2.imread(image_path)
            if image is not None:
                images.append(image)
                labels.append(filename.split('_')[0])  # Exemple d'étiquetage
    return images, labels

def preprocess_images(images):
    processed_images = []
    for image in images:
        processed_image = cv2.resize(image, (100, 100))  # Redimensionner pour simplification
        processed_images.append(processed_image)
    return np.array(processed_images)

def encode_labels(labels):
    le = LabelEncoder()
    encoded_labels = le.fit_transform(labels)
    with open('data/processed/label_encoder.pkl', 'wb') as f:
        pickle.dump(le, f)
    return encoded_labels
