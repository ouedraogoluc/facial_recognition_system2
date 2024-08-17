import os
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D # type: ignore
from face_encoder import FaceEncoder  # Importez votre classe FaceEncoder

# Chemins des fichiers
IMAGES_PATH = "D:/Projet/detection/facial_recognition_system2/src/data/processed/images.npy"
LABELS_PATH = "D:/Projet/detection/facial_recognition_system2/src/data/processed/labels.npy"
ENCODINGS_PATH = "D:/Projet/detection/facial_recognition_system2/src/data/processed/encodings.npy"
NAMES_PATH = "D:/Projet/detection/facial_recognition_system2/src/data/processed/names.npy"
IMAGES_PREPROCESSED_PATH = "D:/Projet/detection/facial_recognition_system2/src/data/processed/images_preprocessed.npy"
LABELS_PREPROCESSED_PATH = "D:/Projet/detection/facial_recognition_system2/src/data/processed/labels_preprocessed.npy"

def load_data(images_path, labels_path):
    if not os.path.exists(images_path) or not os.path.exists(labels_path):
        print(f"Erreur : Les fichiers {images_path} ou {labels_path} n'existent pas.")
        return None, None
    images = np.load(images_path)
    labels = np.load(labels_path)
    return images, labels

def preprocess_data():
    print("Prétraitement des données...")
    images, labels = load_data(IMAGES_PATH, LABELS_PATH)
    
    if images is None or labels is None:
        raise FileNotFoundError("Les données d'entrée sont manquantes.")
    
    images = images / 255.0  # Normalisation des images
    np.save(IMAGES_PREPROCESSED_PATH, images)
    np.save(LABELS_PREPROCESSED_PATH, labels)
    print("Prétraitement terminé.")
    return images, labels

def encode_faces():
    print("Encodage des visages...")
    encoder = FaceEncoder(IMAGES_PREPROCESSED_PATH, LABELS_PREPROCESSED_PATH, ENCODINGS_PATH, NAMES_PATH)
    encoder.process()
    print("Encodage terminé.")

def train_logistic_regression(images, labels):
    X = images.reshape(images.shape[0], -1)
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    model.fit(X, labels)
    print("Régression Logistique entraînée avec succès.")
    return model

def train_decision_tree(images, labels):
    X = images.reshape(images.shape[0], -1)
    model = DecisionTreeClassifier()
    model.fit(X, labels)
    print("Arbre de Décision entraîné avec succès.")
    return model

def train_random_forest(images, labels):
    X = images.reshape(images.shape[0], -1)
    model = RandomForestClassifier()
    model.fit(X, labels)
    print("Forêt Aléatoire entraînée avec succès.")
    return model

def train_knn(images, labels):
    X = images.reshape(images.shape[0], -1)
    model = KNeighborsClassifier()
    model.fit(X, labels)
    print("k-Nearest Neighbors entraîné avec succès.")
    return model

def train_cnn(images, labels):
    images = images.astype('float32') / 255.0
    labels = np.array(labels)

    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(images.shape[1], images.shape[2], images.shape[3])),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(len(np.unique(labels)), activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(images, labels, epochs=10, validation_split=0.2)

    print("Réseau de Neurones Convolutifs entraîné avec succès.")
    return model

def train_svm(images, labels):
    X = images.reshape(images.shape[0], -1)
    model = SVC(probability=True)  # 'probability=True' pour activer predict_proba
    model.fit(X, labels)
    print("SVM entraîné avec succès.")
    return model  # Correction de 'mode' à 'model'


def save_model(model, filename):
    if isinstance(model, tf.keras.Model):
        model.save(filename)  # Sauvegarde le modèle Keras
    else:
        with open(filename, 'wb') as file:
            pickle.dump(model, file)

def preprocess_and_generate():
    # Prétraitement des données
    images, labels = preprocess_data()
    
    # Encodage des visages
    encode_faces()
    
    # Entraînement des modèles
    logistic_regression_model = train_logistic_regression(images, labels)
    save_model(logistic_regression_model, "D:/Projet/detection/facial_recognition_system2/src/model/logistic_regression_model.pkl")

    decision_tree_model = train_decision_tree(images, labels)
    save_model(decision_tree_model, "D:/Projet/detection/facial_recognition_system2/src/model/decision_tree_model.pkl")

    random_forest_model = train_random_forest(images, labels)
    save_model(random_forest_model, "D:/Projet/detection/facial_recognition_system2/src/model/random_forest_model.pkl")

    knn_model = train_knn(images, labels)
    save_model(knn_model, "D:/Projet/detection/facial_recognition_system2/src/model/knn_model.pkl")

    cnn_model = train_cnn(images, labels)
    save_model(cnn_model, "D:/Projet/detection/facial_recognition_system2/src/model/cnn_model.h5")

    svm_model = train_svm(images, labels)
    save_model(svm_model, "D:/Projet/detection/facial_recognition_system2/src/model/svm_model.pkl")
