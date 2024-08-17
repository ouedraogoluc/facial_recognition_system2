#train_svm.py
import numpy as np
import pickle
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Chemins vers les fichiers de données
ENCODINGS_PATH = "D:/Projet/detection/facial_recognition_system2/src/data/processed/encodings.npy"
NAMES_PATH = "D:/Projet/detection/facial_recognition_system2/src/data/processed/names.npy"
SVM_MODEL_PATH = "D:/Projet/detection/facial_recognition_system2/src/model/svm_model.pkl"

def load_encodings_and_names(encodings_path, names_path):
    """Charge les encodages de visages et les noms depuis les fichiers .npy."""
    try:
        print(f"Chargement des encodages depuis {encodings_path}...")
        encodings = np.load(encodings_path)
        print(f"Chargement des noms depuis {names_path}...")
        names = np.load(names_path)
        print(f"Encodings shape: {encodings.shape}")
        print(f"Names shape: {names.shape}")
        print(f"Unique names: {np.unique(names)}")
        return encodings, names
    except Exception as e:
        print(f"Erreur lors du chargement des données : {e}")
        raise

def check_classes(names):
    """Vérifie le nombre de classes dans les noms."""
    unique_classes = np.unique(names)
    print(f"Classes uniques trouvées : {unique_classes}")
    if len(unique_classes) <= 1:
        raise ValueError(f"Le nombre de classes doit être supérieur à 1. Classes trouvées : {len(unique_classes)}")

def train_svm(encodings, names):
    """Entraîne un modèle SVM avec les encodages de visages et les noms."""
    # Préparation des données
    X = encodings
    y = names
    
    # Vérifiez les classes avant d'entraîner le modèle
    check_classes(y)
    
    # Création du pipeline SVM avec normalisation
    model = make_pipeline(StandardScaler(), svm.SVC(kernel='linear', probability=True))
    
    try:
        model.fit(X, y)
        print("Modèle SVM entraîné avec succès.")
    except Exception as e:
        print(f"Erreur lors de l'entraînement du modèle : {e}")
        raise
    
    return model

def save_model(model, filename):
    """Sauvegarde le modèle SVM dans un fichier .pkl."""
    try:
        print(f"Sauvegarde du modèle dans {filename}...")
        with open(filename, 'wb') as file:
            pickle.dump(model, file)
        print("Modèle sauvegardé.")
    except Exception as e:
        print(f"Erreur lors de la sauvegarde du modèle : {e}")
        raise

def main():
    """Exécute le processus d'entraînement et de sauvegarde du modèle SVM."""
    try:
        # Chargement des encodages et des noms
        encodings, names = load_encodings_and_names(ENCODINGS_PATH, NAMES_PATH)
        
        # Entraînement du modèle SVM
        svm_model = train_svm(encodings, names)
        
        # Sauvegarde du modèle
        save_model(svm_model, SVM_MODEL_PATH)
    except Exception as e:
        print(f"Erreur dans le processus principal : {e}")

if __name__ == "__main__":
    main()
