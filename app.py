import tkinter as tk
from tkinter import simpledialog, messagebox
from capture_and_save import Camera
from preprocess import load_data, preprocess_data, encode_faces, save_model, train_logistic_regression, train_decision_tree, train_random_forest, train_knn, train_cnn, train_svm
from real_time_recognition import FaceDetector, start_inference, FaceRecognizer
import numpy as np
import os
from pymongo import MongoClient

# Configuration MongoDB
MONGO_URI = "mongodb://localhost:27017/"
client = MongoClient(MONGO_URI)

# Répertoire pour sauvegarder les modèles
MODEL_DIR = "D:/Projet/detection/facial_recognition_system2/src/model/"

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition System")

        # Initialisation des boutons
        tk.Button(root, text="Capture Images", command=self.capture_images).pack()
        tk.Button(root, text="Encode Faces", command=self.encode_faces).pack()
        tk.Button(root, text="Preprocess Data", command=self.preprocess_data).pack()
        tk.Button(root, text="Train Models", command=self.train_models).pack()
        tk.Button(root, text="Start Inference", command=self.start_inference).pack()

        self.camera = Camera()

    def capture_images(self):
        try:
            name = simpledialog.askstring("Input", "Entrez le prénom:")
            surname = simpledialog.askstring("Input", "Entrez le nom:")
            num_photos = simpledialog.askinteger("Input", "Nombre de photos à capturer:", initialvalue=10)

            if not name or not surname:
                messagebox.showwarning("Input Error", "Le prénom et le nom ne peuvent pas être vides.")
                return

            if num_photos is None or num_photos <= 0:
                messagebox.showwarning("Input Error", "Le nombre de photos doit être un entier positif.")
                return

            self.camera.capture_images(name, surname, num_photos)
            messagebox.showinfo("Capture Terminée", "Les images ont été capturées avec succès.")

        except Exception as e:
            messagebox.showerror("Capture Error", f"Erreur lors de la capture des images: {e}")

    def encode_faces(self):
        try:
            encode_faces()  # Assurez-vous que cette fonction est définie dans preprocess.py
            messagebox.showinfo("Encoding Terminé", "Les visages ont été encodés avec succès.")
        except Exception as e:
            messagebox.showerror("Encoding Error", f"Erreur lors de l'encodage des visages: {e}")

    def preprocess_data(self):
        try:
            preprocess_data()
            messagebox.showinfo("Prétraitement Terminé", "Le prétraitement des données est terminé. Vous pouvez maintenant entraîner le modèle.")
        except Exception as e:
            messagebox.showerror("Preprocessing Error", f"Erreur lors du prétraitement des données: {e}")

    def train_models(self):
        try:
            images, labels = load_data("D:/Projet/detection/facial_recognition_system2/src/data/processed/images_preprocessed.npy",
                                       "D:/Projet/detection/facial_recognition_system2/src/data/processed/labels_preprocessed.npy")
            
            if images is None or labels is None:
                messagebox.showwarning("Data Error", "Les données prétraitées sont manquantes ou endommagées.")
                return

            # Entraînement et sauvegarde des modèles
            logistic_regression_model = train_logistic_regression(images, labels)
            save_model(logistic_regression_model, os.path.join(MODEL_DIR, "logistic_regression_model.pkl"))

            decision_tree_model = train_decision_tree(images, labels)
            save_model(decision_tree_model, os.path.join(MODEL_DIR, "decision_tree_model.pkl"))

            random_forest_model = train_random_forest(images, labels)
            save_model(random_forest_model, os.path.join(MODEL_DIR, "random_forest_model.pkl"))

            knn_model = train_knn(images, labels)
            save_model(knn_model, os.path.join(MODEL_DIR, "knn_model.pkl"))

            cnn_model = train_cnn(images, labels)
            save_model(cnn_model, os.path.join(MODEL_DIR, "cnn_model.h5"))

            svm_model = train_svm(images, labels)  # Assurez-vous que cette fonction est définie dans preprocess.py
            save_model(svm_model, os.path.join(MODEL_DIR, "svm_model.pkl"))

            messagebox.showinfo("Entraînement Terminé", "Les modèles ont été entraînés et sauvegardés.")
        
        except Exception as e:
            messagebox.showerror("Training Error", f"Erreur lors de l'entraînement des modèles: {e}")

    def start_inference(self):
        try:
            detector = FaceDetector()
            known_faces = {
                'encodings': np.load('D:/Projet/detection/facial_recognition_system2/src/data/processed/encodings.npy'),
                'names': np.load('D:/Projet/detection/facial_recognition_system2/src/data/processed/names.npy')
            }
            recognizer = FaceRecognizer(known_faces, 'D:/Projet/detection/facial_recognition_system2/src/model/svm_model.pkl', detector, client)
            start_inference(detector, recognizer)
        except Exception as e:
            messagebox.showerror("Inference Error", f"Erreur lors de l'inférence en temps réel: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
