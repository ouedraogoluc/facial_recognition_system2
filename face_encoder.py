import numpy as np
import face_recognition  # type: ignore
from PIL import Image
import os
import matplotlib.pyplot as plt

class FaceEncoder:
    def __init__(self, images_path, labels_path, encodings_path, names_path):
        self.images_path = images_path
        self.labels_path = labels_path
        self.encodings_path = encodings_path
        self.names_path = names_path

    def load_data(self):
        """Charge les images et les labels depuis les fichiers .npy."""
        if not os.path.exists(self.images_path) or not os.path.exists(self.labels_path):
            raise FileNotFoundError(f"Les fichiers {self.images_path} ou {self.labels_path} n'existent pas.")
        
        images = np.load(self.images_path)
        labels = np.load(self.labels_path)
        print(f"Chargé {len(images)} images et {len(labels)} labels.")
        return images, labels

    def convert_to_rgb(self, image):
        """Convertit une image en format RGB si nécessaire."""
        if len(image.shape) == 2 or image.shape[2] != 3:
            image = Image.fromarray(image).convert('RGB')
            image = np.array(image)
        return image

    def display_image(self, image, index, faces=[]):
        """Affiche une image avec des visages détectés pour débogage."""
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        plt.title(f'Image {index}')
        plt.axis('off')
        
        for (top, right, bottom, left) in faces:
            plt.gca().add_patch(plt.Rectangle((left, top), right - left, bottom - top, edgecolor='red', facecolor='none'))

        plt.show()

    def generate_encodings(self, images, labels):
        """Génère les encodages de visages à partir des images et des labels."""
        print("Génération des encodages...")
        encodings = []
        names = []

        for i, image in enumerate(images):
            name = labels[i]
            image = self.convert_to_rgb(image)  # Assurez-vous que l'image est en RGB

            # Détection des visages
            face_locations = face_recognition.face_locations(image)
            if face_locations:
                print(f"{len(face_locations)} visage(s) détecté(s) dans l'image {i} ({name})")
            else:
                print(f"Aucune face détectée dans l'image {i} ({name})")

            # Affichage de l'image avec les visages détectés pour débogage
            self.display_image(image, i, face_locations)

            try:
                # Encodage du visage
                encodings_faces = face_recognition.face_encodings(image)
                for encoding in encodings_faces:
                    encodings.append(encoding)
                    names.append(name)
                    print(f"Encodage généré pour l'image {i} ({name})")
            except Exception as e:
                print(f"Erreur lors de l'encodage de l'image {i} ({name}) : {e}")

        print(f"Nombre total d'encodages générés : {len(encodings)}")

        # Sauvegarde des encodages et des noms
        try:
            if encodings and names:
                np.save(self.encodings_path, np.array(encodings))
                np.save(self.names_path, np.array(names))
                print("Encodages générés et sauvegardés.")
            else:
                print("Aucun encodage généré.")
        except Exception as e:
            print(f"Erreur lors de la sauvegarde des encodages : {e}")

        return encodings, names

    def process(self):
        """Exécute le processus de génération des encodages."""
        try:
            images, labels = self.load_data()
            if images is not None and labels is not None:
                self.generate_encodings(images, labels)
        except Exception as e:
            print(f"Erreur lors du traitement : {e}")

# Exemple d'utilisation
if __name__ == "__main__":
    encoder = FaceEncoder(
        "D:/Projet/detection/facial_recognition_system2/src/data/processed/images.npy",
        "D:/Projet/detection/facial_recognition_system2/src/data/processed/labels.npy",
        "D:/Projet/detection/facial_recognition_system2/src/data/processed/encodings.npy",
        "D:/Projet/detection/facial_recognition_system2/src/data/processed/names.npy"
    )
    encoder.process()
