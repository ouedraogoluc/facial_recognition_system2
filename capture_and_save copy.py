import cv2
import os
from datetime import datetime
from pymongo import MongoClient

# Configuration de MongoDB
MONGO_URI = "mongodb://localhost:27017/"
DATABASE_NAME = "facial_recognition"
COLLECTION_NAME = "users"

client = MongoClient(MONGO_URI)
db = client[DATABASE_NAME]
collection = db[COLLECTION_NAME]

class Camera:
    def __init__(self, save_dir='D:/Projet/detection/facial_recognition_system2/src/data/raw'):
        self.save_dir = save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.cam = cv2.VideoCapture(0)
        if not self.cam.isOpened():
            raise Exception("Erreur : La caméra ne peut pas être ouverte.")

    def capture_images(self, name, surname, num_photos=10):
        photo_count = 0
        while True:
            ret, frame = self.cam.read()
            if ret:
                cv2.imshow("Camera Feed", frame)
                key = cv2.waitKey(1) & 0xFF
                
                # Quitter si 'q' est pressé
                if key == ord('q'):
                    print("Capture terminée par l'utilisateur.")
                    break

                if photo_count < num_photos:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"{name}_{surname}_{timestamp}_{photo_count + 1}.jpg"
                    file_path = os.path.join(self.save_dir, filename)
                    cv2.imwrite(file_path, frame)
                    print(f"Image {photo_count + 1}/{num_photos} sauvegardée à {file_path}")

                    # Enregistrement dans MongoDB
                    document = {
                        "name": name,
                        "surname": surname,
                        "photo_path": file_path,
                        "timestamp": datetime.now()
                    }
                    collection.insert_one(document)
                    
                    photo_count += 1
                else:
                    print("Nombre de photos atteint.")
                    break
            else:
                print("Échec de la capture de l'image.")
                break

        cv2.destroyAllWindows()

    def close(self):
        if hasattr(self, 'cam'):
            self.cam.release()
        cv2.destroyAllWindows()

    def __del__(self):
        self.close()
