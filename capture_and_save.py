import cv2
import os
from datetime import datetime
from pymongo import MongoClient, errors
import tkinter as tk
from tkinter import simpledialog, messagebox

# Configuration MongoDB
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
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Capture terminée par l'utilisateur.")
                    break

                if photo_count < num_photos:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"{name}_{surname}_{timestamp}_{photo_count + 1}.jpg"
                    file_path = os.path.join(self.save_dir, filename)
                    cv2.imwrite(file_path, frame)
                    print(f"Image {photo_count + 1}/{num_photos} sauvegardée à {file_path}")

                    document = {
                        "name": name,
                        "surname": surname,
                        "photo_path": file_path,
                        "timestamp": datetime.now()
                    }
                    try:
                        result = collection.insert_one(document)
                        print(f"Document inséré avec l'ID : {result.inserted_id}")
                    except errors.PyMongoError as e:
                        print(f"Erreur lors de l'insertion dans MongoDB: {e}")

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

def show_form():
    root = tk.Tk()
    root.withdraw()

    name = simpledialog.askstring("Input", "Entrez votre prénom:")
    surname = simpledialog.askstring("Input", "Entrez votre nom:")
    num_photos = simpledialog.askinteger("Input", "Nombre de photos à prendre:", initialvalue=10)

    if not name or not surname or num_photos is None or num_photos <= 0:
        messagebox.showwarning("Input Error", "Le prénom, le nom et un nombre valide de photos sont nécessaires.")
        return

    try:
        camera = Camera()
        camera.capture_images(name, surname, num_photos)
        messagebox.showinfo("Success", "Les images ont été capturées et sauvegardées dans MongoDB.")
    except Exception as e:
        messagebox.showerror("Error", f"Une erreur est survenue: {e}")

if __name__ == "__main__":
    show_form()
