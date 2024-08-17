import cv2
import numpy as np
import pickle
from skimage.feature import hog
from pymongo import MongoClient
from config import PROTOTXT_PATH, MODEL_PATH

class FaceDetector:
    def __init__(self):
        self.net = cv2.dnn.readNetFromCaffe(PROTOTXT_PATH, MODEL_PATH)
        print("FaceDetector initialisé")

    def detect_faces(self, image: np.ndarray) -> list:
        print("Détection des visages...")
        (h, w) = image.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        self.net.setInput(blob)
        detections = self.net.forward()
        boxes = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                boxes.append((startX, startY, endX, endY))
        return boxes

class FaceRecognizer:
    def __init__(self, known_faces: dict, svm_model_path: str, detector: FaceDetector, mongo_client: MongoClient):
        print("Initialisation du FaceRecognizer")
        self.known_encodings = known_faces['encodings']
        self.known_names = known_faces['names']
        with open(svm_model_path, 'rb') as f:
            self.svm = pickle.load(f)
        self.detector = detector
        self.expected_feature_size = self.known_encodings.shape[1]
        self.mongo_client = mongo_client
        self.collection = self.mongo_client['facial_recognition']['users']
        print(f"Encodages connus : {self.known_encodings.shape}")
        print(f"Noms connus : {self.known_names}")

    def recognize_faces(self, image: np.ndarray) -> list:
        print("Reconnaissance des visages...")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.detector.detect_faces(image)
        names = []
        for (startX, startY, endX, endY) in faces:
            face = image[startY:endY, startX:endX]
            face_resized = cv2.resize(face, (64, 128))
            face_encoding = self.encode_face(face_resized)
            if face_encoding is not None:
                print(f"Dimensions des caractéristiques HOG : {face_encoding.shape}")  # Affichage des dimensions
                if face_encoding.shape[0] == self.expected_feature_size:
                    face_encoding = np.reshape(face_encoding, (1, -1))
                    predictions = self.svm.predict_proba(face_encoding)
                    print(f"Prédictions SVM : {predictions}")  # Affichage des prédictions
                    if predictions.size > 0:
                        best_match_index = np.argmax(predictions)
                        if predictions[0][best_match_index] > 0.3:
                            name = self.known_names[best_match_index]
                            user_info = self.collection.find_one({"name": name})
                            if user_info:
                                additional_info = user_info.get("additional_info", "Aucune info supplémentaire")
                                print(f"Informations utilisateur pour {name} : {additional_info}")  # Affichage des infos utilisateur
                            else:
                                additional_info = "Aucune info supplémentaire"
                        else:
                            name = "Unknown"
                            additional_info = "Aucune info supplémentaire"
                    else:
                        name = "Unknown"
                        additional_info = "Aucune info supplémentaire"
                    names.append((name, additional_info, (startX, startY, endX - startX, endY - startY)))
                else:
                    names.append(("Unknown", "Aucune info supplémentaire", (startX, startY, endX - startX, endY - startY)))
            else:
                names.append(("Unknown", "Aucune info supplémentaire", (startX, startY, endX - startX, endY - startY)))
        return names
    def encode_face(self, face_image: np.ndarray) -> np.ndarray:
        try:
            hog_features, _ = hog(cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY),
                                pixels_per_cell=(8, 8),
                                cells_per_block=(2, 2),
                                visualize=True,
                                block_norm='L2-Hys',
                                feature_vector=True)
            # Redimensionnez les caractéristiques si nécessaire
            expected_size = 24 * 128  # Ajustez si nécessaire
            if hog_features.shape[0] != expected_size:
                hog_features = np.resize(hog_features, (expected_size,))
            print(f"Dimensions des caractéristiques HOG : {hog_features.shape}")
            return hog_features
        except Exception as e:
            print(f"Erreur lors de l'encodage de la face : {e}")
            return None




def start_inference(detector: FaceDetector, recognizer: FaceRecognizer):
    cap = cv2.VideoCapture(0)
    print("Démarrage de l'inférence...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        try:
            results = recognizer.recognize_faces(frame)
            for (name, additional_info, (startX, startY, w, h)) in results:
                cv2.rectangle(frame, (startX, startY), (startX + w, startY + h), (0, 255, 0), 2)
                cv2.putText(frame, f"{name}: {additional_info}", (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
        except Exception as e:
            print(f"Erreur lors de la reconnaissance : {e}")
        cv2.imshow("Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
