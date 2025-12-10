import cv2
import numpy as np
import os
import pickle
import urllib.request
from sklearn.neighbors import KNeighborsClassifier
import datetime
import pandas as pd

class FaceSystem:
    def __init__(self):
        self.detector = None
        self.recognizer = None
        self.classifier = None
        self.labels = []
        self.embeddings = []
        self.label_map = {}  # Maps integer label to name
        self.model_trained = False
        
        # Model paths
        self.det_model_path = "face_detection_yunet_2023mar.onnx"
        self.rec_model_path = "face_recognition_sface_2021dec.onnx"
        self.classifier_path = "face_classifier.pkl"
        self.data_path = "face_data.pkl"
        
        self._initialize_models()
        self._load_data()

    def _initialize_models(self):
        # Download YuNet if missing
        if not os.path.exists(self.det_model_path):
            print("Downloading YuNet...")
            url = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
            urllib.request.urlretrieve(url, self.det_model_path)

        # Download SFace if missing
        if not os.path.exists(self.rec_model_path):
            print("Downloading SFace...")
            url = "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"
            urllib.request.urlretrieve(url, self.rec_model_path)

        # Initialize Detector
        self.detector = cv2.FaceDetectorYN.create(
            model=self.det_model_path,
            config="",
            input_size=(320, 320),
            score_threshold=0.8,
            nms_threshold=0.3,
            top_k=5000
        )

        # Initialize Recognizer (SFace)
        self.recognizer = cv2.FaceRecognizerSF.create(
            model=self.rec_model_path,
            config=""
        )

    def _load_data(self):
        if os.path.exists(self.data_path):
            with open(self.data_path, 'rb') as f:
                data = pickle.load(f)
                self.embeddings = data.get('embeddings', [])
                self.labels = data.get('labels', [])
                self.label_map = data.get('label_map', {})
                print(f"Loaded {len(self.embeddings)} embeddings for {len(self.label_map)} users.")
        
        if os.path.exists(self.classifier_path):
            with open(self.classifier_path, 'rb') as f:
                self.classifier = pickle.load(f)
                self.model_trained = True
                print("Loaded trained classifier.")

    def save_data(self):
        data = {
            'embeddings': self.embeddings,
            'labels': self.labels,
            'label_map': self.label_map
        }
        with open(self.data_path, 'wb') as f:
            pickle.dump(data, f)
        print("Data saved.")

    def detect_faces(self, image):
        h, w, _ = image.shape
        self.detector.setInputSize((w, h))
        _, faces = self.detector.detect(image)
        return faces if faces is not None else []

    def get_embedding(self, image, face_box):
        # Align and crop face
        aligned_face = self.recognizer.alignCrop(image, face_box)
        # Extract feature
        feature = self.recognizer.feature(aligned_face)
        return feature[0]

    def register_face(self, image, name):
        faces = self.detect_faces(image)
        if len(faces) == 0:
            return False, "No face detected"
        if len(faces) > 1:
            return False, "Multiple faces detected"

        # Get embedding
        embedding = self.get_embedding(image, faces[0])
        
        # Check if face is already registered (Duplicate Check)
        if self.model_trained:
            # Find nearest neighbor
            distances, indices = self.classifier.kneighbors([embedding], n_neighbors=1)
            min_dist = distances[0][0]
            if min_dist < 0.4: # Threshold for "Same Person"
                existing_label = self.labels[indices[0][0]]
                existing_name = self.label_map.get(existing_label, "Unknown")
                # If the name is different, block it
                if existing_name != name:
                    return False, f"Face already registered as {existing_name}"
        
        # Assign label
        if name not in self.label_map.values():
            new_label = len(self.label_map)
            self.label_map[new_label] = name
        
        # Find label ID for name
        label_id = [k for k, v in self.label_map.items() if v == name][0]

        self.embeddings.append(embedding)
        self.labels.append(label_id)
        
        return True, f"Captured for {name}"

    def train_classifier(self):
        if len(self.embeddings) < 2:
            return False, "Not enough data to train (need at least 2 samples)"
        
        print("Training classifier...")
        # KNN Classifier
        self.classifier = KNeighborsClassifier(n_neighbors=3, metric='cosine')
        self.classifier.fit(self.embeddings, self.labels)
        
        with open(self.classifier_path, 'wb') as f:
            pickle.dump(self.classifier, f)
        
        self.model_trained = True
        self.save_data()
        return True, "Model trained successfully"

    def recognize(self, image):
        faces = self.detect_faces(image)
        results = []
        
        for face in faces:
            box = list(map(int, face[:4]))
            score = face[-1]
            name = "Unknown"
            confidence = 0.0
            
            if self.model_trained:
                embedding = self.get_embedding(image, face)
                
                # Use distance for better "Unknown" handling
                # kneighbors returns (distances, indices)
                distances, indices = self.classifier.kneighbors([embedding], n_neighbors=1)
                min_dist = distances[0][0]
                
                # Cosine distance: 0 = identical, 1 = opposite
                # Threshold: 0.4 is a good starting point for SFace
                if min_dist < 0.4: 
                    label_id = self.labels[indices[0][0]]
                    name = self.label_map.get(label_id, "Unknown")
                    confidence = 1.0 - min_dist # Convert distance to "confidence"
                else:
                    name = "Unknown"
                    confidence = min_dist # Just for debug/display
            
            results.append({
                'box': box,
                'score': score,
                'name': name,
                'confidence': confidence
            })
            
        return results



    def log_attendance(self, name, mode):
        # mode: 'IN' or 'OUT'
        csv_file = "attendance_log.csv"
        now = datetime.datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H:%M:%S")
        
        # Ensure file exists with correct columns
        expected_columns = ["Name", "Date", "InTime", "OutTime"]
        
        # Check if file exists and is not empty
        if not os.path.exists(csv_file) or os.path.getsize(csv_file) == 0:
            df = pd.DataFrame(columns=expected_columns)
            df.to_csv(csv_file, index=False)
        else:
            # Check if columns match, if not, recreate (simple migration)
            try:
                df = pd.read_csv(csv_file)
                if not all(col in df.columns for col in expected_columns):
                    print("Schema mismatch, recreating log file...")
                    df = pd.DataFrame(columns=expected_columns)
                    df.to_csv(csv_file, index=False)
            except pd.errors.EmptyDataError:
                # Handle case where file exists but is corrupted/empty
                df = pd.DataFrame(columns=expected_columns)
                df.to_csv(csv_file, index=False)
            
        df = pd.read_csv(csv_file)
        
        # Find today's record for this person
        # We look for a row with Name and Date
        mask = (df['Name'] == name) & (df['Date'] == date_str)
        
        if df[mask].empty:
            # New record for today
            if mode == 'IN':
                new_row = pd.DataFrame([{"Name": name, "Date": date_str, "InTime": time_str, "OutTime": ""}])
                df = pd.concat([df, new_row], ignore_index=True)
                print(f"Marked IN for {name}")
        else:
            # Update existing record
            idx = df[mask].index[0]
            if mode == 'IN':
                # Only update if empty (don't overwrite first entry)
                if pd.isna(df.at[idx, 'InTime']) or df.at[idx, 'InTime'] == "":
                    df.at[idx, 'InTime'] = time_str
                    print(f"Marked IN for {name}")
            elif mode == 'OUT':
                # Always update OutTime to the latest
                df.at[idx, 'OutTime'] = time_str
                # Throttle print: Only print if we haven't printed for this user recently?
                # Actually, for OUT, we are updating continuously. Let's just NOT print every time.
                # We can print only if the minute changed, or just silence it.
                # User asked to stop "repeatedly says marked out".
                # Let's check if the stored OutTime minute is different from current minute
                # current_out = df.at[idx, 'OutTime']
                # if current_out and current_out[:5] != time_str[:5]:
                #     print(f"Updated OUT for {name} at {time_str}")
                pass # Silencing the spam completely for OUT updates
                
        df.to_csv(csv_file, index=False)
