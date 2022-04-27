import os
import pickle
from PIL import Image
import numpy as np
import cv2
from scipy import spatial
import time
import glob
from tqdm import tqdm
import json

try:
    import feature_extractor
    import face_detector
except:
    from src import feature_extractor
    from src import face_detector


class FaceRecognitionSystem:
    def __init__(self, dataset_path, image_folder):
        self.extractor = feature_extractor.InceptionResNetV2_FE()

        self.detector = face_detector.FaceDetector()

        self.dataset_path = dataset_path
        if not os.path.exists(self.dataset_path):
            os.mkdir(self.dataset_path)

        self.image_folder = image_folder

        self.feature_folder_path = dataset_path + 'feature_vectors/'
        if not os.path.exists(self.feature_folder_path):
            os.mkdir(self.feature_folder_path)

    def index_dataset(self):
        for user_folder in os.listdir(self.image_folder):
            vector_file = f'{self.feature_folder_path}/{user_folder}.pkl'
            if os.path.exists(vector_file):
                continue
            else:
                pkl_file = open(vector_file, "wb")
                user_path = os.path.join(self.image_folder, user_folder)
                for img_path in tqdm(os.listdir(user_path)):
                    img_path_full = os.path.join(user_path, img_path)
                    img = Image.open(img_path_full)
                    try:
                        feature_vector = self.extractor.extract(img)
                    except:
                        continue

                    pickle.dump(feature_vector, pkl_file)
                pkl_file.close()
        return 'Done!'

    def verify_face(self, img):
        res_dict = {}

        max_similarity = 0.0
        for vector_file in os.listdir(self.feature_folder_path):
            feature_vectors = pickle.load(open(self.feature_folder_path + vector_file, "rb"))

            if isinstance(img, str):
                img = Image.open(img)

            feature_img = self.extractor.extract(img)

            # calculate the cosine similarity
            cos_sim = 1 - spatial.distance.cosine(feature_vectors, feature_img)

            name = vector_file.split('.')[0]
            temp = {'similary': round(cos_sim, 2)}
            res_dict[name] = temp

            if cos_sim > max_similarity:
                max_similarity = cos_sim
                face_name = name

        if max_similarity < 0.5:
            return 'Unknown'

        return face_name, max_similarity, res_dict

    def recognize_face_via_camera(self):
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            raise IOError("Cannot open webcam")

        prev_frame_time = 0
        new_frame_time = 0

        res_dict = {}

        while True:
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)

            if ret is None:
                continue

            # frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

            bbox = self.detector.detect(frame)
            if bbox is None:
                continue

            x_min, y_min, x_max, y_max = bbox
            processed_frame = Image.fromarray(np.uint8(frame[y_min:y_max, x_min:x_max])).convert('RGB')
            res_frame = cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            result = self.verify_face(processed_frame)
            if result == "Unknown":
                face_name = "Unknown"
                cv2.putText(res_frame, face_name, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                face_name, similarity, res_dict = result
                cv2.putText(res_frame, f'{face_name}: {round(similarity, 2)}', (x_min, y_min - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            new_frame_time = time.time()
            fps = 1 / (new_frame_time - prev_frame_time)
            prev_frame_time = new_frame_time
            fps = str(int(fps * 100))
            cv2.putText(res_frame, fps, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(res_frame, 'FPS', (40, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (100, 255, 0), 1, cv2.LINE_AA)

            y_axis = 1
            for face in res_dict:
                string = f'{face}: {res_dict[face]["similary"]}'
                cv2.putText(res_frame, string, (500, y_axis * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 255, 0), 1,
                            cv2.LINE_AA)
                y_axis += 1

            cv2.imshow("Result", res_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
