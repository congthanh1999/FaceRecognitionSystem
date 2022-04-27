import cv2
from facenet_pytorch import MTCNN
import torch
from datetime import datetime
import os


class FaceCapture:
    def __init__(self):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.mtcnn = MTCNN(margin=20, keep_all=False, post_process=False, device=self.device)

    def capture_face(self, dir_images):
        count = 50
        user_name = input("Input ur name: ")

        user_path = os.path.join(dir_images, user_name)
        leap = 1

        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        while cap.isOpened() and count:
            is_success, frame = cap.read()
            frame = cv2.flip(frame, 1)
            if self.mtcnn(frame) is not None and leap % 2:
                path = str(user_path + '/{}.jpg'.format(
                    str(datetime.now())[:-7].replace(":", "-").replace(" ", "-") + str(count)))
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                face_img = self.mtcnn(rgb, save_path=path)
                count -= 1
            leap += 1
            cv2.imshow('Face Capturing', frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
        cap.release()
        cv2.destroyAllWindows()
