import cv2

try:
    from src.system import FaceRecognitionSystem
except:
    from system import FaceRecognitionSystem

if __name__ == '__main__':
    dataset_path = 'datasets/'
    image_folder = 'datasets/images/'

    my_system = FaceRecognitionSystem(dataset_path, image_folder)

    my_system.recognize_face_via_camera()
