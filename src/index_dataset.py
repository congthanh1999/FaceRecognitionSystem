try:
    from src.system import FaceRecognitionSystem
except:
    from system import FaceRecognitionSystem

if __name__ == '__main__':
    dataset_path = 'D:/A jerry/AA lab/CS338.NhanDang/ND_test/datasets/'
    image_folder = 'D:/A jerry/AA lab/CS338.NhanDang/ND_test/datasets/images/'

    my_system = FaceRecognitionSystem(dataset_path, image_folder)

    my_system.index_dataset()
