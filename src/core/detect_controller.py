import os
import logging
import cv2
import pickle
import numpy as np
import requests
from PIL import Image
import io

from src.algo_lib.bog_preprocess import ImageProcessor
from src.algo_lib.feature_selection import extract_color_histogram, calculate_image_stats, calculate_texture_features

class DetectController:


    def __init__(self, model_path):
        self._logger = logging.getLogger('defect controller')
        # 如果results文件夹有文件，读取模型
        # if len(os.listdir('/Users/hanbinliu/PycharmProjects/borgwarner/results')) > 0:
        #     self.model_path = '/Users/hanbinliu/PycharmProjects/borgwarner/results/svm_model_test.pkl'
        # else:
        #     self._logger.info('模型不存在! 需要训练')
        self.model_path = model_path

    def __call__(self, image_url):

        # 读取图片url

        # # 1、使用urllib下载图片
        # response = requests.get(image_url)
        # image_bytes = response.content
        # img = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        # success, image_bytes = cv2.imencode('.jpg', img)

        # 读取模型
        with open(self.model_path, 'rb') as f:
            model_bytes = f.read()

        # 加载模型
        model = load_model_from_bytes(model_bytes)
        mod = model['model']

        # 2、todo bytes to numpy，直接读取本地图片url
        image_bytes = read_bytes_img(image_url)
        frame = cv2.imdecode(np.frombuffer(image_bytes, dtype="uint8"), cv2.IMREAD_COLOR)

        # 预处理+一张张图处理
        process = ImageProcessor(frame)
        preprocess_frame = process.process_image()

        # 保存预处理后的图片
        # cv2.imwrite(r'C:\Users\lhb\Desktop\preprocess_frame.jpg', preprocess_frame)
        X_features = []
        for image in [preprocess_frame]:
            color_features = extract_color_histogram(image)
            stat_features = calculate_image_stats(image)
            # texture_features = calculate_texture_features(image)
            combined_features = np.concatenate((color_features, stat_features))
            X_features.append(combined_features)
        X_features = np.array(X_features)
        predict_result = mod.predict_proba(X_features)
        predict_result = mod.predict(X_features)

        res = int(predict_result)
        return res


def read_bytes_img(img_path: str) -> bytes:
    img = cv2.imread(img_path)
    # 图像编码
    success, img_encoded = cv2.imencode('.jpg', img)
    # 图像转换为bytes
    img_encoded_bytes = img_encoded.tobytes()
    return img_encoded_bytes

def load_model_from_bytes(model_bytes):
    model = pickle.loads(model_bytes)
    return model

def get_file_paths_in_folder(folder_path):
    file_paths = []

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            file_paths.append(file_path)

    return file_paths


if __name__ == '__main__':

    # # http url图片地址
    # imge_url = 'http://192.168.6.37:9750/res/download/image/20231013173939373.jpeg'
    detect = DetectController(r'D:\pycharmProject\borgwarner\results\svm_model_test.pkl')
    # result = detect(imge_url)
    # print(result)


    # 本地图片地址
    image_path = r"C:\Users\lhb\Desktop\20231215-172844314.jpg"
    detect = DetectController(r'D:\pycharmProject\borgwarner\results\svm_model_test.pkl')
    results = detect(image_path)

    # # 整个文件夹图片
    all_url = get_file_paths_in_folder(r'C:\Users\lhb\Desktop\20231026')
    results = []
    for url in all_url:
        result = detect(url)
        if result == 0:
            results.append({url,result})

    print(results)