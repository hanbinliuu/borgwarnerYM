import cv2
import os

import joblib
import numpy as np
from sklearn.svm import SVC, OneClassSVM
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import IsolationForest

from src.algo_lib.feature_selection import calculate_image_stats


class DefectDetection:
    def __init__(self):
        self.clf = None

    def get_image_paths(self, directory):
        image_paths = []
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith('.jpg') or file.endswith('.png') or file.endswith('.bmp') or file.endswith('.jpeg'):
                    image_path = os.path.join(root, file)
                    image_paths.append(image_path)
        return image_paths

    def load_image_data(self, image_paths):
        images = []
        for image_path in image_paths:
            image = cv2.imread(image_path)
            if image is not None:
                images.append(image)
        return images

    def extract_color_histogram(self, image):
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv_image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        return hist

    def train_model(self, X, y):
        # two class
        # self.clf = make_pipeline(StandardScaler(), SVC(kernel='rbf', C=1.0))
        # one class
        self.clf = make_pipeline(StandardScaler(), OneClassSVM(nu=0.1, kernel='rbf', gamma='scale'))
        # self.clf = IsolationForest(contamination=0.01, random_state=42)

        self.clf.fit(X, y)

    def predict(self, X):
        return self.clf.predict(X)

    def evaluate(self, y_true, y_pred):
        return accuracy_score(y_true, y_pred)

    def run(self, abnormal_dir, normal_dir, test_size=0.15, random_state=42):
        # 获取异常和正常样本的图像路径
        abnormal_image_paths = self.get_image_paths(abnormal_dir)
        normal_image_paths = self.get_image_paths(normal_dir)

        # 加载异常和正常样本的图像数据
        abnormal_images = self.load_image_data(abnormal_image_paths)
        normal_images = self.load_image_data(normal_image_paths)

        # 创建标签
        abnormal_labels = np.ones(len(abnormal_images))
        normal_labels = np.zeros(len(normal_images))

        # 将异常和正常样本合并为一个数据集
        X = np.concatenate((abnormal_images, normal_images))
        y = np.concatenate((abnormal_labels, normal_labels))

        # 提取特征
        X_features = []
        for image in X:
            # features = self.extract_color_histogram(image)
            features = calculate_image_stats(image)
            X_features.append(features)
        X_features = np.array(X_features)

        # 划分数据集为训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(X_features, y, test_size=test_size, random_state=random_state)

        # 训练模型
        self.train_model(X_train, y_train)

        # 保存模型
        # joblib.dump(self.clf, 'svm_model.pkl')

        # 预测
        y_pred = self.predict(X_test)

        # 评估模型性能
        accuracy = self.evaluate(y_test, y_pred)
        print("Accuracy:", accuracy)


if __name__ == '__main__':
    # defect_detection = DefectDetection()
    # defect_detection.run('C:/Users/lhb/Desktop/abnormal2', 'C:/Users/lhb/Desktop/normal2')

    model = joblib.load('svm_model.pkl')
    img_predict = cv2.imread(r"C:\Users\lhb\Desktop\testing\20231204184833373.jpeg")
    features = calculate_image_stats(img_predict).reshape(1, -1)
    print(model.predict(features))