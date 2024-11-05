import os
import cv2

import numpy as np

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import pickle

from src.algo_lib.feature_selection import extract_color_histogram, calculate_image_stats


class DefectDetection:

    def __init__(self, abnormal_dir, normal_dir, test_size=0.3):

        self.clf = None
        self.abnormal_dir = abnormal_dir
        self.normal_dir = normal_dir
        self.test_size = test_size
        self.random_state = 42

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

    def train_model(self, X, y):
        self.clf = make_pipeline(StandardScaler(), SVC(kernel='rbf', C=2, probability=True))
        self.clf.fit(X, y)
        return self.clf

    def predict(self, X):
        return self.clf.predict(X)

    def evaluate(self, y_true, y_pred):
        return accuracy_score(y_true, y_pred)

    def run(self):

        # 获取异常和正常样本的图像路径
        abnormal_image_paths = self.get_image_paths(self.abnormal_dir)
        normal_image_paths = self.get_image_paths(self.normal_dir)

        # 加载异常和正常样本的图像数据
        abnormal_images = self.load_image_data(abnormal_image_paths)
        normal_images = self.load_image_data(normal_image_paths)

        # 创建标签
        abnormal_labels = np.zeros(len(abnormal_images))
        normal_labels = np.ones(len(normal_images))

        # 将异常和正常样本合并为一个数据集
        X = np.concatenate((abnormal_images, normal_images))
        y = np.concatenate((abnormal_labels, normal_labels))

        # 提取特征
        X_features = []
        for image in X:
            color_features = extract_color_histogram(image)
            stat_features = calculate_image_stats(image)
            # texture_features = calculate_texture_features(image)
            combined_features = np.concatenate((color_features, stat_features))
            X_features.append(combined_features)
        X_features = np.array(X_features)

        # 划分数据集为训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(X_features, y, test_size=self.test_size,
                                                            random_state=self.random_state)
        # 训练模型
        model = self.train_model(X_train, y_train)

        # 预测
        y_pred = self.predict(X_test)
        # 评估模型性能
        accuracy = self.evaluate(y_test, y_pred)
        print("Accuracy:", accuracy)
        return model
