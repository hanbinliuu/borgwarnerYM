import os
import uuid
import cv2
import random
import shutil
import logging
import pickle

from tqdm import tqdm
from src.algo_lib.bog_preprocess import ImageProcessor
from src.algo_lib.svm_classification import DefectDetection
from src.config_service.config import get_config


class TrainController:

    def __init__(self, cfg):

        self._logger = logging.getLogger('train controller')
        # config
        self._config_path = cfg
        self._config = get_config(cfg)

        # parameter
        self.dataset = None
        self.filepath = None
        self.split_ratio = 0.3

        # dataset path
        self.normal_dir_path = None
        self.abnormal_dir_path = None
        self.test_normal_dir_path = None
        self.test_abnormal_dir_path = None

    def __call__(self, file_path, split_ratio):

        files = os.listdir(r'D:\pycharmProject\borgwarner\results')
        while len(files) != 0:
            model_url = r'D:\pycharmProject\borgwarner\results\svm_model_test.pkl'
            self._logger.info('model trained in the results folder')
            with open(model_url, 'rb') as file:
                loaded_data = pickle.load(file)
            model_id, model = loaded_data['model_id'], loaded_data['model']
            return model_id, model, model_url, 'params'

        if len(files) == 0:

            self._logger.info('model not trained in the results folder')
            self.filepath = file_path
            self.split_ratio = split_ratio
            self._preprocess()  # 预处理

            # train
            self.detect_defect = DefectDetection(self.abnormal_dir_path, self.normal_dir_path)
            model = self.detect_defect.run()
            self.save_model_to_file(model, r'D:\pycharmProject\borgwarner\results\svm_model_test.pkl')
            # 用uuid随机生成一个模型id
            model_id = str(uuid.uuid4())
            self._logger.info('model id: {}'.format(model_id))

            data = {'model_id': model_id, 'model': model}
            model_url = r'D:\pycharmProject\borgwarner\results\svm_model_test.pkl'
            self.save_model_to_file(data, model_url)
            return model_id, model_url, model, 'params'  # 参数备用

    # 先处理，再分train和test集
    def _preprocess(self):

        dataset_path = self._config.dataset.path
        self._check_dir_path(dataset_path)

        # normal dir path
        self.normal_dir_path = os.path.join(dataset_path, self._config.dataset.normal_dir)
        self._check_dir_path(self.normal_dir_path)

        # abnormal dir path
        self.abnormal_dir_path = os.path.join(dataset_path, self._config.dataset.abnormal_dir)
        self._check_dir_path(self.abnormal_dir_path)

        # 遍历输入文件夹下的子文件夹和文件
        for root, dirs, files in os.walk(self.filepath):
            # 创建相应的输出文件夹结构
            relative_path = os.path.relpath(root, self.filepath)
            output_subfolder = os.path.join(dataset_path, relative_path)
            os.makedirs(output_subfolder, exist_ok=True)

            # 遍历文件夹中的文件
            for file in files:

                # 获取文件路径
                input_path = os.path.join(root, file)
                output_path = os.path.join(output_subfolder, file)

                # 读取图像
                image = cv2.imread(input_path)
                image_process = ImageProcessor(image)
                masked_image = image_process.process_image()
                # 保存处理后的图像
                cv2.imwrite(output_path, masked_image)

        # split image
        self.test_normal_dir_path = os.path.join(dataset_path, self._config.dataset.test_normal_dir)
        self._check_dir_path(self.test_normal_dir_path)
        self._split_data(self.normal_dir_path, self.test_normal_dir_path)

        self.test_abnormal_dir_path = os.path.join(dataset_path, self._config.dataset.test_abnormal_dir)
        self._check_dir_path(self.test_abnormal_dir_path)
        self._split_data(self.abnormal_dir_path, self.test_abnormal_dir_path)

    def _check_dir_path(self, path):
        if os.path.exists(path):
            self._logger.info(f"{path} exists, skip create")
        else:
            self._logger.info(f"{path} not exists, create")
            os.makedirs(path)

    def _split_data(self, dir_path, output_path):

        # 获取所有图片的文件名
        files = [f for f in os.listdir(dir_path) if f.endswith(".jpg") or f.endswith(".jpeg") or f.endswith(".bmp")]
        # 随机选择图片
        random_files = random.sample(files, k=int(len(files) * self.split_ratio))

        # 移动图片到新文件夹
        for file in random_files:
            shutil.move(os.path.join(dir_path, file), output_path)

    @staticmethod
    def save_model_to_file(model, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump(model, f)


if __name__ == '__main__':
    filePath = r'D:\ym\borgwarner\borg_tg\tg_data_train\train'
    cfg = r'D:\pycharmProject\borgwarner\src\config_service\config_train.yaml'
    ctrler = TrainController(cfg)
    result = ctrler(filePath, 0.3)

    print(result)

    temp = 1
