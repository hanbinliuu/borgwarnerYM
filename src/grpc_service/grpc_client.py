import logging
import cv2
import grpc
import os
from protos import svm_pb2, svm_pb2_grpc


def read_bytes_img(img_path: str) -> bytes:
    img = cv2.imread(img_path)
    # 图像编码
    success, img_encoded = cv2.imencode('.jpg', img)
    # 图像转换为bytes
    img_encoded_bytes = img_encoded.tobytes()
    return img_encoded_bytes


def run_train(file_path, ip):
    with grpc.insecure_channel(ip) as channel:
        stub = svm_pb2_grpc.SVMTrainingServerStub(channel)
        request = svm_pb2.TrainRequest(data_path=file_path, split_ratio=0.3)
        rep = stub.train(request)
        print(rep)


def run_detect(image_path, ip):
    with grpc.insecure_channel(ip) as channel:
        stub = svm_pb2_grpc.SVMDetectionServerStub(channel)
        request = svm_pb2.DetectRequest(image_url=image_path)
        rep = stub.detect(request)
        print(rep.result)


if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)

    # train
    filePath = r'D:\ym\borgwarner\borg_tg\tg_data_train\train'
    ip = "192.168.6.152:50050"
    run_train(filePath, ip)

    # detection
    ip = "192.168.6.152:7777"
    # ip = "0.0.0.0:7777"
    folder_url = "D:/ym/borgwarner/borgtest/"
    files = os.listdir(folder_url)
    for file in files:
        image_url = folder_url + file
        run_detect(image_url, ip)
