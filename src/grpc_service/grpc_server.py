import os
import logging
import grpc
from concurrent import futures
import time
from iot_lib.stopwatch import StopWatch
from core.detect_controller import DetectController
from core.train_controller2 import TrainController
from protos import svm_pb2_grpc, svm_pb2
_ONE_DAY_IN_SECONDS = 60 * 60 * 24

class SVMTrainingServer(svm_pb2_grpc.SVMTrainingServerServicer):

    def __init__(self, cfg):
        self._logger = logging.getLogger('training grpc-server')
        # 训练
        self.train_ctrler = TrainController(cfg)

    def train(self, request: svm_pb2.TrainRequest, context):
        self._logger.info('gRPC@train receive request')
        log_cat = f"({self.train.__name__})"

        # check folder image
        sub_folders = [f.path for f in os.scandir(request.data_path) if f.is_dir()]
        for subfolder in sub_folders:
            files = os.listdir(subfolder)
            if not files:
                msg = 'invalid image data'
                self._logger.warning(f"{log_cat} {msg}")
                return svm_pb2.TrainResponse(status=False, message=msg, model_id='model is none', model_url='none')

        try:
            stopwatch = StopWatch()
            stopwatch.start()
            model_id, model_url, model, extra_param= self.train_ctrler(request.data_path, request.split_ratio)
            stopwatch.stop()
            self._logger.info(f"{log_cat} using {stopwatch.elapsed_time}s")
            self._logger.info(f"{log_cat} model_id: {model_id}")
            # return train response
            return svm_pb2.TrainResponse(status=True, message='train success', model_id=model_id, model_url=model_url)


        except Exception as e:
            self._logger.exception(f"{log_cat} fail: {e}")
            return svm_pb2.TrainResponse(status=False, message=f"{e}")


class SVMDetectionServer(svm_pb2_grpc.SVMDetectionServerServicer):
    def __init__(self, modpath):
        self._logger = logging.getLogger('detection grpc-server')
        self.detect_ctrler = DetectController(model_path=modpath)

    def detect(self, request: svm_pb2.DetectRequest, context):
        self._logger.info('gRPC@detect receive request')
        log_cat = f"({self.detect.__name__})"

        if request.image_url:
            # 接收图片时间
            self._logger.info(f"{log_cat} request time: {time.time()}")
        try:
            stopwatch = StopWatch()
            stopwatch.start()
            res = self.detect_ctrler(request.image_url)
            stopwatch.stop()
            self._logger.info(f"{log_cat} using {stopwatch.elapsed_time}s")
            self._logger.info(f"{log_cat} result: {res}")
            return svm_pb2.DetectResponse(status=True, message='detection success', result=res)

        except Exception as e:

            self._logger.exception(f"{log_cat} fail: {e}")
            return svm_pb2.DetectResponse(status=False, message=f"{e}")


def run(cfg, modpath):

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    svm_pb2_grpc.add_SVMTrainingServerServicer_to_server(SVMTrainingServer(cfg),server)
    svm_pb2_grpc.add_SVMDetectionServerServicer_to_server(SVMDetectionServer(modpath),server)
    print("start service...")
    server.add_insecure_port('[::]:50052')
    server.start()

    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)
        print("stop service...")


if __name__ == '__main__':
    run()
