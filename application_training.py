import os

import grpc
import argparse
import concurrent.futures
import signal
import threading
import contextlib

import frozen_dir
from protos import svm_pb2_grpc
from iot_lib.iot_logger import LoggerConfigurator
from grpc_service.grpc_server import SVMTrainingServer

from config_service.config import get_config


@contextlib.contextmanager
def run_training_server(host, port, cfg_path):
    # create and start grpc servicer
    options = [
        ('grpc.max_send_message_length', 100 * 1024 * 1024),
        ('grpc.max_receive_message_length', 100 * 1024 * 1024)
    ]
    server = grpc.server(concurrent.futures.ThreadPoolExecutor(max_workers=10), options=options)
    algor_servicer = SVMTrainingServer(cfg_path)
    svm_pb2_grpc.add_SVMTrainingServerServicer_to_server(algor_servicer, server)
    boundport = server.add_insecure_port(f"{host}:{port}")
    server.start()

    try:
        yield server, boundport
    finally:
        logger.info(f"stop grpc server at {host}:{boundport}")
        server.stop(0)



def main():
    # get config
    config = get_config(config_path)

    parser = argparse.ArgumentParser()
    parser.add_argument('--host', type=str, default=config.ip, help='grpc server host')
    parser.add_argument('--port', type=str, default=config.port, help='port')
    # parser.add_argument('--cfg_path', type=str, required=True, help='path to the train cfg')
    parser.add_argument('--cfg_path', type=str, default='./src/config_service/config_train.yaml', help='path to the train cfg')

    args = parser.parse_args()
    print(args)

    # stop event: ctrl+c
    stop_event = threading.Event()

    def signal_handler(signum, frame):
        signal.signal(signum, signal.SIG_IGN)
        logger.warning("receive signal to quit")
        stop_event.set()

    # register the signal with the signal handler first
    signal.signal(signal.SIGINT, signal_handler)

    with run_training_server(args.host, args.port, args.cfg_path) as (server, port):
        logger.info(f"grpc Server is listening at port :{port}")
        stop_event.wait()  # instead of server.wait_for_termination()


if __name__ == '__main__':

    # 获取脚本所在的目录
    current_dir = frozen_dir.app_path()

    # 构建相对路径
    config_path = os.path.join(current_dir, 'src', 'config_service', 'config_train.yaml')
    log_file = os.path.join(current_dir, 'logs', 'train_app.log')
    log_settings_file = os.path.join(current_dir, 'settings', 'log_settings.json')
    logger = LoggerConfigurator(fname=log_settings_file, handlerFileName=log_file).get_logger(__name__)
    try:
        main()
    except Exception as ex:
        logger.exception("exception raised")