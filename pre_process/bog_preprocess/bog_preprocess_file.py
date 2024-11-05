import os
import cv2
import numpy as np


def process_folder(input_folder, output_folder):
    # 遍历输入文件夹下的子文件夹和文件
    for root, dirs, files in os.walk(input_folder):
        # 创建相应的输出文件夹结构
        relative_path = os.path.relpath(root, input_folder)
        output_subfolder = os.path.join(output_folder, relative_path)
        os.makedirs(output_subfolder, exist_ok=True)

        # 遍历文件夹中的文件
        for file in files:
            # 跳过处理 '.DS_Store' 文件
            if file == '.DS_Store':
                continue

            # 获取文件路径
            input_path = os.path.join(root, file)
            output_path = os.path.join(output_subfolder, file)

            # 读取图像
            image = cv2.imread(input_path)

            # 进行图像处理与操作
            image = cv2.GaussianBlur(image, (5, 5), 0)
            edges = cv2.Canny(image, 50, 150, apertureSize=3)

            circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1, minDist=100, param1=50, param2=30, minRadius=0, maxRadius=0)
            if circles is not None:
                circles = np.uint16(np.around(circles))
                distances = np.sqrt(np.power(circles[0][:, 0] - image.shape[1] / 2, 2) + np.power(circles[0][:, 1] - image.shape[0] / 2, 2))
                min_distance_idx = np.argmin(distances)
                max_circle = circles[0][min_distance_idx]  # 提取最中心的圆
                center = (max_circle[0], max_circle[1])
                radius = max_circle[2]

                # 创建空白图像并抠取中心圆部分
                mask = np.zeros_like(image)
                cv2.circle(mask, center, radius, (255, 255, 255), -1)
                masked_image = cv2.bitwise_and(image, mask)

                # 保存处理后的图像
                cv2.imwrite(output_path, masked_image)
            else:
                # 如果未检测到圆，则直接复制原始图像
                cv2.imwrite(output_path, image)


if __name__ == '__main__':

    """ 处理整个文件夹图片 """

    input_folder = '/Users/hanbinliu/Desktop/缺陷检测data/visual_detection_images/博格华纳'
    output_folder = '/Users/hanbinliu/Desktop/bog'

    # 调用函数处理文件夹中的图像
    process_folder(input_folder, output_folder)