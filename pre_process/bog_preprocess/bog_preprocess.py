# -*- coding: utf-8 -*-
# @Time    : 6/5/23 3:43 下午
# @Author  : Hanbin Liu

import cv2
import numpy as np

path = '/Users/hanbinliu/Desktop/缺陷检测data/visual_detection_images/博格华纳/abnormal/Image_20230518164708640.jpg'
image = cv2.imread(path)

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
    print(center, radius)

# 创建一个与原图像大小和通道数相同的空白图像
mask = np.zeros_like(image)
cv2.circle(mask, center, radius, (255, 255, 255), -1)
masked_image = cv2.bitwise_and(image, mask)
cv2.imwrite('/Users/hanbinliu/Desktop/abnormal.png', masked_image)


cv2.circle(image, center, radius, (0, 255, 0), 2)
cv2.imshow('Detected Circle', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
