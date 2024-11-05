import os
import cv2
import numpy as np

def find_and_crop_largest_circle(image_path, output_path):
    # 加载图像
    image = cv2.imread(image_path)
    # 转换为灰度图像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 进行模糊处理
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # 检测边缘
    edges = cv2.Canny(blurred, 50, 150)
    # 执行霍夫圆变换
    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 1, minDist=100, param1=50, param2=30, minRadius=10, maxRadius=100)
    # 确保至少找到一个圆
    if circles is not None:
        # 将结果转换为整数
        circles = np.round(circles[0, :]).astype("int")
        # 找到最大的圆
        max_radius = 0
        max_circle = None
        for (x, y, r) in circles:
            if r > max_radius:
                max_radius = r
                max_circle = (x, y, r)

        if max_circle is not None:
            # 创建一个空白的掩膜图像
            mask = np.zeros(gray.shape, dtype=np.uint8)

            # 在掩膜图像上绘制最大的圆
            (x, y, r) = max_circle
            new_radius = int(r * 0.9)
            cv2.circle(mask, (x, y), new_radius, (255), -1)

            # 使用掩膜图像来提取原始图像中的圆形区域
            cropped_image = cv2.bitwise_and(image, image, mask=mask)

            # 保存裁剪后的图像
            cv2.imwrite(output_path, cropped_image)
            print("Cropped image saved successfully.")
        else:
            print("No circles found in the image.")
    else:
        print("No circles found in the image.")


# 处理文件夹中的所有图像
folder_path = '/Users/hanbinliu/Desktop/0718bog/透明玻璃垫颜色'

output_folder = '/Users/hanbinliu/Desktop/0718bog/move_crop'
os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(folder_path):
    if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".bmp"):
        input_path = os.path.join(folder_path, filename)
        output_path = os.path.join(output_folder, filename)

        find_and_crop_largest_circle(input_path, output_path)
