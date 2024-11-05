import cv2
import numpy as np


def crop_rect(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w >= 300 and w <= 400 and h >= 300 and h <= 400:
            break
    img = img[y:y + h, x:x + w, :]
    return img


class ImageProcessor:
    def __init__(self, image):
        self.image = image

    def process_image(self):
        self.image = crop_rect(self.image)
        # roi区域
        # x, y, w, h = 1089, 429, 848, 820
        # x, y, w, h = 1200, 550, 600, 700
        # self.image = self.image[y:y + h, x:x + w]

        # 图片锐化处理
        sharpen_kernel = np.array([[-1, -1, -1],
                                   [-1, 25, -1],
                                   [-1, -1, -1]])
        # 使用卷积操作来锐化图像
        image = cv2.filter2D(self.image, -1, sharpen_kernel)
        # 高斯模糊
        image = cv2.GaussianBlur(image, (1, 1), 0)
        edges = cv2.Canny(image, 50, 150, apertureSize=3)

        circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=3, minDist=200, param1=100, param2=50,
                                   minRadius=20, maxRadius=200)

        # 可视化circles
        # if circles is not None:
        #     circles = np.uint16(np.around(circles))
        #     for i in circles[0, :]:
        #         # draw the outer circle
        #         cv2.circle(self.image, (i[0], i[1]), i[2], (0, 255, 0), 2)
        #         # draw the center of the circle
        #         cv2.circle(self.image, (i[0], i[1]), 2, (0, 0, 255), 3)
        #     cv2.imwrite(r'C:\Users\lhb\Desktop\cicrle.jpg', self.image)

        if circles is not None:
            circles = np.uint16(np.around(circles))
            distances = np.sqrt(np.power(circles[0][:, 0] - image.shape[1] / 2, 2) +
                                np.power(circles[0][:, 1] - image.shape[0] / 2, 2))
            min_distance_idx = np.argmin(distances)
            max_circle = circles[0][min_distance_idx]
            center = (max_circle[0], max_circle[1])

            # # 统计黑色像素百分比
            # cv2.circle(image, (max_circle[0], max_circle[1]), max_circle[2], (0, 0, 0), 1)
            # mask = np.zeros(image.shape[:2], dtype=np.uint8)
            # cv2.circle(mask, (max_circle[0], max_circle[1]), max_circle[2], 255, -1)
            # # 白色
            # white_pixels_mask = cv2.inRange(image, (200, 200, 200), (255, 255, 255))
            # white_pixels_mask = cv2.bitwise_and(white_pixels_mask, white_pixels_mask, mask=mask)
            # num_white_pixels = cv2.countNonZero(white_pixels_mask)
            # # 黑色
            # black_pixels_mask = cv2.inRange(image, (0, 0, 0), (1, 1, 1))
            # black_pixels_mask = cv2.bitwise_and(black_pixels_mask, black_pixels_mask, mask=mask)
            # num_black_pixels = cv2.countNonZero(black_pixels_mask)
            # # 黑色像素点占比
            # black_ratio = num_black_pixels / (num_black_pixels + num_white_pixels)

            # 计算位移，将检测到的圆移到图像中心
            center_offset = (image.shape[1] // 2 - center[0], image.shape[0] // 2 - center[1])
            M = np.float32([[1, 0, center_offset[0]], [0, 1, center_offset[1]]])
            self.image = cv2.warpAffine(self.image, M, (self.image.shape[1], self.image.shape[0]))

        mask = np.zeros_like(self.image)
        radius = 153 - 25
        cv2.circle(mask, (self.image.shape[1] // 2, self.image.shape[0] // 2), radius, (255, 255, 255), -1)
        masked_image = cv2.bitwise_and(self.image, mask)
        # cv2.imwrite('C:/Users/ALIENWARE/Desktop/abnormal.png', masked_image)

        return masked_image


if __name__ == '__main__':
    image_path = "C:/Users/Administrator/Desktop/borgtest/20231102-143000.jpg"
    image = cv2.imread(image_path)
    processor = ImageProcessor(image)
    processor.process_image()