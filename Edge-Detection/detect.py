import cv2
import numpy as np
from scipy import signal

sobel_x = np.array([[+1, 0, -1],
                    [+2, 0, -2],
                    [+1, 0, -1]])

sobel_y = np.array([[+1, +2, +1],
                    [ 0,  0,  0],
                    [-1, -2, -1]])


def detect():
    img = cv2.imread("Lenna.jpg")
    img_height, img_width = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # OpenCV Sobel library
    s_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    s_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_origin = np.add(s_x, s_y)
    print(np.max(sobel_origin), np.min(sobel_origin))
    cv2.imwrite("sobel_origin.jpg", sobel_origin)

    # Sobel Algorithm implementation
    grad_x = signal.convolve2d(gray, sobel_x, boundary='symm', mode='same')
    grad_y = signal.convolve2d(gray, sobel_y, boundary='symm', mode='same')
    edge = np.add(grad_x, grad_y)
    cv2.imwrite("my_sobel.jpg", edge)
    print(np.max(edge), np.min(edge))


if __name__ == "__main__":
    detect()