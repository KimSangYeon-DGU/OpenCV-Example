import cv2
import numpy as np
from scipy import signal

sobel_x = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]])

sobel_y = np.array([[-1, -2, -1],
                    [ 0,  0,  0],
                    [+1, +2, +1]])

scharr_x = np.array([[ -3, 0, +3],
                     [-10, 0, +10],
                     [ -3, 0, +3]]) 

scharr_y = np.array([[ -3, -10, -3],
                     [  0,   0,  0],
                     [ +3, +10, +3]]) 

def detect():
    img = cv2.imread("Lenna.jpg")
    img_height, img_width = img.shape[:2]
    
    b, g, r = cv2.split(img)

    grad_x_b = signal.convolve2d(b, sobel_x, boundary='symm', mode='same')
    grad_x_g = signal.convolve2d(g, sobel_x, boundary='symm', mode='same')
    grad_x_r = signal.convolve2d(r, sobel_x, boundary='symm', mode='same')
    edge_x = cv2.merge((grad_x_b, grad_x_g, grad_x_r))
    
    grad_y_b = signal.convolve2d(b, sobel_y, boundary='symm', mode='same')
    grad_y_g = signal.convolve2d(g, sobel_y, boundary='symm', mode='same')
    grad_y_r = signal.convolve2d(r, sobel_y, boundary='symm', mode='same')
    edge_y = cv2.merge((grad_y_b, grad_y_g, grad_y_r))

    edge = edge_x + edge_y
    cv2.imwrite("sobel.jpg", edge)

if __name__ == "__main__":
    detect()