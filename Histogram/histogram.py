import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import numpy as np


lower = np.array([0, 0, 0], dtype = "uint8")
upper = np.array([255, 255, 255], dtype = "uint8")

def segment(img):
    new_img = img.copy()
    YCrCb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    mask = cv2.inRange(YCrCb, lower, upper)
    idx = (mask == 0)
    new_img[idx] = 0
    new_img = cv2.medianBlur(new_img, 3)
    plt.imshow(new_img)
    cv2.imwrite("yoo_seg.jpg",cv2.cvtColor(new_img, cv2.COLOR_RGB2BGR))
    plt.show()

def histogram(img):
    new_img = img.copy()
    YCrCb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    Y, Cr, Cb = cv2.split(YCrCb)
    
    fig = plt.figure(figsize=(16,9))

    sub1 = fig.add_subplot(131)
    sub1.set_title("Y")

    sub2 = fig.add_subplot(132)
    sub2.set_title("Cr")

    sub3 = fig.add_subplot(133)
    sub3.set_title("Cb")
    
    sub1.hist(Y.ravel(), bins=256)
    sub2.hist(Cr.ravel(), bins=256)
    sub3.hist(Cb.ravel(), bins=256)

    plt.show()
if __name__ =="__main__":
    img = mpimg.imread("son.jpg")
    #segment(img)
    histogram(img)