import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import numpy as np


lower = np.array([0, 135, 85], dtype = "uint8")
upper = np.array([240, 165, 115], dtype = "uint8")

def segment(img):
    new_img = img.copy()
    YCrCb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    mask = cv2.inRange(YCrCb, lower, upper)
    idx = (mask == 0)
    new_img[idx] = 0
    new_img = cv2.medianBlur(new_img, 3)
    plt.imshow(new_img)
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
    img = mpimg.imread("kim.jpg")
    segment(img)
    #histogram(img)