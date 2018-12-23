import cv2
import numpy as np
from utils import *

def seam_carving(img, img_h, img_w):
    new_img = img.copy()
    img_height, img_width = new_img.shape[:2]
    M = get_energy_matrix(new_img)
    cnt = img_width - img_w
    for c in range(cnt):
        # a vertical seam selection
        min_val = 2e32
        seam = []
    
        # Find starting point
        for j in range(img_width-c):
            if(M[0, j] < min_val):
                min_val = M[0, j]
                min_idx = j
    
        seam.append((0, min_idx))

        # Find Seam
        for i in range(1, img_height):
            j = seam[i-1][1]
            min_val = 2e32
            min_idx = 0
            m_idx = []
            m_idx.append((i, j-1)); m_idx.append((i, j)); m_idx.append((i, j+1))
            range_ret = check_range(m_idx, img_height, img_width-c)

            for t in range(3):
                if range_ret[t] == 0:
                    continue
                if( M[m_idx[t]] < min_val):
                    min_val = M[m_idx[t]]
                    min_idx = m_idx[t][1]

            seam.append((i, min_idx))

        # Show seam    
        for s in seam:
            new_img[s] = [0, 0, 255]

        cv2.imshow("Seam Carving", new_img)
        key = cv2.waitKey(30)
        if key == 27:
            break
        
        # Delete seam
        resized_img = np.zeros((img_height,  img_width-c-1, 3), dtype=np.uint8)
        for s in seam:
            resized_img[s[0],:,0] = np.delete(new_img[s[0],:,0], s[1])
            resized_img[s[0],:,1] = np.delete(new_img[s[0],:,1], s[1])
            resized_img[s[0],:,2] = np.delete(new_img[s[0],:,2], s[1])

        # Update M map
        M = update_M(M, seam)

        new_img = resized_img.copy()

    return resized_img

if __name__ == "__main__":
    img = cv2.imread("Lenna.jpg")
    cv2.namedWindow("Seam Carving")

    img_height, img_width = img.shape[:2]
    resized_img = seam_carving(img, img_height, img_width-100)
    while True:
        cv2.imshow("Seam Carving", resized_img)
        key = cv2.waitKey(30)
        if key == 27:
            break
    cv2.destroyAllWindows()