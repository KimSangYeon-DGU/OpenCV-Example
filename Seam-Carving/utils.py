import cv2
import numpy as np

def check_range(m_idx, h, w):
    ret = [1, 1, 1] # [m1, m2, m3]

    # Check m1's range
    if m_idx[0][0] < 0 or m_idx[0][0] >= h:
        ret[0] = 0
    if m_idx[0][1] < 0 or m_idx[0][1] >= w:
        ret[0] = 0

    # Check m2's range
    if m_idx[1][0] < 0 or m_idx[1][0] >= h:
        ret[1] = 0
    if m_idx[1][1] < 0 or m_idx[1][1] >= w:
        ret[1] = 0    
    
    # Check m3's range
    if m_idx[2][0] < 0 or m_idx[2][0] >= h:
        ret[2] = 0
    if m_idx[2][1] < 0 or m_idx[2][1] >= w:
        ret[2] = 0
    
    return ret

def get_energy_matrix(img):
    M = np.zeros(img.shape[:2])
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    b, g, r = cv2.split(img)
    e = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)) + np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3))
    e_height, e_width = e.shape[:2]

    # Initialization for M
    for i in range(e_width):
        M[0, i] = e[0, i]

    # M Calculation
    for i in range(1, e_height):
        for j in range(e_width):
            m_idx = []
            m_idx.append((i-1, j-1)); m_idx.append((i-1, j)); m_idx.append((i-1, j+1))
            range_ret = check_range(m_idx, e_height, e_width)
            
            if range_ret[0] == 1:
                m1 = M[m_idx[0]]
            if range_ret[1] == 1:
                m2 = M[m_idx[1]]
            if range_ret[2] == 1:
                m3 = M[m_idx[2]]
            
            if range_ret[0] == 0 and range_ret[1] == 0 and range_ret[2] == 0:
                M[i,j] = e[i,j]
            elif range_ret[0] == 0 and range_ret[1] == 0 and range_ret[2] == 1:
                M[i,j] = e[i,j] + m3
            elif range_ret[0] == 0 and range_ret[1] == 1 and range_ret[2] == 0:
                M[i,j] = e[i,j] + m2
            elif range_ret[0] == 0 and range_ret[1] == 1 and range_ret[2] == 1:
                M[i,j] = e[i,j] + min(m2, m3)
            elif range_ret[0] == 1 and range_ret[1] == 0 and range_ret[2] == 0:
                M[i,j] = e[i,j] + m1
            elif range_ret[0] == 1 and range_ret[1] == 0 and range_ret[2] == 1:
                M[i,j] = e[i,j] + min(m1, m3)
            elif range_ret[0] == 1 and range_ret[1] == 1 and range_ret[2] == 0:
                M[i,j] = e[i,j] + min(m1, m2)
            elif range_ret[0] == 1 and range_ret[1] == 1 and range_ret[2] == 1:
                M[i,j] = e[i,j] + min(m1, m2, m3)
    return M