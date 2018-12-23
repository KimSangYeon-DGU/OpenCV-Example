import cv2
import numpy as np

def update_M(M, seam):
    M_height, M_width = M.shape[:2]
    new_M = np.zeros((M_height, M_width - 1), dtype=np.uint8)
    for s in seam:
        new_M[s[0],:] = np.delete(M[s[0],:], s[1])

    return new_M

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
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
    e = abs(sobel_x) + abs(sobel_y)
    e_height, e_width = e.shape[:2]

    # M Calculation
    for i in range(e_height):
        for j in range(e_width):
            m_idx = []
            m_idx.append((i, j-1)); m_idx.append((i, j)); m_idx.append((i, j+1))
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