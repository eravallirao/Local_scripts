import math
import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('1_gray.jpg').astype(np.float32) / 255

kernel = cv2.getGaborKernel((21, 21), 5, 1, 10, 1, 0, cv2.CV_32F)
cv2.imwrite("zz_ke_1.jpg", kernel)
kernel /= math.sqrt((kernel * kernel).sum())
cv2.imwrite("zz_ke_2.jpg", kernel)

filtered = cv2.filter2D(image, -1, kernel)

cv2.imwrite("zz_gabo.jpg", filtered)
