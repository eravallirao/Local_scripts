import cv2
import numpy as np

# load image as YUV (or YCbCR) and select Y (intensity)
# or convert to grayscale, which should be the same.
# Alternately, use L (luminance) from LAB.
img = cv2.imread("11p.png")
Y = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)[:,:,0]

# compute min and max of Y
Nu_min = np.min(Y)
Nu_max = np.max(Y)
Nu = Nu_max - Nu_min
print(Nu)
p = int(Nu_max)
q = int(Nu_min)
De = p + q
print(De)

# compute contrast
contrast = Nu/De
print(Nu_min,Nu_max,contrast)