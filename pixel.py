import numpy as np
from PIL import Image

img = Image.open('95d7a2f733d0d3d7ec3eca0f4250a51f.png').convert('RGBA')
arr = np.array(img)

# record the original shape
shape = arr.shape

# make a 1-dimensional view of arr
flat_arr = arr.ravel()

# convert it to a matrix
vector = np.matrix(flat_arr)

# do something to the vector
vector[:,::10] = 128

# reform a numpy array of the original shape
arr2 = np.asarray(vector).reshape(shape)
print(arr2)

# make a PIL image
# img2 = Image.fromarray(arr2, 'RGBA')
# img2.show()
