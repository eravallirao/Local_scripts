# Parameters
# ----------
# image : ndarray
#     Input image data. Will be converted to float.
# mode : str
#     One of the following strings, selecting the type of noise to add:

#     'gauss'     Gaussian-distributed additive noise.
#     'poisson'   Poisson-distributed noise generated from the data.
#     's&p'       Replaces random pixels with 0 or 1.
#     'speckle'   Multiplicative noise using out = image + n*image,where
#                 n is uniform noise with specified mean & variance.


import numpy as np
import os
import cv2

import argparse

def noisy(noise_typ,image):
  if noise_typ == "gauss":
    row,col,ch= image.shape
    mean = 0
    var = 0.8
    sigma = var**0.9
    gauss = np.random.normal(mean,sigma,(row,col,ch))
    gauss = gauss.reshape(row,col,ch)
    noisy = image + gauss
    return noisy
  elif noise_typ == "s&p":
    row,col,ch = image.shape
    s_vs_p = 0.5
    amount = 0.004
    out = np.copy(image)
    # Salt mode
    num_salt = np.ceil(amount * image.size * s_vs_p)
    coords = [np.random.randint(0, i - 1, int(num_salt))
            for i in image.shape]
    out[coords] = 1

    # Pepper mode
    num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
    coords = [np.random.randint(0, i - 1, int(num_pepper))
            for i in image.shape]
    out[coords] = 0
    return out
  elif noise_typ == "poisson":
    vals = len(np.unique(image))
    vals = 2 ** np.ceil(np.log2(vals))
    noisy = np.random.poisson(image * vals) / float(vals)
    return noisy
  elif noise_typ =="speckle":
    row,col,ch = image.shape
    gauss = np.random.randn(row,col,ch)
    gauss = gauss.reshape(row,col,ch)        
    noisy = image + image * gauss
    return noisy


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str, default="",
  help="path to input image file")
args = vars(ap.parse_args())


images = args["image"]  

# cv2.imshow('Scaling - Linear Interpolation', img_scaled) img_scaled = cv2.resize(img,None,fx=1.2, fy=1.2, interpolation = cv2.INTER_CUBIC)


for fl in os.listdir(images):
  #print(fl)
  if fl == ".DS_Store" or fl == "_DS_Store":
    #print(fl)
    print("stupid files")

  else:
    images2 = os.path.join(images,fl)
    # print(fl) 
    frame = cv2.imread(images2)
    height, width, channels = frame.shape   
    fl2 = fl.split(".")[0]
    cropped = noisy("gauss",frame)
    cv2.imwrite( str(images) + "/" + str(fl2) + ".jpg",cropped)