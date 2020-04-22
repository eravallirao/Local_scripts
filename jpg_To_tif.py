import argparse
import dlib
import cv2
import os
import imageio
import csv
import sys
from PIL import Image

count =0

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str, default="",
	help="path to input image file")
# ap.add_argument("-cc", "--csvss", type=str, default="",
# 	help="path to input image file")
args = vars(ap.parse_args())


images = args["image"]
# csvs = args["csvss"]
for fl in os.listdir(images):
	if fl == ".DS_Store" or fl == "_DS_Store":
		print ("sorry")
		print(fl)
	else:
		#print(fl)
		try:
			images2 = os.path.join(images,fl)
			print(images2)
			fl2 = fl.split(".")[0]
			# frame = imread(images2)
			# pic = imageio.imread(images2)
			imgg = Image.open(images2)
			imgg.save(str(fl2) + ".tif")
		except:
			print("failing *********************************************")
			continue

			

		