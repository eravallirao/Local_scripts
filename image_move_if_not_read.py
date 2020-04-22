import cv2
import numpy as np
import argparse
import os
import shutil


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str, default="",
	help="path to input folder")
ap.add_argument("-m", "--move", type=str, default="",
	help="path to destination folder")

args = vars(ap.parse_args())


images = args["image"]
move = args["move"]

# path=os.getcwd()

for fl in os.listdir(images):
	print(fl)
	if fl == ".DS_Store" or fl == "_DS_Store":
		print(fl)
		print("stupid files")

	else:
		try:
			images2 = os.path.join(images,fl)	
			frame = cv2.imread(images2)
			# detector = dlib.get_frontal_face_detector()
			height, width, channels = frame.shape
			print(height, width, channels)
			# if height != 224 or width != 224:
			# 	shutil.move(images2, move)
			# else:
			# 	continue
		except:
			shutil.move(images2, move)