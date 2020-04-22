# python puru_xcallids.py -i folder_name >output.txt

import argparse
import json
import os
import cv2

ap = argparse.ArgumentParser()

ap.add_argument("-i", "--image", type=str, default="",
	help="path to input image folder")

args = vars(ap.parse_args())

images = args["image"]

count = 0
for fl in os.listdir(images):
	#print("messi")
	if fl == ".DS_Store" or fl == "_DS_Store":
		count += 1
		#print(fl)

	else:
		fl = fl[:-10]
		print(fl)
		

			
				




			