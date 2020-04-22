import os
import argparse
import dlib
import cv2
import imutils
import csv
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str, default="",
	help="path to input image file")
args = vars(ap.parse_args())


images = args["image"]	


for fl in os.listdir(images):
	#print(fl)
	if fl == ".DS_Store" or fl == "_DS_Store":
		print(fl)
		# print("stupid files")

	else:
		images2 = os.path.join(images,fl)
		print(images2)
		fl2 = fl.split(".")[0]
		bgr = cv2.imread(images2)
		# # ret,thresh_img = cv2.threshold(bgr,100,255,cv2.THRESH_BINARY)
		# # cv2.imwrite("1st_binary.jpg", thresh_img)
		# lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
		# gridsize = 8
		# lab_planes = cv2.split(lab)
		# clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(gridsize,gridsize))
		# lab_planes[0] = clahe.apply(lab_planes[0])
		# lab = cv2.merge(lab_planes)
		# bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
		ret,thresh_img = cv2.threshold(bgr,60,255,cv2.THRESH_BINARY)
		cv2.imwrite( str(images) + "/" + str(fl2) + ".jpg",thresh_img)
		# img.save('greyscale.png')




			

		
