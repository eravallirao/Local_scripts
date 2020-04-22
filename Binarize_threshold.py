import os
import argparse
import dlib
import cv2
import imutils
import csv

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
		print(fl)	
		frame = cv2.imread(images2)
		frame_2 = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
		image = cv2.threshold(frame_2,128,255,cv2.THRESH_OTSU)[1]	
		fl2 = fl.split(".")[0]
		cv2.imwrite( str(images) + "/" + str(fl2) + ".png",image)





			

		
