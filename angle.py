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
		# 3.7649328e-01
		# angle = -3.661935575519803
		angle = 0.58675
		rotated = imutils.rotate(frame, angle)
		# height, width, channels = frame.shape
		# print (height, width, channels)
		# # detector = dlib.get_frontal_face_detector()
		# # try:			
		fl2 = fl.split(".")[0]
		# # cv2.imwrite( "cropped" + ".jpg",face)
		# # cv_interpolation = cv2.INTER_LANCZOS4
		# # cropped = cv2.resize(frame,(224,224))
		# img_scaled = cv2.resize(frame,None,fx=2, fy=2, interpolation = cv2.INTER_CUBIC)
		# cropped = cv2.resize(frame, dsize=(224, 224), interpolation=cv_interpolation)
		# cv2.imwrite( str(fl) + ".jpg",cropped)
		cv2.imwrite( str(images) + "/" + str(fl2) + ".jpg",rotated)





			

		
