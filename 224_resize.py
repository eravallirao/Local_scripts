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
		try:
			# print(fl)	
			frame = cv2.imread(images2)
			# dst = cv2.fastNlMeansDenoisingColored(frame,None,5,5,7,21)
			# gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			# 3.7649328e-01
			# angle = -3.7
			# rotated = imutils.rotate(frame, angle)
			height, width, channels = frame.shape
			# print (height, width, channels)
			# if height < 620 and width < 900:
			# 	print(fl)
			# cropped = frame[7:35, 16:35]
			# fh = 1000/int(height)
			# fw = 1500/int(width)
			# print(fh*height)
			# print(fw*width)
			# # detector = dlib.get_frontal_face_detector()
			# # try:			
			fl2 = fl.split(".")[0]
			# # # cv2.imwrite( "cropped" + ".jpg",face)
			cv_interpolation = cv2.INTER_LANCZOS4
			# # # # cropped = cv2.resize(frame,(224,224))
			# # img_scaled_1 = cv2.resize(frame,None,fx=1.088, fy=1.087, interpolation = cv2.INTER_CUBIC)
			# # img_scaled_2 = cv2.resize(frame,None,fx=1.155, fy=1.154, interpolation = cv2.INTER_CUBIC)
			# # img_scaled_3 = cv2.resize(frame,None,fx=1.311, fy=1.309, interpolation = cv2.INTER_CUBIC)
			# # img_scaled_4 = cv2.resize(frame,None,fx=1.511, fy=1.509, interpolation = cv2.INTER_CUBIC)
			# img_scaled_1 = cv2.resize(frame,None,fx=1.3, fy=1.087, interpolation = cv2.INTER_CUBIC)
			# img_scaled_2 = cv2.resize(frame,None,fx=1.55, fy=1.154, interpolation = cv2.INTER_CUBIC)
			# img_scaled_3 = cv2.resize(frame,None,fx=1.82, fy=1.309, interpolation = cv2.INTER_CUBIC)
			# img_scaled_4 = cv2.resize(frame,None,fx=2, fy=1.509, interpolation = cv2.INTER_CUBIC)
			cropped = cv2.resize(frame, dsize=(414, 286), interpolation=cv_interpolation)
			# # img = cv2.bilateralFilter(frame,9,75,75)
			# # img = cv2.medianBlur(frame, 5)
			
			# # img_scaled = cv2.resize(frame,None,fx=fw, fy=fh, interpolation = cv2.INTER_AREA)
			cv2.imwrite( str(fl2) + "_resized.png",cropped)
			# cv2.imwrite( str(images) + "/" + str(fl2) + "_up_13" + ".png",img_scaled_1)
			# cv2.imwrite( str(images) + "/" + str(fl2) + "_up_155" + ".png",img_scaled_2)
			# cv2.imwrite( str(images) + "/" + str(fl2) + "_up_182" + ".png",img_scaled_3)
			# cv2.imwrite( str(images) + "/" + str(fl2) + "_up_20" + ".png",img_scaled_4)
		except:
			print(fl)
			continue





			

		
