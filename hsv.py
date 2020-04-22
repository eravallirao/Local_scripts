import argparse
import dlib
import cv2
import os

count =0

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str, default="",
	help="path to input image file")
args = vars(ap.parse_args())


images = args["image"]
for fl in os.listdir(images):
	if fl == ".DS_Store" or fl == "_DS_Store":
		print ("sorry")
		print(fl)
	else:
		#print(fl)
		images2 = os.path.join(images,fl)
		frame = cv2.imread(images2)
		fl2 = fl.split(".")[0]
		# img = cv2.imread('example.jpg')
		img_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
		# cv2.imwrite( str(fl) + ".jpg",img_hsv)
		cv2.imwrite( str(images) + "/" + str(fl2) + ".jpg",img_hsv)

			

		