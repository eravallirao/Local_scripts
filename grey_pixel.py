
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


		frame = cv2.imread(images2)
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		fl2 = fl.split(".")[0]
		# img = Image.open('res96.jpg').convert('L')
		
		# img = Image.open(images2).convert('LA')
		# img2 = Image.open(img).convert('L')
		# data = list(img2.getdata())
		# print(data)
		cv2.imwrite( str(images) + "/" + str(fl2) + ".jpg",gray)
		# img.save('greyscale.png')




			

		
