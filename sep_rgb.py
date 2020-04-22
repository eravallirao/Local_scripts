import argparse
import dlib
import cv2
import os
import imageio
import csv
import sys

count =0

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str, default="",
	help="path to input image file")
ap.add_argument("-cc", "--csvss", type=str, default="",
	help="path to input image file")
args = vars(ap.parse_args())


images = args["image"]
csvs = args["csvss"]
for fl in os.listdir(images):
	if fl == ".DS_Store" or fl == "_DS_Store":
		print ("sorry")
		print(fl)
	else:
		#print(fl)
		images2 = os.path.join(images,fl)
		# frame = imread(images2)
		pic = imageio.imread(images2)
		# r=frame(:,:,1);
		# g=frame(:,:,2);
		# b=frame(:,:,3);
		# print(r g b)
		# print('Shape of the image : {}'.format(pic.shape))
		# print('Maximum RGB value in this image {}'.format(pic.max()))
		print(images2)
		print('Minimum RGB value in this image {}'.format(pic.min()))

		with open(str(csvs), 'a', newline='') as csvfile:
			fieldnames = ['imagename', 'probability_1']
			# writer.writeheader()
			writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
			writer.writerow({'imagename': fl, 'probability_1': format(pic.min())})
		# print("zipak.............")
		# fl2 = fl.split(".")[0]
		# img = cv2.imread('example.jpg')
		# img_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
		# # cv2.imwrite( str(fl) + ".jpg",img_hsv)
		# cv2.imwrite( str(images) + "/" + str(fl2) + ".jpg",img_hsv)

			

		