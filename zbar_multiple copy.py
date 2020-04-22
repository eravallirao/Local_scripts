import os
import argparse
# import dlib
import cv2
# import imutils
import csv
from pyzbar.pyzbar import decode
from PIL import Image
import shutil

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str, default="",
	help="path to input image file")
args = vars(ap.parse_args())


images = args["image"]
dest1 = "/Users/sukshi/Downloads/images_qr_not_read/"

# cv2.imshow('Scaling - Linear Interpolation', img_scaled) img_scaled = cv2.resize(img,None,fx=1.2, fy=1.2, interpolation = cv2.INTER_CUBIC)
i = 0

for fl in os.listdir(images):
	#print(fl)
	if fl == ".DS_Store" or fl == "_DS_Store":
		#print(fl)
		print("stupid files")
	else:
		images2 = os.path.join(images,fl)
		try:	
			frame = cv2.imread(images2)
			print(fl)
			if decode(frame) == []:
				j = 0
				os.rename(images2, "/Users/sukshi/Downloads/img_q_n_r_d_5/" + str(fl))
				# shutil.move(images+'/'+fl, dest1)
			else:
				i = i + 1
				print(fl)
				print(decode(frame))
				print(i)
				with open("fisrst_round_10k.csv", 'a', newline='') as csvfile:
					fieldnames = ['filename', 'width']
					# writer.writeheader()
					writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
					writer.writerow({'filename': fl, 'width': decode(frame)})
		except:
			print("image not decoded")
			# print(exception)
print(i)




			

		
