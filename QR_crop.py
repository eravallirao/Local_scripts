# USAGE
# python barcode_scanner_image.py --image barcode_example.png

# import the necessary packages
from pyzbar import pyzbar
import argparse
import cv2
import os

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
		image = cv2.imread(images2)
		# load the input image
		# image = cv2.imread(args["image"])

		# find the barcodes in the image and decode each of the barcodes
		barcodes = pyzbar.decode(image)

		# loop over the detected barcodes
		for barcode in barcodes:
			# extract the bounding box location of the barcode and draw the
			# bounding box surrounding the barcode on the image
			(x, y, w, h) = barcode.rect
			print(x,y,w,h)
			x = x - 20
			y = y - 20
			w = w + 40
			h = h + 40
			x1 = x + w
			y1 = y + h

			img_new = image[y:y1, x:x1]
			fl2 = fl.split(".")[0]
			cv2.imwrite( str(images) + "/" + str(fl2) + ".png",img_new)
		# 	cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

		# 	# the barcode data is a bytes object so if we want to draw it on
		# 	# our output image we need to convert it to a string first
		# 	barcodeData = barcode.data.decode("utf-8")
		# 	barcodeType = barcode.type

		# 	# draw the barcode data and barcode type on the image
		# 	text = "{} ({})".format(barcodeData, barcodeType)
		# 	cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
		# 		0.5, (0, 0, 255), 2)

		# # show the output image
		# cv2.imshow("Image", image)
		# cv2.waitKey(0)